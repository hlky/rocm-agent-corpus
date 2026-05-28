#include <hip/hip_runtime.h>

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>

namespace {

constexpr int kMaxK = 8;

__device__ __forceinline__ bool better_candidate(float lhs_value,
                                                 int lhs_index,
                                                 float rhs_value,
                                                 int rhs_index) {
  return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index);
}

__device__ __forceinline__ void insert_candidate(float value,
                                                 int index,
                                                 float* values,
                                                 int* indices,
                                                 int k) {
  for (int slot = 0; slot < k; ++slot) {
    if (better_candidate(value, index, values[slot], indices[slot])) {
      for (int move = k - 1; move > slot; --move) {
        values[move] = values[move - 1];
        indices[move] = indices[move - 1];
      }
      values[slot] = value;
      indices[slot] = index;
      return;
    }
  }
}

__device__ void initialize_candidates(float* values, int* indices) {
  for (int i = 0; i < kMaxK; ++i) {
    values[i] = -FLT_MAX;
    indices[i] = INT_MAX;
  }
}

__device__ void write_topk_and_sample(const float* values,
                                      const int* indices,
                                      const float* uniforms,
                                      int* sampled_indices,
                                      int* topk_indices,
                                      float* topk_values,
                                      float* topk_probs,
                                      int row,
                                      int k,
                                      float temperature) {
  const int output_base = row * k;
  const float row_max = values[0];
  float denominator = 0.0f;
  for (int i = 0; i < k; ++i) {
    denominator += expf((values[i] - row_max) / temperature);
  }

  float u = uniforms[row];
  if (u < 0.0f) {
    u = 0.0f;
  } else if (u > 1.0f) {
    u = 1.0f;
  }

  float cumulative = 0.0f;
  int chosen = indices[k - 1];
  for (int i = 0; i < k; ++i) {
    const float probability = expf((values[i] - row_max) / temperature) / denominator;
    topk_indices[output_base + i] = indices[i];
    topk_values[output_base + i] = values[i];
    topk_probs[output_base + i] = probability;
    cumulative += probability;
    if (u <= cumulative && chosen == indices[k - 1]) {
      chosen = indices[i];
    }
  }
  sampled_indices[row] = chosen;
}

__global__ void topk_sampling_one_block_per_row(const float* __restrict__ logits,
                                                const float* __restrict__ uniforms,
                                                int* __restrict__ sampled_indices,
                                                int* __restrict__ topk_indices,
                                                float* __restrict__ topk_values,
                                                float* __restrict__ topk_probs,
                                                int rows,
                                                int vocab_size,
                                                int k,
                                                float temperature) {
  extern __shared__ unsigned char shared_raw[];
  float* shared_values = reinterpret_cast<float*>(shared_raw);
  int* shared_indices = reinterpret_cast<int*>(shared_values + blockDim.x * kMaxK);

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows || k <= 0 || k > kMaxK || vocab_size < k || temperature <= 0.0f) {
    return;
  }

  float local_values[kMaxK];
  int local_indices[kMaxK];
  initialize_candidates(local_values, local_indices);

  const int row_base = row * vocab_size;
  for (int col = tid; col < vocab_size; col += blockDim.x) {
    insert_candidate(logits[row_base + col], col, local_values, local_indices, k);
  }

  const int shared_base = tid * kMaxK;
  for (int i = 0; i < k; ++i) {
    shared_values[shared_base + i] = local_values[i];
    shared_indices[shared_base + i] = local_indices[i];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float merged_values[kMaxK];
      int merged_indices[kMaxK];
      initialize_candidates(merged_values, merged_indices);

      const int left_base = tid * kMaxK;
      const int right_base = (tid + stride) * kMaxK;
      for (int i = 0; i < k; ++i) {
        if (shared_indices[left_base + i] != INT_MAX) {
          insert_candidate(shared_values[left_base + i],
                           shared_indices[left_base + i],
                           merged_values,
                           merged_indices,
                           k);
        }
        if (shared_indices[right_base + i] != INT_MAX) {
          insert_candidate(shared_values[right_base + i],
                           shared_indices[right_base + i],
                           merged_values,
                           merged_indices,
                           k);
        }
      }
      for (int i = 0; i < k; ++i) {
        shared_values[left_base + i] = merged_values[i];
        shared_indices[left_base + i] = merged_indices[i];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    write_topk_and_sample(shared_values,
                          shared_indices,
                          uniforms,
                          sampled_indices,
                          topk_indices,
                          topk_values,
                          topk_probs,
                          row,
                          k,
                          temperature);
  }
}

}  // namespace

extern "C" void launch_optimized(const float* logits,
                                 const float* uniforms,
                                 int* sampled_indices,
                                 int* topk_indices,
                                 float* topk_values,
                                 float* topk_probs,
                                 int rows,
                                 int vocab_size,
                                 int k,
                                 float temperature,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const size_t shared_bytes =
      static_cast<size_t>(kBlockSize) * kMaxK * (sizeof(float) + sizeof(int));
  topk_sampling_one_block_per_row<<<rows, kBlockSize, shared_bytes, stream>>>(logits,
                                                                             uniforms,
                                                                             sampled_indices,
                                                                             topk_indices,
                                                                             topk_values,
                                                                             topk_probs,
                                                                             rows,
                                                                             vocab_size,
                                                                             k,
                                                                             temperature);
}
