#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>

namespace {

__device__ __forceinline__ bool better_pair(float value, int index, float best, int best_index) {
  return value > best || (value == best && index < best_index);
}

__device__ __forceinline__ void insert_top2(float value,
                                            int index,
                                            float& best0,
                                            int& index0,
                                            float& best1,
                                            int& index1) {
  if (better_pair(value, index, best0, index0)) {
    best1 = best0;
    index1 = index0;
    best0 = value;
    index0 = index;
  } else if (index != index0 && better_pair(value, index, best1, index1)) {
    best1 = value;
    index1 = index;
  }
}

__global__ void block_top2_route_kernel(const float* __restrict__ logits,
                                        int* __restrict__ route_indices,
                                        float* __restrict__ route_weights,
                                        int* __restrict__ expert_counts,
                                        int* __restrict__ expert_token_slots,
                                        float* __restrict__ expert_token_weights,
                                        int tokens,
                                        int experts,
                                        int capacity_per_expert) {
  constexpr int kBlockSize = 128;
  __shared__ float shared_best0[kBlockSize];
  __shared__ float shared_best1[kBlockSize];
  __shared__ int shared_index0[kBlockSize];
  __shared__ int shared_index1[kBlockSize];

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  if (token >= tokens) {
    return;
  }

  float best0 = -FLT_MAX;
  float best1 = -FLT_MAX;
  int index0 = 0;
  int index1 = 0;
  const float* row = logits + static_cast<size_t>(token) * experts;
  for (int expert = tid; expert < experts; expert += blockDim.x) {
    insert_top2(row[expert], expert, best0, index0, best1, index1);
  }

  shared_best0[tid] = best0;
  shared_best1[tid] = best1;
  shared_index0[tid] = index0;
  shared_index1[tid] = index1;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      insert_top2(shared_best0[tid + stride], shared_index0[tid + stride],
                  shared_best0[tid], shared_index0[tid], shared_best1[tid], shared_index1[tid]);
      insert_top2(shared_best1[tid + stride], shared_index1[tid + stride],
                  shared_best0[tid], shared_index0[tid], shared_best1[tid], shared_index1[tid]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float max_logit = fmaxf(shared_best0[0], shared_best1[0]);
    const float w0_raw = expf(shared_best0[0] - max_logit);
    const float w1_raw = expf(shared_best1[0] - max_logit);
    const float denom = w0_raw + w1_raw;
    const float weight0 = w0_raw / denom;
    const float weight1 = w1_raw / denom;
    const int expert0 = shared_index0[0];
    const int expert1 = shared_index1[0];

    route_indices[token * 2 + 0] = expert0;
    route_indices[token * 2 + 1] = expert1;
    route_weights[token * 2 + 0] = weight0;
    route_weights[token * 2 + 1] = weight1;

    const int slot0 = atomicAdd(expert_counts + expert0, 1);
    if (slot0 < capacity_per_expert) {
      const int offset = expert0 * capacity_per_expert + slot0;
      expert_token_slots[offset] = token;
      expert_token_weights[offset] = weight0;
    }

    const int slot1 = atomicAdd(expert_counts + expert1, 1);
    if (slot1 < capacity_per_expert) {
      const int offset = expert1 * capacity_per_expert + slot1;
      expert_token_slots[offset] = token;
      expert_token_weights[offset] = weight1;
    }
  }
}

}  // namespace

extern "C" void launch_optimized(const float* logits,
                                 int* route_indices,
                                 float* route_weights,
                                 int* expert_counts,
                                 int* expert_token_slots,
                                 float* expert_token_weights,
                                 int tokens,
                                 int experts,
                                 int capacity_per_expert,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 128;
  block_top2_route_kernel<<<tokens, kBlockSize, 0, stream>>>(
      logits, route_indices, route_weights, expert_counts, expert_token_slots,
      expert_token_weights, tokens, experts, capacity_per_expert);
}
