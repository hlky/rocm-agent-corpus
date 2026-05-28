#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>
#include <stddef.h>

namespace {

constexpr int kBlockSize = 128;
constexpr int kKeyTile = 32;

__device__ __forceinline__ int causal_query_offset(int query_length, int key_length) {
  const int delta = key_length - query_length;
  return delta > 0 ? delta : 0;
}

__device__ __forceinline__ bool key_is_visible(int query_index,
                                               int key_index,
                                               int query_length,
                                               int key_length,
                                               int causal) {
  if (!causal) {
    return true;
  }
  const int absolute_query = query_index + causal_query_offset(query_length, key_length);
  return key_index <= absolute_query;
}

__global__ void attention_online_tiled(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v,
                                       float* __restrict__ output,
                                       int batch_size,
                                       int num_heads,
                                       int query_length,
                                       int key_length,
                                       int head_dim,
                                       float scale,
                                       int causal) {
  extern __shared__ float shared[];

  float* tile_k = shared;
  float* tile_v = tile_k + kKeyTile * head_dim;
  float* scores = tile_v + kKeyTile * head_dim;
  float* reduction = scores + kKeyTile;
  float* output_accumulator = reduction + blockDim.x;
  float* scalars = output_accumulator + head_dim;

  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  const int query_index = row % query_length;
  const int head = (row / query_length) % num_heads;
  const int batch = row / (query_length * num_heads);
  if (batch >= batch_size) {
    return;
  }

  const size_t q_base = ((static_cast<size_t>(batch) * num_heads + head) *
                         query_length + query_index) * head_dim;
  const size_t kv_base = (static_cast<size_t>(batch) * num_heads + head) *
                         key_length * head_dim;

  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    output_accumulator[dim] = 0.0f;
  }
  if (tid == 0) {
    scalars[0] = -FLT_MAX;  // running row max
    scalars[1] = 0.0f;      // running denominator
  }
  __syncthreads();

  for (int tile_start = 0; tile_start < key_length; tile_start += kKeyTile) {
    const int remaining_keys = key_length - tile_start;
    const int tile_count = remaining_keys < kKeyTile ? remaining_keys : kKeyTile;
    const int tile_elements = kKeyTile * head_dim;

    for (int index = tid; index < tile_elements; index += blockDim.x) {
      const int local_key = index / head_dim;
      const int dim = index - local_key * head_dim;
      if (local_key < tile_count) {
        const size_t source_index = kv_base + static_cast<size_t>(tile_start + local_key) *
                                                head_dim + dim;
        tile_k[index] = k[source_index];
        tile_v[index] = v[source_index];
      } else {
        tile_k[index] = 0.0f;
        tile_v[index] = 0.0f;
      }
    }
    __syncthreads();

    for (int local_key = tid; local_key < kKeyTile; local_key += blockDim.x) {
      const int key_index = tile_start + local_key;
      float score = -FLT_MAX;
      if (local_key < tile_count &&
          key_is_visible(query_index, key_index, query_length, key_length, causal)) {
        float dot = 0.0f;
        const int key_offset = local_key * head_dim;
        for (int dim = 0; dim < head_dim; ++dim) {
          dot += q[q_base + dim] * tile_k[key_offset + dim];
        }
        score = dot * scale;
      }
      scores[local_key] = score;
    }
    __syncthreads();

    reduction[tid] = tid < kKeyTile ? scores[tid] : -FLT_MAX;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float old_max = scalars[0];
      const float tile_max = reduction[0];
      const float new_max = fmaxf(old_max, tile_max);
      scalars[2] = new_max;
      scalars[3] = old_max == -FLT_MAX ? 0.0f : expf(old_max - new_max);
    }
    __syncthreads();

    const float new_max = scalars[2];
    if (tid < kKeyTile) {
      const float score = scores[tid];
      scores[tid] = score == -FLT_MAX ? 0.0f : expf(score - new_max);
    }
    __syncthreads();

    reduction[tid] = tid < kKeyTile ? scores[tid] : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      scalars[1] = scalars[1] * scalars[3] + reduction[0];
    }
    __syncthreads();

    const float old_output_scale = scalars[3];
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
      float tile_accumulator = 0.0f;
      for (int local_key = 0; local_key < tile_count; ++local_key) {
        tile_accumulator += scores[local_key] * tile_v[local_key * head_dim + dim];
      }
      output_accumulator[dim] = output_accumulator[dim] * old_output_scale +
                                tile_accumulator;
    }
    __syncthreads();

    if (tid == 0) {
      scalars[0] = scalars[2];
    }
    __syncthreads();
  }

  const float denominator = scalars[1];
  const size_t output_base = q_base;
  for (int dim = tid; dim < head_dim; dim += blockDim.x) {
    output[output_base + dim] = denominator > 0.0f ?
                                output_accumulator[dim] / denominator : 0.0f;
  }
}

}  // namespace

extern "C" void launch_optimized(const float* q,
                                 const float* k,
                                 const float* v,
                                 float* output,
                                 int batch_size,
                                 int num_heads,
                                 int query_length,
                                 int key_length,
                                 int head_dim,
                                 float scale,
                                 int causal,
                                 hipStream_t stream) {
  const int rows = batch_size * num_heads * query_length;
  const size_t shared_floats = 2 * kKeyTile * static_cast<size_t>(head_dim) +
                               kKeyTile + kBlockSize + head_dim + 4;
  attention_online_tiled<<<rows, kBlockSize, shared_floats * sizeof(float), stream>>>(
      q, k, v, output, batch_size, num_heads, query_length, key_length, head_dim,
      scale, causal);
}
