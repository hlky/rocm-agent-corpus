#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>
#include <stddef.h>

namespace {

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

__global__ void attention_one_thread_per_output(const float* __restrict__ q,
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
  size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t element_count = static_cast<size_t>(batch_size) * num_heads *
                               query_length * head_dim;
  if (linear >= element_count) {
    return;
  }

  const int value_dim = static_cast<int>(linear % head_dim);
  linear /= head_dim;
  const int query_index = static_cast<int>(linear % query_length);
  linear /= query_length;
  const int head = static_cast<int>(linear % num_heads);
  const int batch = static_cast<int>(linear / num_heads);

  const size_t q_base = ((static_cast<size_t>(batch) * num_heads + head) *
                         query_length + query_index) * head_dim;
  const size_t kv_base = (static_cast<size_t>(batch) * num_heads + head) *
                         key_length * head_dim;
  const size_t output_index = q_base + value_dim;

  float max_score = -FLT_MAX;
  bool has_visible_key = false;
  for (int key_index = 0; key_index < key_length; ++key_index) {
    if (!key_is_visible(query_index, key_index, query_length, key_length, causal)) {
      continue;
    }
    float dot = 0.0f;
    const size_t key_base = kv_base + static_cast<size_t>(key_index) * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q[q_base + dim] * k[key_base + dim];
    }
    max_score = fmaxf(max_score, dot * scale);
    has_visible_key = true;
  }

  if (!has_visible_key) {
    output[output_index] = 0.0f;
    return;
  }

  float denominator = 0.0f;
  float accumulator = 0.0f;
  for (int key_index = 0; key_index < key_length; ++key_index) {
    if (!key_is_visible(query_index, key_index, query_length, key_length, causal)) {
      continue;
    }
    float dot = 0.0f;
    const size_t key_base = kv_base + static_cast<size_t>(key_index) * head_dim;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += q[q_base + dim] * k[key_base + dim];
    }
    const float weight = expf(dot * scale - max_score);
    denominator += weight;
    accumulator += weight * v[key_base + value_dim];
  }

  output[output_index] = accumulator / denominator;
}

}  // namespace

extern "C" void launch_baseline(const float* q,
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
  constexpr int kBlockSize = 128;
  const size_t element_count = static_cast<size_t>(batch_size) * num_heads *
                               query_length * head_dim;
  const int grid = static_cast<int>((element_count + kBlockSize - 1) / kBlockSize);
  attention_one_thread_per_output<<<grid, kBlockSize, 0, stream>>>(
      q, k, v, output, batch_size, num_heads, query_length, key_length, head_dim,
      scale, causal);
}
