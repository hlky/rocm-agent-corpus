#include <hip/hip_runtime.h>

#include <stddef.h>

namespace {

__device__ __forceinline__ void rope_pair(float x0,
                                          float x1,
                                          float c,
                                          float s,
                                          float& y0,
                                          float& y1) {
  y0 = x0 * c - x1 * s;
  y1 = x0 * s + x1 * c;
}

__global__ void fused_rope_kv_cache_kernel(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const float* __restrict__ cos,
                                           const float* __restrict__ sin,
                                           const int* __restrict__ token_positions,
                                           float* __restrict__ q_out,
                                           float* __restrict__ k_cache,
                                           float* __restrict__ v_cache,
                                           int batch_tokens,
                                           int num_heads,
                                           int head_dim) {
  const int half_dim = head_dim / 2;
  const size_t total_pairs = static_cast<size_t>(batch_tokens) * num_heads * half_dim;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total_pairs;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(linear % half_dim);
    const int head = static_cast<int>((linear / half_dim) % num_heads);
    const int token = static_cast<int>(linear / (half_dim * num_heads));
    const int position = token_positions[token];
    const size_t source_base = (static_cast<size_t>(token) * num_heads + head) * head_dim + d;
    const size_t source_mate = source_base + half_dim;
    const size_t cache_base = (static_cast<size_t>(position) * num_heads + head) * head_dim + d;
    const size_t cache_mate = cache_base + half_dim;
    const size_t table = static_cast<size_t>(position) * half_dim + d;
    const float c = cos[table];
    const float s = sin[table];

    float y0;
    float y1;
    rope_pair(q[source_base], q[source_mate], c, s, y0, y1);
    q_out[source_base] = y0;
    q_out[source_mate] = y1;

    rope_pair(k[source_base], k[source_mate], c, s, y0, y1);
    k_cache[cache_base] = y0;
    k_cache[cache_mate] = y1;

    v_cache[cache_base] = v[source_base];
    v_cache[cache_mate] = v[source_mate];
  }
}

}  // namespace

extern "C" void launch_optimized(const float* q,
                                 const float* k,
                                 const float* v,
                                 const float* cos,
                                 const float* sin,
                                 const int* token_positions,
                                 float* q_out,
                                 float* k_cache,
                                 float* v_cache,
                                 int batch_tokens,
                                 int num_heads,
                                 int head_dim,
                                 int max_seq_len,
                                 hipStream_t stream) {
  (void)max_seq_len;
  constexpr int kBlockSize = 256;
  const int half_dim = head_dim / 2;
  const size_t total_pairs = static_cast<size_t>(batch_tokens) * num_heads * half_dim;
  const int grid = static_cast<int>((total_pairs + kBlockSize - 1) / kBlockSize);
  fused_rope_kv_cache_kernel<<<grid, kBlockSize, 0, stream>>>(
      q, k, v, cos, sin, token_positions, q_out, k_cache, v_cache,
      batch_tokens, num_heads, head_dim);
}
