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

__global__ void rotate_qk_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ cos,
                                 const float* __restrict__ sin,
                                 const int* __restrict__ token_positions,
                                 float* __restrict__ q_out,
                                 float* __restrict__ k_rotated,
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
    const size_t base = (static_cast<size_t>(token) * num_heads + head) * head_dim + d;
    const size_t mate = base + half_dim;
    const size_t table = static_cast<size_t>(position) * half_dim + d;

    float y0;
    float y1;
    rope_pair(q[base], q[mate], cos[table], sin[table], y0, y1);
    q_out[base] = y0;
    q_out[mate] = y1;

    rope_pair(k[base], k[mate], cos[table], sin[table], y0, y1);
    k_rotated[base] = y0;
    k_rotated[mate] = y1;
  }
}

__global__ void store_k_cache_kernel(const float* __restrict__ k_rotated,
                                     const int* __restrict__ token_positions,
                                     float* __restrict__ k_cache,
                                     int batch_tokens,
                                     int num_heads,
                                     int head_dim) {
  const size_t total = static_cast<size_t>(batch_tokens) * num_heads * head_dim;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(linear % head_dim);
    const int head = static_cast<int>((linear / head_dim) % num_heads);
    const int token = static_cast<int>(linear / (head_dim * num_heads));
    const int position = token_positions[token];
    const size_t cache = (static_cast<size_t>(position) * num_heads + head) * head_dim + d;
    k_cache[cache] = k_rotated[linear];
  }
}

__global__ void store_v_cache_kernel(const float* __restrict__ v,
                                     const int* __restrict__ token_positions,
                                     float* __restrict__ v_cache,
                                     int batch_tokens,
                                     int num_heads,
                                     int head_dim) {
  const size_t total = static_cast<size_t>(batch_tokens) * num_heads * head_dim;
  for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const int d = static_cast<int>(linear % head_dim);
    const int head = static_cast<int>((linear / head_dim) % num_heads);
    const int token = static_cast<int>(linear / (head_dim * num_heads));
    const int position = token_positions[token];
    const size_t cache = (static_cast<size_t>(position) * num_heads + head) * head_dim + d;
    v_cache[cache] = v[linear];
  }
}

}  // namespace

extern "C" void launch_baseline(const float* q,
                                const float* k,
                                const float* v,
                                const float* cos,
                                const float* sin,
                                const int* token_positions,
                                float* q_out,
                                float* k_rotated,
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
  const size_t pair_total = static_cast<size_t>(batch_tokens) * num_heads * half_dim;
  const size_t value_total = static_cast<size_t>(batch_tokens) * num_heads * head_dim;
  const int pair_grid = static_cast<int>((pair_total + kBlockSize - 1) / kBlockSize);
  const int value_grid = static_cast<int>((value_total + kBlockSize - 1) / kBlockSize);
  rotate_qk_kernel<<<pair_grid, kBlockSize, 0, stream>>>(
      q, k, cos, sin, token_positions, q_out, k_rotated, batch_tokens, num_heads, head_dim);
  store_k_cache_kernel<<<value_grid, kBlockSize, 0, stream>>>(
      k_rotated, token_positions, k_cache, batch_tokens, num_heads, head_dim);
  store_v_cache_kernel<<<value_grid, kBlockSize, 0, stream>>>(
      v, token_positions, v_cache, batch_tokens, num_heads, head_dim);
}
