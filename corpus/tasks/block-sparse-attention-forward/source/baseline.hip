#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>
#include <stddef.h>

namespace {

__global__ void dense_block_mask_attention(const float* __restrict__ q,
                                           const float* __restrict__ k,
                                           const float* __restrict__ v,
                                           const int* __restrict__ block_mask,
                                           float* __restrict__ output,
                                           int query_length,
                                           int key_length,
                                           int head_dim,
                                           int block_m,
                                           int block_n,
                                           float scale) {
  size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t total = static_cast<size_t>(query_length) * head_dim;
  if (linear >= total) {
    return;
  }

  const int dim = static_cast<int>(linear % head_dim);
  const int query = static_cast<int>(linear / head_dim);
  const int query_block = query / block_m;
  const int key_blocks = (key_length + block_n - 1) / block_n;

  float row_max = -FLT_MAX;
  bool any = false;
  for (int key = 0; key < key_length; ++key) {
    const int key_block = key / block_n;
    if (block_mask[query_block * key_blocks + key_block] == 0) {
      continue;
    }
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
      dot += q[static_cast<size_t>(query) * head_dim + d] *
             k[static_cast<size_t>(key) * head_dim + d];
    }
    row_max = fmaxf(row_max, dot * scale);
    any = true;
  }

  float denom = 0.0f;
  float acc = 0.0f;
  if (any) {
    for (int key = 0; key < key_length; ++key) {
      const int key_block = key / block_n;
      if (block_mask[query_block * key_blocks + key_block] == 0) {
        continue;
      }
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q[static_cast<size_t>(query) * head_dim + d] *
               k[static_cast<size_t>(key) * head_dim + d];
      }
      const float weight = expf(dot * scale - row_max);
      denom += weight;
      acc += weight * v[static_cast<size_t>(key) * head_dim + dim];
    }
  }

  output[linear] = denom > 0.0f ? acc / denom : 0.0f;
}

}  // namespace

extern "C" void launch_baseline(const float* q,
                                const float* k,
                                const float* v,
                                const int* block_mask,
                                float* output,
                                int query_length,
                                int key_length,
                                int head_dim,
                                int block_m,
                                int block_n,
                                float scale,
                                hipStream_t stream) {
  constexpr int kBlockSize = 128;
  const size_t total = static_cast<size_t>(query_length) * head_dim;
  const int grid = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
  dense_block_mask_attention<<<grid, kBlockSize, 0, stream>>>(
      q, k, v, block_mask, output, query_length, key_length, head_dim, block_m,
      block_n, scale);
}
