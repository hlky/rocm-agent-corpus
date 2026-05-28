#include <hip/hip_runtime.h>

#include <float.h>
#include <math.h>
#include <stddef.h>

namespace {

__global__ void listed_block_sparse_attention(const float* __restrict__ q,
                                              const float* __restrict__ k,
                                              const float* __restrict__ v,
                                              const int* __restrict__ block_offsets,
                                              const int* __restrict__ block_columns,
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

  float row_max = -FLT_MAX;
  float denom = 0.0f;
  float acc = 0.0f;

  for (int cursor = block_offsets[query_block]; cursor < block_offsets[query_block + 1]; ++cursor) {
    const int key_block = block_columns[cursor];
    const int key_begin = key_block * block_n;
    const int key_end = min(key_begin + block_n, key_length);
    for (int key = key_begin; key < key_end; ++key) {
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += q[static_cast<size_t>(query) * head_dim + d] *
               k[static_cast<size_t>(key) * head_dim + d];
      }
      const float score = dot * scale;
      const float new_max = fmaxf(row_max, score);
      const float old_scale = row_max == -FLT_MAX ? 0.0f : expf(row_max - new_max);
      const float weight = expf(score - new_max);
      acc = acc * old_scale + weight * v[static_cast<size_t>(key) * head_dim + dim];
      denom = denom * old_scale + weight;
      row_max = new_max;
    }
  }

  output[linear] = denom > 0.0f ? acc / denom : 0.0f;
}

}  // namespace

extern "C" void launch_optimized(const float* q,
                                 const float* k,
                                 const float* v,
                                 const int* block_offsets,
                                 const int* block_columns,
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
  listed_block_sparse_attention<<<grid, kBlockSize, 0, stream>>>(
      q, k, v, block_offsets, block_columns, output, query_length, key_length,
      head_dim, block_m, block_n, scale);
}
