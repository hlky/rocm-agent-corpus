#include <hip/hip_runtime.h>

__global__ void small_sgemm_one_cta_per_matrix_kernel(const float* __restrict__ a,
                                                      const float* __restrict__ b,
                                                      float* __restrict__ c,
                                                      int batch_count,
                                                      int m,
                                                      int n,
                                                      int k) {
  const int batch = blockIdx.x;
  if (batch >= batch_count) {
    return;
  }

  extern __shared__ float shared[];
  float* tile_a = shared;
  float* tile_b = shared + m * k;

  const int tid = threadIdx.x;
  const int a_elems = m * k;
  const int b_elems = k * n;
  const int c_elems = m * n;

  const float* matrix_a = a + batch * a_elems;
  const float* matrix_b = b + batch * b_elems;
  float* matrix_c = c + batch * c_elems;

  for (int idx = tid; idx < a_elems; idx += blockDim.x) {
    tile_a[idx] = matrix_a[idx];
  }
  for (int idx = tid; idx < b_elems; idx += blockDim.x) {
    tile_b[idx] = matrix_b[idx];
  }
  __syncthreads();

  for (int idx = tid; idx < c_elems; idx += blockDim.x) {
    const int row = idx / n;
    const int col = idx - row * n;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      acc += tile_a[row * k + kk] * tile_b[kk * n + col];
    }
    matrix_c[idx] = acc;
  }
}

extern "C" void launch_optimized(const float* a,
                                 const float* b,
                                 float* c,
                                 int batch_count,
                                 int m,
                                 int n,
                                 int k,
                                 hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const size_t shared_bytes = static_cast<size_t>(m * k + k * n) * sizeof(float);
  small_sgemm_one_cta_per_matrix_kernel<<<batch_count, kBlockSize, shared_bytes, stream>>>(
      a, b, c, batch_count, m, n, k);
}
