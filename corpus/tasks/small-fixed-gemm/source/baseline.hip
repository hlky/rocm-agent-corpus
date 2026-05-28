#include <hip/hip_runtime.h>

__global__ void small_sgemm_naive_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int batch_count,
                                         int m,
                                         int n,
                                         int k) {
  const int global = blockIdx.x * blockDim.x + threadIdx.x;
  const int outputs_per_matrix = m * n;
  const int total_outputs = batch_count * outputs_per_matrix;
  if (global >= total_outputs) {
    return;
  }

  const int batch = global / outputs_per_matrix;
  const int matrix_index = global - batch * outputs_per_matrix;
  const int row = matrix_index / n;
  const int col = matrix_index - row * n;

  const float* matrix_a = a + batch * m * k;
  const float* matrix_b = b + batch * k * n;

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += matrix_a[row * k + kk] * matrix_b[kk * n + col];
  }

  c[global] = acc;
}

extern "C" void launch_baseline(const float* a,
                                const float* b,
                                float* c,
                                int batch_count,
                                int m,
                                int n,
                                int k,
                                hipStream_t stream) {
  constexpr int kBlockSize = 256;
  const int total_outputs = batch_count * m * n;
  const int grid = (total_outputs + kBlockSize - 1) / kBlockSize;
  small_sgemm_naive_kernel<<<grid, kBlockSize, 0, stream>>>(a, b, c, batch_count, m, n, k);
}
