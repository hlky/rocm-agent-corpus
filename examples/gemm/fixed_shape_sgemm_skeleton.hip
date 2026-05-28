#include <hip/hip_runtime.h>

// Small fixed-shape SGEMM skeleton for tasks like `sgemm-fixed-small`.
//
// Contract:
//   Row-major A[M, K], B[K, N], D[M, N]
//   D = relu(A * B + bias[col]) when BiasRelu is true
//
// This is intentionally simple. It is a starting point for specialization:
// unroll K, map one warp/CTA per matrix, batch many matrices per launch, then
// compare against hipBLAS/rocBLAS/hipBLASLt/Composable Kernel baselines for the same contract.

template <int M, int N, int K, bool BiasRelu>
__global__ void fixed_shape_sgemm_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         const float* __restrict__ bias,
                                         float* __restrict__ d,
                                         int batch_count,
                                         int stride_a,
                                         int stride_b,
                                         int stride_d) {
  int batch = blockIdx.z;
  if (batch >= batch_count) {
    return;
  }

  int row = threadIdx.y;
  int col = threadIdx.x;
  if (row >= M || col >= N) {
    return;
  }

  const float* batch_a = a + batch * stride_a;
  const float* batch_b = b + batch * stride_b;
  float* batch_d = d + batch * stride_d;

  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < K; ++k) {
    acc += batch_a[row * K + k] * batch_b[k * N + col];
  }

  if constexpr (BiasRelu) {
    acc += bias[col];
    acc = acc > 0.0f ? acc : 0.0f;
  }

  batch_d[row * N + col] = acc;
}

struct FixedShapeSgemmLaunch {
  int m;
  int n;
  int k;
  int batch_count;
  int stride_a;
  int stride_b;
  int stride_d;
  bool bias_relu;
};

// Example launch shape for M,N <= 32:
//
// dim3 block(N, M, 1);
// dim3 grid(1, 1, batch_count);
// fixed_shape_sgemm_kernel<16, 16, 16, true>
//     <<<grid, block>>>(a, b, bias, d, batch_count, 16 * 16, 16 * 16, 16 * 16);
//
// Metadata to record before benchmarking:
// - task id, GPU, ROCm version, compiler flags, and arch flags
// - M/N/K, batch count, strides, layout, dtype, epilogue, tolerance
// - whether timing includes launch overhead and whether HIP Graphs are used
// - hipBLAS/rocBLAS/hipBLASLt/Composable Kernel baselines for the same shape and epilogue

