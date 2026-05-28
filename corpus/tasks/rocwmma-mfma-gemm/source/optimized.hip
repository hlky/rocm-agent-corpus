#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

namespace {

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;

__global__ void hgemm_scalar_fallback_kernel(const half* __restrict__ a,
                                             const half* __restrict__ b,
                                             float* __restrict__ c,
                                             int m,
                                             int n,
                                             int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += __half2float(a[row * k + kk]) * __half2float(b[col * k + kk]);
  }
  c[row * n + col] = acc;
}

__global__ void hgemm_rocwmma_kernel(const half* __restrict__ a,
                                     const half* __restrict__ b,
                                     float* __restrict__ c,
                                     int m,
                                     int n,
                                     int k) {
  const int tile_col = blockIdx.x;
  const int tile_row = blockIdx.y;
  const int row = tile_row * kWmmaM;
  const int col = tile_col * kWmmaN;

  rocwmma::fragment<rocwmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc_frag;
  rocwmma::fill_fragment(acc_frag, 0.0f);

  for (int kk = 0; kk < k; kk += kWmmaK) {
    rocwmma::fragment<rocwmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half, rocwmma::row_major> a_frag;
    rocwmma::fragment<rocwmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half, rocwmma::col_major> b_frag;

    const half* a_tile = a + row * k + kk;
    const half* b_tile = b + col * k + kk;
    rocwmma::load_matrix_sync(a_frag, a_tile, k);
    rocwmma::load_matrix_sync(b_frag, b_tile, k);
    rocwmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  rocwmma::store_matrix_sync(c + row * n + col, acc_frag, n, rocwmma::mem_row_major);
}

bool is_rocwmma_aligned(int m, int n, int k) {
  return (m % kWmmaM == 0) && (n % kWmmaN == 0) && (k % kWmmaK == 0);
}

void launch_scalar_fallback(const half* a,
                            const half* b,
                            float* c,
                            int m,
                            int n,
                            int k,
                            hipStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
  hgemm_scalar_fallback_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}

}  // namespace

extern "C" void launch_optimized(const half* a,
                                 const half* b,
                                 float* c,
                                 int m,
                                 int n,
                                 int k,
                                 hipStream_t stream) {
  if (!is_rocwmma_aligned(m, n, k)) {
    launch_scalar_fallback(a, b, c, m, n, k, stream);
    return;
  }

  const dim3 block(64, 1, 1);
  const dim3 grid(n / kWmmaN, m / kWmmaM, 1);
  hgemm_rocwmma_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k);
}
