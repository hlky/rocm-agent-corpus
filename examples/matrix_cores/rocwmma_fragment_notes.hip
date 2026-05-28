#include <hip/hip_fp16.h>
#include <mma.h>

// Reference sketch only. This file demonstrates a safe fragment-local epilogue:
// every accumulator element receives the same operation, so no lane-to-coordinate
// mapping is assumed.

namespace tensor_core_examples {

__device__ void uniform_accumulator_scale(float alpha,
                                          const half* a_tile,
                                          const half* b_tile,
                                          float* c_tile,
                                          int lda,
                                          int ldb,
                                          int ldc) {
#if defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ < 700)
  return;
#else
  namespace rocwmma = rocwmma;

  rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major> a_frag;
  rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, half, rocwmma::col_major> b_frag;
  rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> acc_frag;

  rocwmma::fill_fragment(acc_frag, 0.0f);
  rocwmma::load_matrix_sync(a_frag, a_tile, lda);
  rocwmma::load_matrix_sync(b_frag, b_tile, ldb);
  rocwmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

  for (int i = 0; i < acc_frag.num_elements; ++i) {
    acc_frag.x[i] *= alpha;
  }

  rocwmma::store_matrix_sync(c_tile, acc_frag, ldc, rocwmma::mem_row_major);
#endif
}

}  // namespace tensor_core_examples
