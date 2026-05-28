#include <hip/hip_runtime.h>

#if defined(ENABLE_RCCL_TEMPLATE)
#include <rccl.h>

__global__ void local_transform_baseline(const float* in, float* scratch, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) scratch[i] = in[i] * 1.25f + 0.5f;
}

extern "C" hipError_t launch_baseline(const float* local_in,
                                       float* local_scratch,
                                       float* reduced_out,
                                       size_t elements,
                                       rcclComm_t comm,
                                       hipStream_t stream) {
  dim3 block(256);
  dim3 grid(static_cast<unsigned>((elements + block.x - 1) / block.x));
  local_transform_baseline<<<grid, block, 0, stream>>>(local_in, local_scratch, elements);
  hipError_t kernel_status = hipGetLastError();
  if (kernel_status != hipSuccess) return kernel_status;

  // Baseline: finish all local compute before issuing one monolithic collective.
  // TODO: future harness should time this with events on the same stream and
  // record RCCL version, rank count, topology, and relevant environment knobs.
  rcclResult_t rccl_status = rcclAllReduce(local_scratch,
                                           reduced_out,
                                           elements,
                                           rcclFloat,
                                           rcclSum,
                                           comm,
                                           stream);
  return (rccl_status == rcclSuccess) ? hipSuccess : hipErrorUnknown;
}

#else

extern "C" hipError_t launch_baseline(const float*,
                                       float*,
                                       float*,
                                       size_t,
                                       void*,
                                       hipStream_t) {
  // Compile without RCCL by default. Define ENABLE_RCCL_TEMPLATE and link RCCL
  // in a multi-GPU harness to exercise the scaffold.
  return hipErrorNotSupported;
}

#endif
