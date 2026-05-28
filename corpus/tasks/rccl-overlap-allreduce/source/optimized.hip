#include <hip/hip_runtime.h>

#if defined(ENABLE_RCCL_TEMPLATE)
#include <rccl.h>

__global__ void local_transform_chunked(const float* in, float* scratch, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) scratch[i] = in[i] * 1.25f + 0.5f;
}

extern "C" hipError_t launch_optimized(const float* local_in,
                                        float* local_scratch,
                                        float* reduced_out,
                                        size_t elements,
                                        int chunk_elements,
                                        rcclComm_t comm,
                                        hipStream_t compute_stream,
                                        hipStream_t comm_stream,
                                        hipEvent_t* chunk_ready_events) {
  if (chunk_elements <= 0) return hipErrorInvalidValue;

  dim3 block(256);
  int chunk_id = 0;
  for (size_t offset = 0; offset < elements; offset += static_cast<size_t>(chunk_elements), ++chunk_id) {
    size_t remaining = elements - offset;
    size_t chunk = remaining < static_cast<size_t>(chunk_elements) ? remaining : static_cast<size_t>(chunk_elements);
    dim3 grid(static_cast<unsigned>((chunk + block.x - 1) / block.x));

    local_transform_chunked<<<grid, block, 0, compute_stream>>>(local_in + offset, local_scratch + offset, chunk);
    hipError_t kernel_status = hipGetLastError();
    if (kernel_status != hipSuccess) return kernel_status;

    hipError_t event_status = hipEventRecord(chunk_ready_events[chunk_id], compute_stream);
    if (event_status != hipSuccess) return event_status;
    event_status = hipStreamWaitEvent(comm_stream, chunk_ready_events[chunk_id], 0);
    if (event_status != hipSuccess) return event_status;

    // Optimized hypothesis: split the message so communication for early chunks
    // can run while compute prepares later chunks. This is template-only until a
    // multi-GPU run records timing or timeline evidence.
    rcclResult_t rccl_status = rcclAllReduce(local_scratch + offset,
                                             reduced_out + offset,
                                             chunk,
                                             rcclFloat,
                                             rcclSum,
                                             comm,
                                             comm_stream);
    if (rccl_status != rcclSuccess) return hipErrorUnknown;
  }
  return hipSuccess;
}

#else

extern "C" hipError_t launch_optimized(const float*,
                                        float*,
                                        float*,
                                        size_t,
                                        int,
                                        void*,
                                        hipStream_t,
                                        hipStream_t,
                                        hipEvent_t*) {
  // Compile without RCCL by default. Define ENABLE_RCCL_TEMPLATE and link RCCL
  // in a multi-GPU harness to exercise the scaffold.
  return hipErrorNotSupported;
}

#endif
