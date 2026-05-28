#include <hip/hip_runtime.h>

struct QueueEntry {
  int source_pe;
  int sequence;
  float payload;
};

#if defined(ENABLE_rocSHMEM_TEMPLATE)
#include <rocshmem.h>
#include <rocshmemx.h>

__global__ void gpu_initiated_enqueue(QueueEntry* symmetric_remote_queue,
                                      int* symmetric_remote_tail,
                                      int source_pe,
                                      int entry_count,
                                      int queue_capacity,
                                      int target_pe) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= entry_count) return;

  // Device-side enqueue hypothesis: allocate a remote slot and publish payload
  // from the GPU, avoiding host round trips. This needs symmetric allocations
  // and a measured multi-PE harness before any performance claim.
  int ticket = rocshmem_int_atomic_fetch_inc(symmetric_remote_tail, target_pe);
  int slot = ticket % queue_capacity;
  rocshmem_int_p(&symmetric_remote_queue[slot].source_pe, source_pe, target_pe);
  rocshmem_int_p(&symmetric_remote_queue[slot].sequence, i, target_pe);
  rocshmem_float_p(&symmetric_remote_queue[slot].payload, static_cast<float>(i) * 0.25f, target_pe);
  rocshmem_quiet();
}

extern "C" hipError_t launch_optimized(QueueEntry* symmetric_remote_queue,
                                        int* symmetric_remote_tail,
                                        int source_pe,
                                        int entry_count,
                                        int queue_capacity,
                                        int target_pe,
                                        hipStream_t stream) {
  if (queue_capacity <= 0) return hipErrorInvalidValue;
  dim3 block(256);
  dim3 grid((entry_count + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
  gpu_initiated_enqueue<<<grid, block, 0, stream>>>(symmetric_remote_queue,
                                                    symmetric_remote_tail,
                                                    source_pe,
                                                    entry_count,
                                                    queue_capacity,
                                                    target_pe);
  return hipGetLastError();
}

#else

extern "C" hipError_t launch_optimized(QueueEntry*,
                                        int*,
                                        int,
                                        int,
                                        int,
                                        int,
                                        hipStream_t) {
  // Compile without rocSHMEM by default. Define ENABLE_rocSHMEM_TEMPLATE and link
  // rocSHMEM in a multi-PE harness to exercise the scaffold.
  return hipErrorNotSupported;
}

#endif
