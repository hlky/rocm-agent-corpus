#include <hip/hip_runtime.h>

struct QueueEntry {
  int source_pe;
  int sequence;
  float payload;
};

__global__ void prepare_queue_entries(QueueEntry* local_queue, int source_pe, int entry_count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < entry_count) {
    local_queue[i].source_pe = source_pe;
    local_queue[i].sequence = i;
    local_queue[i].payload = static_cast<float>(i) * 0.25f;
  }
}

extern "C" hipError_t launch_baseline(QueueEntry* local_queue,
                                       QueueEntry* host_staging,
                                       QueueEntry* remote_queue_host_mapping,
                                       int* remote_tail_host_mapping,
                                       int source_pe,
                                       int entry_count,
                                       int queue_capacity,
                                       hipStream_t stream) {
  dim3 block(256);
  dim3 grid((entry_count + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
  prepare_queue_entries<<<grid, block, 0, stream>>>(local_queue, source_pe, entry_count);
  hipError_t status = hipGetLastError();
  if (status != hipSuccess) return status;

  // Baseline: device prepares entries, then the host mediates queue publication.
  // TODO: a real harness should replace the host mappings with IPC, MPI, or an
  // rocSHMEM host path and define the queue overflow policy explicitly.
  status = hipMemcpyAsync(host_staging, local_queue, sizeof(QueueEntry) * entry_count, hipMemcpyDeviceToHost, stream);
  if (status != hipSuccess) return status;
  status = hipStreamSynchronize(stream);
  if (status != hipSuccess) return status;

  for (int i = 0; i < entry_count; ++i) {
    int slot = (*remote_tail_host_mapping)++ % queue_capacity;
    remote_queue_host_mapping[slot] = host_staging[i];
  }
  return hipSuccess;
}
