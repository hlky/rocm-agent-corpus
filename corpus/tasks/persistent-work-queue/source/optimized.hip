#include <hip/hip_runtime.h>

struct Task {
  int begin;
  int count;
};

__global__ void persistent_queue(const Task* tasks, const float* values, float* out, int task_count, int* next_task) {
  __shared__ float partial[256];
  while (true) {
    int task_id = 0;
    if (threadIdx.x == 0) task_id = atomicAdd(next_task, 1);
    task_id = __shfl_sync(0xffffffffu, task_id, 0);
    if (task_id >= task_count) return;
    Task task = tasks[task_id];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < task.count; i += blockDim.x) {
      sum += values[task.begin + i];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
      __syncthreads();
    }
    if (threadIdx.x == 0) out[task_id] = partial[0];
    __syncthreads();
  }
}

extern "C" void launch_optimized(const Task* tasks, const float* values, float* out, int task_count, int* next_task, int resident_blocks, hipStream_t stream) {
  hipMemsetAsync(next_task, 0, sizeof(int), stream);
  persistent_queue<<<resident_blocks, 256, 0, stream>>>(tasks, values, out, task_count, next_task);
}
