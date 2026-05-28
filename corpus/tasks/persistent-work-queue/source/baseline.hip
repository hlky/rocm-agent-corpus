#include <hip/hip_runtime.h>

struct Task {
  int begin;
  int count;
};

__global__ void one_block_per_task(const Task* tasks, const float* values, float* out, int task_count) {
  int task_id = blockIdx.x;
  if (task_id >= task_count) return;
  Task task = tasks[task_id];
  float sum = 0.0f;
  for (int i = threadIdx.x; i < task.count; i += blockDim.x) {
    sum += values[task.begin + i];
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[task_id] = partial[0];
}

extern "C" void launch_baseline(const Task* tasks, const float* values, float* out, int task_count, hipStream_t stream) {
  one_block_per_task<<<task_count, 256, 0, stream>>>(tasks, values, out, task_count);
}
