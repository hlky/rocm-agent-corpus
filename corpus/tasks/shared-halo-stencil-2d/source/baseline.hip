#include <hip/hip_runtime.h>

__global__ void stencil5_global_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int width,
                                       int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) {
    return;
  }

  const int idx = y * width + x;
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[idx] = input[idx];
    return;
  }

  output[idx] = 0.25f * input[idx] +
                0.1875f * (input[idx - 1] + input[idx + 1] +
                           input[idx - width] + input[idx + width]);
}

extern "C" void launch_baseline(const float* input,
                                float* output,
                                int width,
                                int height,
                                hipStream_t stream) {
  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);
  stencil5_global_kernel<<<grid, block, 0, stream>>>(input, output, width, height);
}
