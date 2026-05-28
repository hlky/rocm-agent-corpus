#include <cub/cub.hpp>

#include <functional>
#include <hip/hip_runtime.h>

// Template-only scaffold.
//
// Baseline to beat:
//   hipcub::DeviceReduce::TransformReduce, or transform iterator + DeviceReduce
//   when TransformReduce is not available in the target rocPRIM/hipCUB/rocThrust version.
//
// Custom hypothesis:
//   Fuse transform, local accumulation, block reduction, and final accumulation
//   for a known transform and contiguous float input. This avoids a separate
//   transform output and can be embedded into a larger fused kernel.

struct Square
{
  __host__ __device__ float operator()(float x) const { return x * x; }
};

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void fused_square_sum_kernel(const float* __restrict__ in,
                                        float* __restrict__ out,
                                        int n)
{
  using block_reduce_t = hipcub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename block_reduce_t::TempStorage temp_storage;

  const int base = (blockIdx.x * BLOCK_THREADS + threadIdx.x) * ITEMS_PER_THREAD;
  float thread_sum = 0.0f;

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    const int idx = base + item * BLOCK_THREADS;
    if (idx < n)
    {
      const float x = in[idx];
      thread_sum += x * x;
    }
  }

  const float block_sum = block_reduce_t(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0)
  {
    atomicAdd(out, block_sum);
  }
}

inline hipError_t launch_fused_square_sum(const float* d_in,
                                           float* d_out,
                                           int n,
                                           hipStream_t stream)
{
  constexpr int block_threads = 256;
  constexpr int items_per_thread = 4;
  const int items_per_block = block_threads * items_per_thread;
  const int blocks = (n + items_per_block - 1) / items_per_block;
  return (fused_square_sum_kernel<block_threads, items_per_thread>
          <<<blocks, block_threads, 0, stream>>>(d_in, d_out, n),
          hipGetLastError());
}

inline hipError_t cub_square_sum_baseline(void* temp_storage,
                                           size_t& temp_storage_bytes,
                                           const float* d_in,
                                           float* d_out,
                                           int n,
                                           hipStream_t stream)
{
  return hipcub::DeviceReduce::TransformReduce(temp_storage,
                                            temp_storage_bytes,
                                            d_in,
                                            d_out,
                                            n,
                                            std::plus<float>{},
                                            Square{},
                                            0.0f,
                                            stream);
}

