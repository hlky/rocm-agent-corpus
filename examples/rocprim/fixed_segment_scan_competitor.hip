#include <cub/cub.hpp>

#include <hip/hip_runtime.h>

// Template-only scaffold.
//
// Baselines to challenge:
//   hipcub::DeviceScan::{InclusiveSum,ExclusiveSum} for global scans and
//   rocthrust::inclusive_scan_by_key or a hipCUB-based prototype for segmented scans.
//
// Custom hypothesis:
//   For fixed-size independent segments, one CTA per segment can use
//   hipcub::BlockScan and avoid a global scan, segment-key materialization, and
//   extra launches around the primitive boundary.

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void fixed_segment_inclusive_scan_kernel(const float* __restrict__ in,
                                                   float* __restrict__ out,
                                                   int segment_size)
{
  using block_scan_t = hipcub::BlockScan<float, BLOCK_THREADS>;
  __shared__ typename block_scan_t::TempStorage temp_storage;

  const int segment = blockIdx.x;
  const int segment_base = segment * segment_size;
  float thread_items[ITEMS_PER_THREAD];

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    const int local = threadIdx.x * ITEMS_PER_THREAD + item;
    thread_items[item] = (local < segment_size) ? in[segment_base + local] : 0.0f;
  }

  block_scan_t(temp_storage).InclusiveSum(thread_items, thread_items);

#pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item)
  {
    const int local = threadIdx.x * ITEMS_PER_THREAD + item;
    if (local < segment_size)
    {
      out[segment_base + local] = thread_items[item];
    }
  }
}

inline hipError_t launch_fixed_segment_inclusive_scan(const float* d_in,
                                                       float* d_out,
                                                       int segments,
                                                       int segment_size,
                                                       hipStream_t stream)
{
  constexpr int block_threads = 256;
  constexpr int items_per_thread = 4;
  if (segment_size > block_threads * items_per_thread)
  {
    return hipErrorInvalidValue;
  }

  fixed_segment_inclusive_scan_kernel<block_threads, items_per_thread>
    <<<segments, block_threads, 0, stream>>>(d_in, d_out, segment_size);
  return hipGetLastError();
}
