#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

__global__ void touch_buffer_kernel(unsigned char* buffer,
                                    size_t bytes,
                                    unsigned long long* checksum,
                                    unsigned int tag) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= bytes) {
    return;
  }
  const unsigned char value = static_cast<unsigned char>((i + tag) & 0xffu);
  buffer[i] = value;
  if ((i & 1023u) == 0u) {
    atomicAdd(checksum, static_cast<unsigned long long>(value + 1u));
  }
}

hipError_t sync_streams(hipStream_t* streams, int stream_count) {
  for (int i = 0; i < stream_count; ++i) {
    const hipError_t status = hipStreamSynchronize(streams[i]);
    if (status != hipSuccess) {
      return status;
    }
  }
  return hipSuccess;
}

}  // namespace

extern "C" const char* memory_pool_allocator_variant_name() {
  return "hipMalloc-hipFree";
}

extern "C" hipError_t memory_pool_allocator_prepare(size_t allocation_bytes,
                                                     int stream_count,
                                                     int allocations_per_stream) {
  (void)allocation_bytes;
  (void)stream_count;
  (void)allocations_per_stream;
  return hipSuccess;
}

extern "C" size_t memory_pool_allocator_reserved_bytes() {
  return 0;
}

extern "C" hipError_t memory_pool_allocator_run_iteration(
    size_t allocation_bytes,
    int stream_count,
    int allocations_per_stream,
    hipStream_t* streams,
    unsigned long long* checksum) {
  if (allocation_bytes == 0 || stream_count <= 0 || allocations_per_stream <= 0 ||
      streams == nullptr || checksum == nullptr) {
    return hipErrorInvalidValue;
  }

  const int block_size = 256;
  const int grid_size =
      static_cast<int>((allocation_bytes + static_cast<size_t>(block_size) - 1) /
                       static_cast<size_t>(block_size));

  for (int s = 0; s < stream_count; ++s) {
    for (int a = 0; a < allocations_per_stream; ++a) {
      unsigned char* buffer = nullptr;
      hipError_t status = hipMalloc(&buffer, allocation_bytes);
      if (status != hipSuccess) {
        return status;
      }

      const unsigned int tag = static_cast<unsigned int>(s * 65537 + a * 17);
      touch_buffer_kernel<<<grid_size, block_size, 0, streams[s]>>>(
          buffer, allocation_bytes, checksum, tag);
      status = hipGetLastError();
      if (status != hipSuccess) {
        hipFree(buffer);
        return status;
      }

      status = hipFree(buffer);
      if (status != hipSuccess) {
        return status;
      }
    }
  }

  return sync_streams(streams, stream_count);
}
