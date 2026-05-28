#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

extern "C" const char* memory_pool_allocator_variant_name();
extern "C" hipError_t memory_pool_allocator_prepare(size_t allocation_bytes,
                                                     int stream_count,
                                                     int allocations_per_stream);
extern "C" size_t memory_pool_allocator_reserved_bytes();
extern "C" hipError_t memory_pool_allocator_run_iteration(
    size_t allocation_bytes,
    int stream_count,
    int allocations_per_stream,
    hipStream_t* streams,
    unsigned long long* checksum);

namespace {

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

float percentile(std::vector<float> values, double q) {
  std::sort(values.begin(), values.end());
  const double index = q * static_cast<double>(values.size() - 1);
  const auto low = static_cast<size_t>(index);
  const auto high =
      static_cast<size_t>(index + (index > static_cast<double>(low) ? 1.0 : 0.0));
  if (low == high) {
    return values[low];
  }
  const double weight = index - static_cast<double>(low);
  return static_cast<float>((1.0 - weight) * values[low] + weight * values[high]);
}

}  // namespace

int main(int argc, char** argv) {
  const size_t allocation_bytes =
      argc > 1 ? static_cast<size_t>(std::strtoull(argv[1], nullptr, 10)) : 65536;
  const int stream_count = argc > 2 ? std::atoi(argv[2]) : 1;
  const int allocations_per_stream = argc > 3 ? std::atoi(argv[3]) : 16;
  const int warmup_iters = argc > 4 ? std::atoi(argv[4]) : 5;
  const int measure_iters = argc > 5 ? std::atoi(argv[5]) : 50;

  if (allocation_bytes == 0 || stream_count <= 0 || allocations_per_stream <= 0 ||
      warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [allocation_bytes>0] [streams>0] "
                 "[allocations_per_stream>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  std::vector<hipStream_t> streams(static_cast<size_t>(stream_count), nullptr);
  for (int i = 0; i < stream_count; ++i) {
    check(hipStreamCreateWithFlags(&streams[static_cast<size_t>(i)],
                                    hipStreamNonBlocking),
          "hipStreamCreateWithFlags");
  }

  unsigned long long* device_checksum = nullptr;
  check(hipMalloc(&device_checksum, sizeof(unsigned long long)),
        "hipMalloc checksum");
  check(hipMemset(device_checksum, 0, sizeof(unsigned long long)),
        "hipMemset checksum");

  check(memory_pool_allocator_prepare(allocation_bytes, stream_count,
                                      allocations_per_stream),
        "memory_pool_allocator_prepare");

  for (int i = 0; i < warmup_iters; ++i) {
    check(memory_pool_allocator_run_iteration(allocation_bytes, stream_count,
                                              allocations_per_stream,
                                              streams.data(), device_checksum),
          "warmup iteration");
  }
  check(hipMemset(device_checksum, 0, sizeof(unsigned long long)),
        "reset checksum");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    const auto start = std::chrono::steady_clock::now();
    check(memory_pool_allocator_run_iteration(allocation_bytes, stream_count,
                                              allocations_per_stream,
                                              streams.data(), device_checksum),
          "measured iteration");
    const auto stop = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    timings_ms.push_back(static_cast<float>(elapsed_ms));
  }

  unsigned long long checksum = 0;
  check(hipMemcpy(&checksum, device_checksum, sizeof(unsigned long long),
                   hipMemcpyDeviceToHost),
        "copy checksum");

  const int allocations_per_iter = stream_count * allocations_per_stream;
  const float median_ms = percentile(timings_ms, 0.5);
  const double median_us_per_allocation =
      (static_cast<double>(median_ms) * 1000.0) /
      static_cast<double>(allocations_per_iter);
  const double total_bytes_per_iter =
      static_cast<double>(allocation_bytes) *
      static_cast<double>(allocations_per_iter);
  const double effective_gib_per_s =
      (total_bytes_per_iter / (1024.0 * 1024.0 * 1024.0)) /
      (static_cast<double>(median_ms) / 1000.0);
  const auto minmax = std::minmax_element(timings_ms.begin(), timings_ms.end());
  const double mean_ms =
      std::accumulate(timings_ms.begin(), timings_ms.end(), 0.0) /
      static_cast<double>(timings_ms.size());
  const bool passed = checksum != 0;
  const size_t reserved_bytes = memory_pool_allocator_reserved_bytes();

  std::cout << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << memory_pool_allocator_variant_name()
            << "\",\n"
            << "  \"operation\": \"memory-pool-allocator\",\n"
            << "  \"allocation_bytes\": " << allocation_bytes << ",\n"
            << "  \"stream_count\": " << stream_count << ",\n"
            << "  \"allocations_per_stream\": " << allocations_per_stream << ",\n"
            << "  \"allocations_per_iter\": " << allocations_per_iter << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"timer_type\": \"host-steady-clock-with-stream-sync\",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"mean_ms\": " << mean_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << *minmax.first << ",\n"
            << "  \"max_ms\": " << *minmax.second << ",\n"
            << "  \"median_us_per_allocation\": "
            << median_us_per_allocation << ",\n"
            << "  \"effective_allocation_gib_per_s\": " << effective_gib_per_s
            << ",\n"
            << "  \"pool_reserved_bytes_after\": " << reserved_bytes << ",\n"
            << "  \"checksum\": " << checksum << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_checksum);
  for (hipStream_t stream : streams) {
    hipStreamDestroy(stream);
  }
  return passed ? 0 : 1;
}
