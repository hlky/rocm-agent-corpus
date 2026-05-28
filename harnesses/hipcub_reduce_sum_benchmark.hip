#include <hipcub/hipcub.hpp>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

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
  const auto low = static_cast<size_t>(std::floor(index));
  const auto high = static_cast<size_t>(std::ceil(index));
  if (low == high) {
    return values[low];
  }
  const double weight = index - static_cast<double>(low);
  return static_cast<float>((1.0 - weight) * values[low] + weight * values[high]);
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 16777216;
  const int warmup_iters = argc > 2 ? std::atoi(argv[2]) : 5;
  const int measure_iters = argc > 3 ? std::atoi(argv[3]) : 30;
  if (n <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [n>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t byte_count = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> host_input(n, 1.0f);
  float host_output = 0.0f;
  const double expected = static_cast<double>(n);

  float* device_input = nullptr;
  float* device_output = nullptr;
  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_input, byte_count), "hipMalloc(input)");
  check(hipMalloc(&device_output, sizeof(float)), "hipMalloc(output)");
  check(hipMemcpyAsync(device_input, host_input.data(), byte_count, hipMemcpyHostToDevice, stream), "copy input");
  check(hipcub::DeviceReduce::Sum(nullptr, temp_storage_bytes, device_input, device_output, n, stream),
        "cub temp query");
  check(hipMalloc(&temp_storage, temp_storage_bytes), "hipMalloc(temp_storage)");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    check(hipcub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, device_input, device_output, n, stream),
          "cub warmup");
  }
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    check(hipcub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, device_input, device_output, n, stream),
          "cub measured");
    check(hipEventRecord(stop, stream), "record stop");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(&host_output, device_output, sizeof(float), hipMemcpyDeviceToHost, stream), "copy output");
  check(hipStreamSynchronize(stream), "final sync");

  const double max_abs_error = std::abs(static_cast<double>(host_output) - expected);
  const bool passed = max_abs_error <= 0.5;
  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(byte_count) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = gib / (static_cast<double>(median_ms) / 1000.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"task_id\": \"block-reduction-sum\",\n"
            << "  \"baseline_id\": \"cub-device-reduce-sum\",\n"
            << "  \"variant\": \"cub\",\n"
            << "  \"operation\": \"sum\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"timer_scope\": \"HIP events around CUB DeviceReduce::Sum only; allocation and temp-storage query are excluded\",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"temp_storage_bytes\": " << temp_storage_bytes << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"effective_gib_per_s\": " << effective_gib_per_s << ",\n"
            << "  \"result\": " << host_output << ",\n"
            << "  \"expected\": " << expected << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(temp_storage);
  hipFree(device_input);
  hipFree(device_output);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
