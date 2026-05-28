#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const float* input,
                                float* output,
                                int segments,
                                int segment_size,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int segments,
                                 int segment_size,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

namespace {

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void launch_variant(const float* input,
                    float* output,
                    int segments,
                    int segment_size,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, output, segments, segment_size, stream);
#else
  launch_optimized(input, output, segments, segment_size, stream);
#endif
}

std::string variant_name() {
#if defined(VARIANT_BASELINE)
  return "baseline";
#else
  return "optimized";
#endif
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

void fill_input(std::vector<float>& values, int segments, int segment_size) {
  for (int segment = 0; segment < segments; ++segment) {
    const int base = segment * segment_size;
    for (int i = 0; i < segment_size; ++i) {
      const int pattern = (segment * 31 + i * 7) % 17;
      values[base + i] = static_cast<float>(pattern - 8) * 0.125f;
    }
  }
}

std::vector<float> cpu_reference(const std::vector<float>& input,
                                 int segments,
                                 int segment_size) {
  std::vector<float> reference(input.size(), 0.0f);
  for (int segment = 0; segment < segments; ++segment) {
    const int base = segment * segment_size;
    float running = 0.0f;
    for (int i = 0; i < segment_size; ++i) {
      running += input[base + i];
      reference[base + i] = running;
    }
  }
  return reference;
}

}  // namespace

int main(int argc, char** argv) {
  const int segments = argc > 1 ? std::atoi(argv[1]) : 65536;
  const int segment_size = argc > 2 ? std::atoi(argv[2]) : 256;
  const int warmup_iters = argc > 3 ? std::atoi(argv[3]) : 5;
  const int measure_iters = argc > 4 ? std::atoi(argv[4]) : 30;
  if (segments <= 0 || segment_size <= 0 || segment_size > 1024 ||
      warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [segments>0] [segment_size 1..1024] [warmup>=0] [iters>0]\n";
    return 2;
  }
  if (segments > std::numeric_limits<int>::max() / segment_size) {
    std::cerr << "segments * segment_size overflows int indexing\n";
    return 2;
  }

  const int n = segments * segment_size;
  const size_t byte_count = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> host_input(n);
  std::vector<float> host_output(n, 0.0f);
  fill_input(host_input, segments, segment_size);
  const std::vector<float> reference = cpu_reference(host_input, segments, segment_size);

  float* device_input = nullptr;
  float* device_output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_input, byte_count), "hipMalloc(input)");
  check(hipMalloc(&device_output, byte_count), "hipMalloc(output)");
  check(hipMemcpyAsync(device_input, host_input.data(), byte_count, hipMemcpyHostToDevice, stream), "copy input");
  check(hipMemsetAsync(device_output, 0, byte_count, stream), "clear output");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_input, device_output, segments, segment_size, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_input, device_output, segments, segment_size, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_output.data(), device_output, byte_count, hipMemcpyDeviceToHost, stream), "copy output");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  size_t bad_count = 0;
  for (int i = 0; i < n; ++i) {
    const double error = std::abs(static_cast<double>(host_output[i]) -
                                  static_cast<double>(reference[i]));
    max_abs_error = std::max(max_abs_error, error);
    if (error > 1e-5) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(byte_count) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = (2.0 * gib) / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"fixed-segment-inclusive-prefix-scan\",\n"
            << "  \"segments\": " << segments << ",\n"
            << "  \"segment_size\": " << segment_size << ",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"effective_gib_per_s\": " << effective_gib_per_s << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_input);
  hipFree(device_output);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
