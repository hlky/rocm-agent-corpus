#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const std::uint32_t* input,
                                std::uint32_t* histogram,
                                int n,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const std::uint32_t* input,
                                 std::uint32_t* histogram,
                                 int n,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

namespace {

constexpr int kBins = 256;

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void launch_variant(const std::uint32_t* input,
                    std::uint32_t* histogram,
                    int n,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, histogram, n, stream);
#else
  launch_optimized(input, histogram, n, stream);
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

std::uint32_t lcg(std::uint32_t x) {
  return x * 1664525u + 1013904223u;
}

std::uint32_t make_input_value(int i, const std::string& pattern) {
  const std::uint32_t x = lcg(static_cast<std::uint32_t>(i) ^ 0x9e3779b9u);
  if (pattern == "single-bin") {
    return 7u;
  }
  if (pattern == "sequential") {
    return static_cast<std::uint32_t>(i);
  }
  if (pattern == "uniform") {
    return x;
  }
  if (pattern == "skewed") {
    return (i % 4 == 0) ? x : (x & 15u);
  }
  std::cerr << "unknown input pattern '" << pattern
            << "'; expected skewed, uniform, single-bin, or sequential\n";
  std::exit(2);
}

void fill_input(std::vector<std::uint32_t>* input, const std::string& pattern) {
  for (size_t i = 0; i < input->size(); ++i) {
    (*input)[i] = make_input_value(static_cast<int>(i), pattern);
  }
}

std::vector<std::uint64_t> cpu_histogram(const std::vector<std::uint32_t>& input) {
  std::vector<std::uint64_t> expected(kBins, 0);
  for (std::uint32_t value : input) {
    ++expected[value & (kBins - 1)];
  }
  return expected;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 16777216;
  const int warmup_iters = argc > 2 ? std::atoi(argv[2]) : 5;
  const int measure_iters = argc > 3 ? std::atoi(argv[3]) : 30;
  const std::string pattern = argc > 4 ? argv[4] : "skewed";
  if (n <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [n>0] [warmup>=0] [iters>0] "
              << "[skewed|uniform|single-bin|sequential]\n";
    return 2;
  }

  const size_t input_bytes = static_cast<size_t>(n) * sizeof(std::uint32_t);
  const size_t histogram_bytes = kBins * sizeof(std::uint32_t);
  std::vector<std::uint32_t> host_input(n);
  std::vector<std::uint32_t> host_histogram(kBins, 0);
  fill_input(&host_input, pattern);
  const std::vector<std::uint64_t> expected = cpu_histogram(host_input);

  std::uint32_t* device_input = nullptr;
  std::uint32_t* device_histogram = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_input, input_bytes), "hipMalloc(input)");
  check(hipMalloc(&device_histogram, histogram_bytes), "hipMalloc(histogram)");
  check(hipMemcpyAsync(device_input,
                        host_input.data(),
                        input_bytes,
                        hipMemcpyHostToDevice,
                        stream),
        "copy input");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_input, device_histogram, n, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_input, device_histogram, n, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_histogram.data(),
                        device_histogram,
                        histogram_bytes,
                        hipMemcpyDeviceToHost,
                        stream),
        "copy histogram");
  check(hipStreamSynchronize(stream), "final sync");

  int mismatched_bins = 0;
  std::uint64_t max_abs_error = 0;
  std::uint64_t total = 0;
  std::uint64_t expected_total = 0;
  for (int bin = 0; bin < kBins; ++bin) {
    const std::uint64_t got = host_histogram[bin];
    const std::uint64_t want = expected[bin];
    const std::uint64_t abs_error = got > want ? got - want : want - got;
    if (abs_error != 0) {
      ++mismatched_bins;
    }
    max_abs_error = std::max(max_abs_error, abs_error);
    total += got;
    expected_total += want;
  }
  const bool passed = mismatched_bins == 0 && total == expected_total;

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(input_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = gib / (static_cast<double>(median_ms) / 1000.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"histogram-256\",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"bins\": " << kBins << ",\n"
            << "  \"input_pattern\": \"" << pattern << "\",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"effective_input_gib_per_s\": " << effective_gib_per_s << ",\n"
            << "  \"total\": " << total << ",\n"
            << "  \"expected_total\": " << expected_total << ",\n"
            << "  \"mismatched_bins\": " << mismatched_bins << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"timing_only\": true,\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_input);
  hipFree(device_histogram);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
