#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const int* input,
                                const unsigned char* flags,
                                int* output,
                                int* selected_count,
                                int n,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const int* input,
                                 const unsigned char* flags,
                                 int* output,
                                 int* selected_count,
                                 int n,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

extern "C" size_t select_filter_compact_temp_bytes(const int* input,
                                                   const unsigned char* flags,
                                                   int* output,
                                                   int* selected_count,
                                                   int n,
                                                   hipStream_t stream);

namespace {

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

std::string variant_name() {
#if defined(VARIANT_BASELINE)
  return "cub-device-select-flagged";
#else
  return "custom-global-atomic-unstable";
#endif
}

void launch_variant(const int* input,
                    const unsigned char* flags,
                    int* output,
                    int* selected_count,
                    int n,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, flags, output, selected_count, n, stream);
#else
  launch_optimized(input, flags, output, selected_count, n, stream);
#endif
}

float percentile(std::vector<float> values, double q) {
  std::sort(values.begin(), values.end());
  const double index = q * static_cast<double>(values.size() - 1);
  const auto low = static_cast<size_t>(index);
  const auto high = static_cast<size_t>(index + (index > static_cast<double>(low) ? 1.0 : 0.0));
  if (low == high) {
    return values[low];
  }
  const double weight = index - static_cast<double>(low);
  return static_cast<float>((1.0 - weight) * values[low] + weight * values[high]);
}

uint64_t mix64(uint64_t value) {
  value ^= value >> 30;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27;
  value *= 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return value;
}

struct Digest {
  int count = 0;
  int64_t sum = 0;
  uint64_t xor_hash = 0;
};

unsigned char make_flag(int i, const std::string& mode) {
  const uint32_t hash =
      static_cast<uint32_t>(i) * 1103515245u + 12345u + static_cast<uint32_t>(i >> 7);
  if (mode == "sparse") {
    return (hash & 1023u) < 10u;
  }
  if (mode == "medium") {
    return (hash & 1023u) < 512u;
  }
  if (mode == "dense") {
    return (hash & 1023u) < 922u;
  }
  if (mode == "periodic7") {
    return (i % 7) == 0;
  }
  std::cerr << "unknown density mode: " << mode << "\n";
  std::exit(2);
}

void fill_inputs(std::vector<int>& input,
                 std::vector<unsigned char>& flags,
                 const std::string& mode) {
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<int>(i * 17 + 3);
    flags[i] = make_flag(static_cast<int>(i), mode);
  }
}

Digest digest_selected(const std::vector<int>& values, int count) {
  Digest digest;
  digest.count = count;
  for (int i = 0; i < count; ++i) {
    digest.sum += static_cast<int64_t>(values[i]);
    digest.xor_hash ^= mix64(static_cast<uint64_t>(values[i]));
  }
  return digest;
}

Digest expected_digest(const std::vector<int>& input,
                       const std::vector<unsigned char>& flags) {
  Digest digest;
  for (size_t i = 0; i < input.size(); ++i) {
    if (flags[i] != 0) {
      ++digest.count;
      digest.sum += static_cast<int64_t>(input[i]);
      digest.xor_hash ^= mix64(static_cast<uint64_t>(input[i]));
    }
  }
  return digest;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 1048576;
  const std::string mode = argc > 2 ? argv[2] : "medium";
  const int warmup_iters = argc > 3 ? std::atoi(argv[3]) : 5;
  const int measure_iters = argc > 4 ? std::atoi(argv[4]) : 30;

  if (n <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [n>0] [sparse|medium|dense|periodic7] "
                 "[warmup>=0] [iters>0]\n";
    return 2;
  }

  std::vector<int> host_input(n);
  std::vector<unsigned char> host_flags(n);
  fill_inputs(host_input, host_flags, mode);
  const Digest expected = expected_digest(host_input, host_flags);

  int* input = nullptr;
  unsigned char* flags = nullptr;
  int* output = nullptr;
  int* selected_count = nullptr;
  check(hipMalloc(&input, static_cast<size_t>(n) * sizeof(int)), "hipMalloc input");
  check(hipMalloc(&flags, static_cast<size_t>(n) * sizeof(unsigned char)), "hipMalloc flags");
  check(hipMalloc(&output, static_cast<size_t>(n) * sizeof(int)), "hipMalloc output");
  check(hipMalloc(&selected_count, sizeof(int)), "hipMalloc selected_count");
  check(hipMemcpy(input,
                   host_input.data(),
                   static_cast<size_t>(n) * sizeof(int),
                   hipMemcpyHostToDevice),
        "copy input");
  check(hipMemcpy(flags,
                   host_flags.data(),
                   static_cast<size_t>(n) * sizeof(unsigned char),
                   hipMemcpyHostToDevice),
        "copy flags");

  hipStream_t stream;
  hipEvent_t start;
  hipEvent_t stop;
  check(hipStreamCreate(&stream), "create stream");
  check(hipEventCreate(&start), "create start");
  check(hipEventCreate(&stop), "create stop");

  const size_t temp_storage_bytes =
      select_filter_compact_temp_bytes(input, flags, output, selected_count, n, stream);

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(input, flags, output, selected_count, n, stream);
  }
  check(hipStreamSynchronize(stream), "warmup synchronize");

  std::vector<float> timings;
  timings.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(input, flags, output, selected_count, n, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(hipEventElapsedTime(&ms, start, stop), "elapsed");
    timings.push_back(ms);
  }

  int actual_count = 0;
  check(hipMemcpy(&actual_count, selected_count, sizeof(int), hipMemcpyDeviceToHost),
        "copy selected_count");
  if (actual_count < 0 || actual_count > n) {
    std::cerr << "invalid selected count: " << actual_count << "\n";
    return 1;
  }
  std::vector<int> host_output(static_cast<size_t>(actual_count));
  check(hipMemcpy(host_output.data(),
                   output,
                   static_cast<size_t>(actual_count) * sizeof(int),
                   hipMemcpyDeviceToHost),
        "copy output");
  const Digest actual = digest_selected(host_output, actual_count);
  const bool passed = actual.count == expected.count && actual.sum == expected.sum &&
                      actual.xor_hash == expected.xor_hash;

  const auto minmax = std::minmax_element(timings.begin(), timings.end());
  std::cout << std::fixed << std::setprecision(6)
            << "{\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"select-filter-compact\",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"density_mode\": \"" << mode << "\",\n"
            << "  \"selected_count\": " << actual.count << ",\n"
            << "  \"expected_selected_count\": " << expected.count << ",\n"
            << "  \"selected_fraction\": "
            << static_cast<double>(expected.count) / static_cast<double>(n) << ",\n"
            << "  \"output_order_contract\": \"unordered-equivalent\",\n"
            << "  \"temp_storage_bytes\": " << temp_storage_bytes << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << percentile(timings, 0.5) << ",\n"
            << "  \"p10_ms\": " << percentile(timings, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings, 0.9) << ",\n"
            << "  \"min_ms\": " << *minmax.first << ",\n"
            << "  \"max_ms\": " << *minmax.second << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
            << "  \"expected_sum\": " << expected.sum << ",\n"
            << "  \"actual_sum\": " << actual.sum << ",\n"
            << "  \"expected_xor_hash\": " << expected.xor_hash << ",\n"
            << "  \"actual_xor_hash\": " << actual.xor_hash << "\n"
            << "}\n";

  check(hipEventDestroy(start), "destroy start");
  check(hipEventDestroy(stop), "destroy stop");
  check(hipStreamDestroy(stream), "destroy stream");
  hipFree(input);
  hipFree(flags);
  hipFree(output);
  hipFree(selected_count);

  return passed ? 0 : 1;
}
