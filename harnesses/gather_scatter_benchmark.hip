#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const float* input,
                                const int* indices,
                                float* gathered,
                                float* scattered,
                                int n,
                                int coalesced_mode,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* input,
                                 const int* indices,
                                 float* gathered,
                                 float* scattered,
                                 int n,
                                 int coalesced_mode,
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
                    const int* indices,
                    float* gathered,
                    float* scattered,
                    int n,
                    int coalesced_mode,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, indices, gathered, scattered, n, coalesced_mode, stream);
#else
  launch_optimized(input, indices, gathered, scattered, n, coalesced_mode, stream);
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

int choose_coprime_stride(int n) {
  int stride = 8191;
  while (stride > 1 && std::gcd(stride, n) != 1) {
    stride -= 2;
  }
  return stride > 1 ? stride : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 16777216;
  const int coalesced_mode = argc > 2 ? std::atoi(argv[2]) : 0;
  const int warmup_iters = argc > 3 ? std::atoi(argv[3]) : 10;
  const int measure_iters = argc > 4 ? std::atoi(argv[4]) : 50;

  if (n <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [n>0] [coalesced_mode 0|1] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t float_bytes = static_cast<size_t>(n) * sizeof(float);
  const size_t index_bytes = static_cast<size_t>(n) * sizeof(int);
  std::vector<float> host_input(n);
  std::vector<int> host_indices(n);
  std::vector<float> host_gathered(n, 0.0f);
  std::vector<float> host_scattered(n, 0.0f);
  std::vector<float> expected_gathered(n);
  std::vector<float> expected_scattered(n);

  for (int i = 0; i < n; ++i) {
    host_input[i] = static_cast<float>((i % 257) - 128) * 0.125f;
  }

  const int stride = coalesced_mode ? 1 : choose_coprime_stride(n);
  for (int i = 0; i < n; ++i) {
    host_indices[i] = coalesced_mode ? i : static_cast<int>((static_cast<long long>(i) * stride) % n);
  }
  for (int i = 0; i < n; ++i) {
    const int mapped = coalesced_mode ? i : host_indices[i];
    expected_gathered[i] = host_input[mapped];
    expected_scattered[mapped] = host_input[i];
  }

  float* device_input = nullptr;
  float* device_gathered = nullptr;
  float* device_scattered = nullptr;
  int* device_indices = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;

  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_input, float_bytes), "hipMalloc(input)");
  check(hipMalloc(&device_gathered, float_bytes), "hipMalloc(gathered)");
  check(hipMalloc(&device_scattered, float_bytes), "hipMalloc(scattered)");
  check(hipMalloc(&device_indices, index_bytes), "hipMalloc(indices)");
  check(hipMemcpyAsync(device_input, host_input.data(), float_bytes, hipMemcpyHostToDevice, stream), "copy input");
  check(hipMemcpyAsync(device_indices, host_indices.data(), index_bytes, hipMemcpyHostToDevice, stream), "copy indices");
  check(hipMemsetAsync(device_gathered, 0, float_bytes, stream), "clear gathered");
  check(hipMemsetAsync(device_scattered, 0, float_bytes, stream), "clear scattered");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_input, device_indices, device_gathered, device_scattered, n, coalesced_mode, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_input, device_indices, device_gathered, device_scattered, n, coalesced_mode, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_gathered.data(), device_gathered, float_bytes, hipMemcpyDeviceToHost, stream), "copy gathered");
  check(hipMemcpyAsync(host_scattered.data(), device_scattered, float_bytes, hipMemcpyDeviceToHost, stream), "copy scattered");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  size_t bad_count = 0;
  for (int i = 0; i < n; ++i) {
    const double gather_error = std::abs(static_cast<double>(host_gathered[i] - expected_gathered[i]));
    const double scatter_error = std::abs(static_cast<double>(host_scattered[i] - expected_scattered[i]));
    max_abs_error = std::max(max_abs_error, std::max(gather_error, scatter_error));
    if (gather_error != 0.0 || scatter_error != 0.0) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(4 * float_bytes + (coalesced_mode ? 0 : 2 * index_bytes)) /
                     (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = gib / (static_cast<double>(median_ms) / 1000.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"gather-scatter\",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"coalesced_mode\": " << (coalesced_mode ? "true" : "false") << ",\n"
            << "  \"index_stride\": " << stride << ",\n"
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
            << "  \"passed\": " << (bad_count == 0 ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_input);
  hipFree(device_gathered);
  hipFree(device_scattered);
  hipFree(device_indices);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return bad_count == 0 ? 0 : 1;
}
