#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(float alpha,
                                const float* x,
                                const float* y,
                                float* output,
                                int n,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(float alpha,
                                 const float* x,
                                 const float* y,
                                 float* output,
                                 int n,
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

void launch_variant(float alpha,
                    const float* x,
                    const float* y,
                    float* output,
                    int n,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(alpha, x, y, output, n, stream);
#else
  launch_optimized(alpha, x, y, output, n, stream);
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

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 16777216;
  const int warmup_iters = argc > 2 ? std::atoi(argv[2]) : 10;
  const int measure_iters = argc > 3 ? std::atoi(argv[3]) : 50;
  const float alpha = 1.25f;
  if (n <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [n>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t byte_count = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> host_x(n);
  std::vector<float> host_y(n);
  std::vector<float> host_output(n, 0.0f);
  for (int i = 0; i < n; ++i) {
    host_x[i] = static_cast<float>((i % 97) - 48) * 0.125f;
    host_y[i] = static_cast<float>((i % 53) - 26) * 0.25f;
  }

  float* device_x = nullptr;
  float* device_y = nullptr;
  float* device_output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_x, byte_count), "hipMalloc(x)");
  check(hipMalloc(&device_y, byte_count), "hipMalloc(y)");
  check(hipMalloc(&device_output, byte_count), "hipMalloc(output)");
  check(hipMemcpyAsync(device_x, host_x.data(), byte_count, hipMemcpyHostToDevice, stream), "copy x");
  check(hipMemcpyAsync(device_y, host_y.data(), byte_count, hipMemcpyHostToDevice, stream), "copy y");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(alpha, device_x, device_y, device_output, n, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(alpha, device_x, device_y, device_output, n, stream);
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
    const float expected = alpha * host_x[i] + host_y[i];
    const double error = std::abs(static_cast<double>(host_output[i]) - expected);
    max_abs_error = std::max(max_abs_error, error);
    if (error > 1e-5) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(byte_count) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = (3.0 * gib) / (static_cast<double>(median_ms) / 1000.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"saxpy\",\n"
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
            << "  \"passed\": " << (bad_count == 0 ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_x);
  hipFree(device_y);
  hipFree(device_output);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return bad_count == 0 ? 0 : 1;
}

