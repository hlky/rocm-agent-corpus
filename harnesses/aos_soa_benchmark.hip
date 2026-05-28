#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const float* aos,
                                float* x,
                                float* y,
                                float* z,
                                float* w,
                                int n,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* aos,
                                 float* x,
                                 float* y,
                                 float* z,
                                 float* w,
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

void launch_variant(const float* aos,
                    float* x,
                    float* y,
                    float* z,
                    float* w,
                    int n,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(aos, x, y, z, w, n, stream);
#else
  launch_optimized(aos, x, y, z, w, n, stream);
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

  if (n <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [n>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t tuple_count = static_cast<size_t>(n);
  const size_t aos_bytes = tuple_count * 4 * sizeof(float);
  const size_t soa_bytes = tuple_count * sizeof(float);
  std::vector<float> host_aos(tuple_count * 4);
  std::vector<float> host_x(n, 0.0f);
  std::vector<float> host_y(n, 0.0f);
  std::vector<float> host_z(n, 0.0f);
  std::vector<float> host_w(n, 0.0f);

  for (int i = 0; i < n; ++i) {
    host_aos[4 * i + 0] = static_cast<float>(i % 101) * 0.25f;
    host_aos[4 * i + 1] = static_cast<float>(i % 103) * -0.5f;
    host_aos[4 * i + 2] = static_cast<float>(i % 107) * 0.75f;
    host_aos[4 * i + 3] = static_cast<float>(i % 109) * -1.0f;
  }

  float* device_aos = nullptr;
  float* device_x = nullptr;
  float* device_y = nullptr;
  float* device_z = nullptr;
  float* device_w = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;

  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_aos, aos_bytes), "hipMalloc(aos)");
  check(hipMalloc(&device_x, soa_bytes), "hipMalloc(x)");
  check(hipMalloc(&device_y, soa_bytes), "hipMalloc(y)");
  check(hipMalloc(&device_z, soa_bytes), "hipMalloc(z)");
  check(hipMalloc(&device_w, soa_bytes), "hipMalloc(w)");
  check(hipMemcpyAsync(device_aos, host_aos.data(), aos_bytes, hipMemcpyHostToDevice, stream), "copy aos");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_aos, device_x, device_y, device_z, device_w, n, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_aos, device_x, device_y, device_z, device_w, n, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_x.data(), device_x, soa_bytes, hipMemcpyDeviceToHost, stream), "copy x");
  check(hipMemcpyAsync(host_y.data(), device_y, soa_bytes, hipMemcpyDeviceToHost, stream), "copy y");
  check(hipMemcpyAsync(host_z.data(), device_z, soa_bytes, hipMemcpyDeviceToHost, stream), "copy z");
  check(hipMemcpyAsync(host_w.data(), device_w, soa_bytes, hipMemcpyDeviceToHost, stream), "copy w");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  size_t bad_count = 0;
  for (int i = 0; i < n; ++i) {
    const double ex = std::abs(static_cast<double>(host_x[i] - host_aos[4 * i + 0]));
    const double ey = std::abs(static_cast<double>(host_y[i] - host_aos[4 * i + 1]));
    const double ez = std::abs(static_cast<double>(host_z[i] - host_aos[4 * i + 2]));
    const double ew = std::abs(static_cast<double>(host_w[i] - host_aos[4 * i + 3]));
    const double error = std::max(std::max(ex, ey), std::max(ez, ew));
    max_abs_error = std::max(max_abs_error, error);
    if (error != 0.0) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(aos_bytes + 4 * soa_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = gib / (static_cast<double>(median_ms) / 1000.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"aos-to-soa\",\n"
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

  hipFree(device_aos);
  hipFree(device_x);
  hipFree(device_y);
  hipFree(device_z);
  hipFree(device_w);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return bad_count == 0 ? 0 : 1;
}
