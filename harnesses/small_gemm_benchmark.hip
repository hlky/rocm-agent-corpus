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
extern "C" void launch_baseline(const float* a,
                                const float* b,
                                float* c,
                                int batch_count,
                                int m,
                                int n,
                                int k,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* a,
                                 const float* b,
                                 float* c,
                                 int batch_count,
                                 int m,
                                 int n,
                                 int k,
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

void launch_variant(const float* a,
                    const float* b,
                    float* c,
                    int batch_count,
                    int m,
                    int n,
                    int k,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(a, b, c, batch_count, m, n, k, stream);
#else
  launch_optimized(a, b, c, batch_count, m, n, k, stream);
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
  if (values.empty()) {
    return 0.0f;
  }
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

float input_value(size_t index, int salt) {
  const int mixed = static_cast<int>((index * 1315423911ull + static_cast<size_t>(salt) * 2654435761ull) % 1021ull);
  return static_cast<float>(mixed - 510) * 0.001953125f;
}

void cpu_reference(const std::vector<float>& a,
                   const std::vector<float>& b,
                   std::vector<float>* reference,
                   int batch_count,
                   int m,
                   int n,
                   int k) {
  for (int batch = 0; batch < batch_count; ++batch) {
    const float* matrix_a = a.data() + static_cast<size_t>(batch) * m * k;
    const float* matrix_b = b.data() + static_cast<size_t>(batch) * k * n;
    float* matrix_c = reference->data() + static_cast<size_t>(batch) * m * n;
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < n; ++col) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          acc += matrix_a[row * k + kk] * matrix_b[kk * n + col];
        }
        matrix_c[row * n + col] = acc;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 16;
  const int n = argc > 2 ? std::atoi(argv[2]) : 16;
  const int k = argc > 3 ? std::atoi(argv[3]) : 16;
  const int batch_count = argc > 4 ? std::atoi(argv[4]) : 4096;
  const int warmup_iters = argc > 5 ? std::atoi(argv[5]) : 10;
  const int measure_iters = argc > 6 ? std::atoi(argv[6]) : 50;

  if (m <= 0 || n <= 0 || k <= 0 || batch_count <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [M>0] [N>0] [K>0] [batch>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t a_elems = static_cast<size_t>(batch_count) * m * k;
  const size_t b_elems = static_cast<size_t>(batch_count) * k * n;
  const size_t c_elems = static_cast<size_t>(batch_count) * m * n;
  const size_t a_bytes = a_elems * sizeof(float);
  const size_t b_bytes = b_elems * sizeof(float);
  const size_t c_bytes = c_elems * sizeof(float);

  std::vector<float> host_a(a_elems);
  std::vector<float> host_b(b_elems);
  std::vector<float> host_c(c_elems, 0.0f);
  std::vector<float> reference(c_elems, 0.0f);
  for (size_t i = 0; i < a_elems; ++i) {
    host_a[i] = input_value(i, 17);
  }
  for (size_t i = 0; i < b_elems; ++i) {
    host_b[i] = input_value(i, 29);
  }
  cpu_reference(host_a, host_b, &reference, batch_count, m, n, k);

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipDeviceProp_t prop{};

  int device = 0;
  check(hipGetDevice(&device), "hipGetDevice");
  check(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_a, a_bytes), "hipMalloc(a)");
  check(hipMalloc(&device_b, b_bytes), "hipMalloc(b)");
  check(hipMalloc(&device_c, c_bytes), "hipMalloc(c)");
  check(hipMemcpyAsync(device_a, host_a.data(), a_bytes, hipMemcpyHostToDevice, stream), "copy a");
  check(hipMemcpyAsync(device_b, host_b.data(), b_bytes, hipMemcpyHostToDevice, stream), "copy b");
  check(hipMemsetAsync(device_c, 0, c_bytes, stream), "clear c");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_a, device_b, device_c, batch_count, m, n, k, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_a, device_b, device_c, batch_count, m, n, k, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_c.data(), device_c, c_bytes, hipMemcpyDeviceToHost, stream), "copy c");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t bad_count = 0;
  constexpr double kAbsTolerance = 1.0e-3;
  constexpr double kRelTolerance = 1.0e-4;
  for (size_t i = 0; i < c_elems; ++i) {
    const double actual = static_cast<double>(host_c[i]);
    const double expected = static_cast<double>(reference[i]);
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(1.0, std::abs(expected));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
    if (abs_error > kAbsTolerance && rel_error > kRelTolerance) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double flops = static_cast<double>(batch_count) * 2.0 * static_cast<double>(m) *
                       static_cast<double>(n) * static_cast<double>(k);
  const double gflop_per_s = flops / (static_cast<double>(median_ms) * 1.0e6);
  const double matrices_per_s = static_cast<double>(batch_count) / (static_cast<double>(median_ms) / 1000.0);
  const double minimum_logical_gib =
      static_cast<double>(a_bytes + b_bytes + c_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double minimum_logical_gib_per_s = minimum_logical_gib / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"batched-small-fixed-sgemm\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"scope\": \"row-major strided batched SGEMM, one HIP launch per timed iteration\",\n"
            << "  \"device_name\": \"" << prop.name << "\",\n"
            << "  \"gfx_target\": \"" << prop.gcnArchName << "\",\n"
            << "  \"m\": " << m << ",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"k\": " << k << ",\n"
            << "  \"batch_count\": " << batch_count << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"gflop_per_s\": " << gflop_per_s << ",\n"
            << "  \"matrices_per_s\": " << matrices_per_s << ",\n"
            << "  \"minimum_logical_gib_per_s\": " << minimum_logical_gib_per_s << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_rel_error\": " << max_rel_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_a);
  hipFree(device_b);
  hipFree(device_c);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
