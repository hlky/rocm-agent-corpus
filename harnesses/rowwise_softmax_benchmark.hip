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
                                int rows,
                                int cols,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int rows,
                                 int cols,
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

void launch_variant(const float* input, float* output, int rows, int cols, hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, output, rows, cols, stream);
#else
  launch_optimized(input, output, rows, cols, stream);
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
  const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int warmup_iters = argc > 3 ? std::atoi(argv[3]) : 5;
  const int measure_iters = argc > 4 ? std::atoi(argv[4]) : 30;
  if (rows <= 0 || cols <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [rows>0] [cols>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t byte_count = element_count * sizeof(float);
  std::vector<float> host_input(element_count);
  std::vector<float> host_output(element_count, 0.0f);
  for (size_t i = 0; i < element_count; ++i) {
    host_input[i] = static_cast<float>((static_cast<int>(i % 257) - 128) * 0.03125f);
  }

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
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_input, device_output, rows, cols, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_input, device_output, rows, cols, stream);
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
  double max_row_sum_error = 0.0;
  size_t bad_count = 0;
  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    double row_max = -std::numeric_limits<double>::infinity();
    for (int col = 0; col < cols; ++col) {
      row_max = std::max(row_max, static_cast<double>(host_input[base + col]));
    }
    double sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      sum += std::exp(static_cast<double>(host_input[base + col]) - row_max);
    }
    double output_sum = 0.0;
    for (int col = 0; col < cols; ++col) {
      const double expected = std::exp(static_cast<double>(host_input[base + col]) - row_max) / sum;
      const double got = static_cast<double>(host_output[base + col]);
      const double error = std::abs(got - expected);
      output_sum += got;
      max_abs_error = std::max(max_abs_error, error);
      if (error > 1e-5) {
        ++bad_count;
      }
    }
    max_row_sum_error = std::max(max_row_sum_error, std::abs(output_sum - 1.0));
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(byte_count) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = (2.0 * gib) / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0 && max_row_sum_error <= 1e-4;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"rowwise-softmax\",\n"
            << "  \"rows\": " << rows << ",\n"
            << "  \"cols\": " << cols << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"effective_gib_per_s\": " << effective_gib_per_s << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_row_sum_error\": " << max_row_sum_error << ",\n"
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

