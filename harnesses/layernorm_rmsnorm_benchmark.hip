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
                                const float* gamma,
                                const float* beta,
                                float* output,
                                int rows,
                                int cols,
                                float epsilon,
                                int mode,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* input,
                                 const float* gamma,
                                 const float* beta,
                                 float* output,
                                 int rows,
                                 int cols,
                                 float epsilon,
                                 int mode,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

namespace {

constexpr int kModeRmsNorm = 0;
constexpr int kModeLayerNorm = 1;

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

int parse_mode(const char* value) {
  const std::string mode(value);
  if (mode == "rmsnorm" || mode == "rms" || mode == "0") {
    return kModeRmsNorm;
  }
  if (mode == "layernorm" || mode == "layer" || mode == "ln" || mode == "1") {
    return kModeLayerNorm;
  }
  std::cerr << "mode must be rmsnorm or layernorm\n";
  std::exit(2);
}

std::string mode_name(int mode) {
  return mode == kModeLayerNorm ? "layernorm" : "rmsnorm";
}

void launch_variant(const float* input,
                    const float* gamma,
                    const float* beta,
                    float* output,
                    int rows,
                    int cols,
                    float epsilon,
                    int mode,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, gamma, beta, output, rows, cols, epsilon, mode, stream);
#else
  launch_optimized(input, gamma, beta, output, rows, cols, epsilon, mode, stream);
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

void make_reference(const std::vector<float>& input,
                    const std::vector<float>& gamma,
                    const std::vector<float>& beta,
                    std::vector<float>& reference,
                    int rows,
                    int cols,
                    double epsilon,
                    int mode) {
  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    if (mode == kModeLayerNorm) {
      double sum = 0.0;
      for (int col = 0; col < cols; ++col) {
        sum += static_cast<double>(input[base + col]);
      }
      const double mean = sum / static_cast<double>(cols);

      double sum_squares = 0.0;
      for (int col = 0; col < cols; ++col) {
        const double centered = static_cast<double>(input[base + col]) - mean;
        sum_squares += centered * centered;
      }
      const double inv_std = 1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + epsilon);

      for (int col = 0; col < cols; ++col) {
        const double normalized = (static_cast<double>(input[base + col]) - mean) * inv_std;
        reference[base + col] = static_cast<float>(
            normalized * static_cast<double>(gamma[col]) + static_cast<double>(beta[col]));
      }
      continue;
    }

    double sum_squares = 0.0;
    for (int col = 0; col < cols; ++col) {
      const double value = static_cast<double>(input[base + col]);
      sum_squares += value * value;
    }
    const double inv_rms = 1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + epsilon);
    for (int col = 0; col < cols; ++col) {
      reference[base + col] =
          static_cast<float>(static_cast<double>(input[base + col]) * inv_rms *
                             static_cast<double>(gamma[col]));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 1024;
  const int mode = argc > 3 ? parse_mode(argv[3]) : kModeRmsNorm;
  const int warmup_iters = argc > 4 ? std::atoi(argv[4]) : 5;
  const int measure_iters = argc > 5 ? std::atoi(argv[5]) : 30;
  const float epsilon = argc > 6 ? std::strtof(argv[6], nullptr) : 1.0e-5f;
  if (rows <= 0 || cols <= 0 || warmup_iters < 0 || measure_iters <= 0 || epsilon <= 0.0f) {
    std::cerr << "usage: benchmark [rows>0] [cols>0] [rmsnorm|layernorm] "
                 "[warmup>=0] [iters>0] [epsilon>0]\n";
    return 2;
  }

  const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t matrix_bytes = element_count * sizeof(float);
  const size_t vector_bytes = static_cast<size_t>(cols) * sizeof(float);
  std::vector<float> host_input(element_count);
  std::vector<float> host_gamma(static_cast<size_t>(cols));
  std::vector<float> host_beta(static_cast<size_t>(cols));
  std::vector<float> host_output(element_count, 0.0f);
  std::vector<float> host_reference(element_count, 0.0f);

  for (size_t i = 0; i < element_count; ++i) {
    const int pattern = static_cast<int>((i * 17 + i / 7) % 251) - 125;
    host_input[i] = static_cast<float>(pattern) * 0.015625f;
  }
  for (int col = 0; col < cols; ++col) {
    host_gamma[static_cast<size_t>(col)] = 0.75f + static_cast<float>(col % 13) * 0.03125f;
    host_beta[static_cast<size_t>(col)] = static_cast<float>((col % 17) - 8) * 0.0078125f;
  }
  make_reference(host_input, host_gamma, host_beta, host_reference, rows, cols, epsilon, mode);

  float* device_input = nullptr;
  float* device_gamma = nullptr;
  float* device_beta = nullptr;
  float* device_output = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_input, matrix_bytes), "hipMalloc(input)");
  check(hipMalloc(&device_gamma, vector_bytes), "hipMalloc(gamma)");
  check(hipMalloc(&device_beta, vector_bytes), "hipMalloc(beta)");
  check(hipMalloc(&device_output, matrix_bytes), "hipMalloc(output)");
  check(hipMemcpyAsync(device_input, host_input.data(), matrix_bytes, hipMemcpyHostToDevice, stream),
        "copy input");
  check(hipMemcpyAsync(device_gamma, host_gamma.data(), vector_bytes, hipMemcpyHostToDevice, stream),
        "copy gamma");
  check(hipMemcpyAsync(device_beta, host_beta.data(), vector_bytes, hipMemcpyHostToDevice, stream),
        "copy beta");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_input, device_gamma, device_beta, device_output, rows, cols, epsilon, mode, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_input, device_gamma, device_beta, device_output, rows, cols, epsilon, mode, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_output.data(), device_output, matrix_bytes, hipMemcpyDeviceToHost, stream),
        "copy output");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t bad_count = 0;
  for (size_t i = 0; i < element_count; ++i) {
    const double got = static_cast<double>(host_output[i]);
    const double expected = static_cast<double>(host_reference[i]);
    const double abs_error = std::abs(got - expected);
    const double rel_error = abs_error / std::max(1.0e-12, std::abs(expected));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
    if (abs_error > 1.0e-4 && rel_error > 1.0e-3) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double read_bytes = mode == kModeLayerNorm ? 2.0 * static_cast<double>(matrix_bytes)
                                                   : static_cast<double>(matrix_bytes);
  const double write_bytes = static_cast<double>(matrix_bytes);
  const double affine_bytes = mode == kModeLayerNorm ? 2.0 * static_cast<double>(vector_bytes) * rows
                                                     : static_cast<double>(vector_bytes) * rows;
  const double gib = (read_bytes + write_bytes + affine_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = gib / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"rowwise-layernorm-rmsnorm\",\n"
            << "  \"mode\": \"" << mode_name(mode) << "\",\n"
            << "  \"rows\": " << rows << ",\n"
            << "  \"cols\": " << cols << ",\n"
            << "  \"epsilon\": " << epsilon << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"effective_gib_per_s\": " << effective_gib_per_s << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_rel_error\": " << max_rel_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_input);
  hipFree(device_gamma);
  hipFree(device_beta);
  hipFree(device_output);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
