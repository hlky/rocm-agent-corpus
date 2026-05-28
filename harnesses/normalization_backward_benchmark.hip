#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const float* x,
                                const float* dy,
                                const float* gamma,
                                const float* saved_mean,
                                const float* saved_inv,
                                float* dx,
                                float* dgamma,
                                float* dbeta,
                                int rows,
                                int cols,
                                int mode,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* x,
                                 const float* dy,
                                 const float* gamma,
                                 const float* saved_mean,
                                 const float* saved_inv,
                                 float* dx,
                                 float* dgamma,
                                 float* dbeta,
                                 int rows,
                                 int cols,
                                 int mode,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

namespace {

constexpr int kModeRmsNorm = 0;
constexpr int kModeLayerNorm = 1;

struct ErrorStats {
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t bad_count = 0;
};

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

std::string variant_name() {
#if defined(VARIANT_BASELINE)
  return "baseline";
#else
  return "optimized";
#endif
}

void launch_variant(const float* x,
                    const float* dy,
                    const float* gamma,
                    const float* saved_mean,
                    const float* saved_inv,
                    float* dx,
                    float* dgamma,
                    float* dbeta,
                    int rows,
                    int cols,
                    int mode,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(x, dy, gamma, saved_mean, saved_inv, dx, dgamma, dbeta, rows, cols, mode, stream);
#else
  launch_optimized(x, dy, gamma, saved_mean, saved_inv, dx, dgamma, dbeta, rows, cols, mode, stream);
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

void fill_inputs(std::vector<float>* values, int salt, float scale) {
  for (size_t i = 0; i < values->size(); ++i) {
    const int pattern = static_cast<int>((i * 1103515245ull + salt * 2654435761ull) % 4099ull);
    (*values)[i] = static_cast<float>((pattern % 257) - 128) * scale;
  }
}

void compute_saved_stats(const std::vector<float>& x,
                         std::vector<float>* saved_mean,
                         std::vector<float>* saved_inv,
                         int rows,
                         int cols,
                         double epsilon,
                         int mode) {
  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    if (mode == kModeLayerNorm) {
      double sum = 0.0;
      for (int col = 0; col < cols; ++col) {
        sum += static_cast<double>(x[base + col]);
      }
      const double mean = sum / static_cast<double>(cols);
      double sum_squares = 0.0;
      for (int col = 0; col < cols; ++col) {
        const double centered = static_cast<double>(x[base + col]) - mean;
        sum_squares += centered * centered;
      }
      (*saved_mean)[row] = static_cast<float>(mean);
      (*saved_inv)[row] = static_cast<float>(
          1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + epsilon));
      continue;
    }

    double sum_squares = 0.0;
    for (int col = 0; col < cols; ++col) {
      const double value = static_cast<double>(x[base + col]);
      sum_squares += value * value;
    }
    (*saved_mean)[row] = 0.0f;
    (*saved_inv)[row] = static_cast<float>(
        1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + epsilon));
  }
}

void make_reference(const std::vector<float>& x,
                    const std::vector<float>& dy,
                    const std::vector<float>& gamma,
                    const std::vector<float>& saved_mean,
                    const std::vector<float>& saved_inv,
                    std::vector<float>* dx,
                    std::vector<float>* dgamma,
                    std::vector<float>* dbeta,
                    int rows,
                    int cols,
                    int mode) {
  std::fill(dx->begin(), dx->end(), 0.0f);
  std::fill(dgamma->begin(), dgamma->end(), 0.0f);
  std::fill(dbeta->begin(), dbeta->end(), 0.0f);

  for (int row = 0; row < rows; ++row) {
    const int base = row * cols;
    const double inv = static_cast<double>(saved_inv[row]);
    if (mode == kModeLayerNorm) {
      const double mean = static_cast<double>(saved_mean[row]);
      double sum_dy_gamma = 0.0;
      double sum_dy_gamma_xhat = 0.0;
      for (int col = 0; col < cols; ++col) {
        const double xhat = (static_cast<double>(x[base + col]) - mean) * inv;
        const double dy_gamma = static_cast<double>(dy[base + col]) *
                                static_cast<double>(gamma[col]);
        sum_dy_gamma += dy_gamma;
        sum_dy_gamma_xhat += dy_gamma * xhat;
      }
      const double inv_cols = 1.0 / static_cast<double>(cols);
      const double mean_dy_gamma = sum_dy_gamma * inv_cols;
      const double mean_dy_gamma_xhat = sum_dy_gamma_xhat * inv_cols;
      for (int col = 0; col < cols; ++col) {
        const double dy_value = static_cast<double>(dy[base + col]);
        const double xhat = (static_cast<double>(x[base + col]) - mean) * inv;
        const double dy_gamma = dy_value * static_cast<double>(gamma[col]);
        (*dx)[base + col] =
            static_cast<float>((dy_gamma - mean_dy_gamma - xhat * mean_dy_gamma_xhat) * inv);
        (*dgamma)[col] += static_cast<float>(dy_value * xhat);
        (*dbeta)[col] += static_cast<float>(dy_value);
      }
      continue;
    }

    double dot = 0.0;
    for (int col = 0; col < cols; ++col) {
      dot += static_cast<double>(dy[base + col]) * static_cast<double>(gamma[col]) *
             static_cast<double>(x[base + col]);
    }
    const double scale = inv * inv * inv * dot / static_cast<double>(cols);
    for (int col = 0; col < cols; ++col) {
      const double x_value = static_cast<double>(x[base + col]);
      const double dy_value = static_cast<double>(dy[base + col]);
      (*dx)[base + col] =
          static_cast<float>(dy_value * static_cast<double>(gamma[col]) * inv - x_value * scale);
      (*dgamma)[col] += static_cast<float>(dy_value * x_value * inv);
    }
  }
}

void check_vector(const std::vector<float>& got,
                  const std::vector<float>& expected,
                  double abs_tolerance,
                  double rel_tolerance,
                  ErrorStats* stats) {
  for (size_t i = 0; i < got.size(); ++i) {
    const double actual = static_cast<double>(got[i]);
    const double ref = static_cast<double>(expected[i]);
    const double abs_error = std::abs(actual - ref);
    const double rel_error = abs_error / std::max(1.0e-12, std::abs(ref));
    stats->max_abs_error = std::max(stats->max_abs_error, abs_error);
    stats->max_rel_error = std::max(stats->max_rel_error, rel_error);
    if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
      ++stats->bad_count;
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
  const double abs_tolerance = argc > 7 ? std::atof(argv[7]) : 5.0e-3;
  const double rel_tolerance = argc > 8 ? std::atof(argv[8]) : 5.0e-3;
  if (rows <= 0 || cols <= 0 || warmup_iters < 0 || measure_iters <= 0 || epsilon <= 0.0f) {
    std::cerr << "usage: benchmark [rows>0] [cols>0] [rmsnorm|layernorm] "
                 "[warmup>=0] [iters>0] [epsilon>0] [abs_tol] [rel_tol]\n";
    return 2;
  }

  const size_t element_count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t matrix_bytes = element_count * sizeof(float);
  const size_t vector_bytes = static_cast<size_t>(cols) * sizeof(float);
  const size_t row_bytes = static_cast<size_t>(rows) * sizeof(float);

  std::vector<float> host_x(element_count);
  std::vector<float> host_dy(element_count);
  std::vector<float> host_gamma(static_cast<size_t>(cols));
  std::vector<float> host_mean(static_cast<size_t>(rows));
  std::vector<float> host_inv(static_cast<size_t>(rows));
  std::vector<float> host_dx(element_count, 0.0f);
  std::vector<float> host_dgamma(static_cast<size_t>(cols), 0.0f);
  std::vector<float> host_dbeta(static_cast<size_t>(cols), 0.0f);
  std::vector<float> ref_dx(element_count, 0.0f);
  std::vector<float> ref_dgamma(static_cast<size_t>(cols), 0.0f);
  std::vector<float> ref_dbeta(static_cast<size_t>(cols), 0.0f);

  fill_inputs(&host_x, 17, 0.0078125f);
  fill_inputs(&host_dy, 29, 0.00390625f);
  for (int col = 0; col < cols; ++col) {
    host_gamma[static_cast<size_t>(col)] = 0.75f + static_cast<float>(col % 17) * 0.015625f;
  }
  compute_saved_stats(host_x, &host_mean, &host_inv, rows, cols, epsilon, mode);
  make_reference(host_x, host_dy, host_gamma, host_mean, host_inv,
                 &ref_dx, &ref_dgamma, &ref_dbeta, rows, cols, mode);

  float* device_x = nullptr;
  float* device_dy = nullptr;
  float* device_gamma = nullptr;
  float* device_mean = nullptr;
  float* device_inv = nullptr;
  float* device_dx = nullptr;
  float* device_dgamma = nullptr;
  float* device_dbeta = nullptr;
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
  check(hipMalloc(&device_x, matrix_bytes), "hipMalloc(x)");
  check(hipMalloc(&device_dy, matrix_bytes), "hipMalloc(dy)");
  check(hipMalloc(&device_gamma, vector_bytes), "hipMalloc(gamma)");
  check(hipMalloc(&device_mean, row_bytes), "hipMalloc(mean)");
  check(hipMalloc(&device_inv, row_bytes), "hipMalloc(inv)");
  check(hipMalloc(&device_dx, matrix_bytes), "hipMalloc(dx)");
  check(hipMalloc(&device_dgamma, vector_bytes), "hipMalloc(dgamma)");
  check(hipMalloc(&device_dbeta, vector_bytes), "hipMalloc(dbeta)");

  check(hipMemcpyAsync(device_x, host_x.data(), matrix_bytes, hipMemcpyHostToDevice, stream), "copy x");
  check(hipMemcpyAsync(device_dy, host_dy.data(), matrix_bytes, hipMemcpyHostToDevice, stream), "copy dy");
  check(hipMemcpyAsync(device_gamma, host_gamma.data(), vector_bytes, hipMemcpyHostToDevice, stream),
        "copy gamma");
  check(hipMemcpyAsync(device_mean, host_mean.data(), row_bytes, hipMemcpyHostToDevice, stream),
        "copy mean");
  check(hipMemcpyAsync(device_inv, host_inv.data(), row_bytes, hipMemcpyHostToDevice, stream), "copy inv");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    check(hipMemsetAsync(device_dgamma, 0, vector_bytes, stream), "warmup clear dgamma");
    check(hipMemsetAsync(device_dbeta, 0, vector_bytes, stream), "warmup clear dbeta");
    launch_variant(device_x, device_dy, device_gamma, device_mean, device_inv, device_dx,
                   device_dgamma, device_dbeta, rows, cols, mode, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    check(hipMemsetAsync(device_dgamma, 0, vector_bytes, stream), "clear dgamma");
    check(hipMemsetAsync(device_dbeta, 0, vector_bytes, stream), "clear dbeta");
    launch_variant(device_x, device_dy, device_gamma, device_mean, device_inv, device_dx,
                   device_dgamma, device_dbeta, rows, cols, mode, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_dx.data(), device_dx, matrix_bytes, hipMemcpyDeviceToHost, stream), "copy dx");
  check(hipMemcpyAsync(host_dgamma.data(), device_dgamma, vector_bytes, hipMemcpyDeviceToHost, stream),
        "copy dgamma");
  check(hipMemcpyAsync(host_dbeta.data(), device_dbeta, vector_bytes, hipMemcpyDeviceToHost, stream),
        "copy dbeta");
  check(hipStreamSynchronize(stream), "final sync");

  ErrorStats dx_errors;
  ErrorStats dgamma_errors;
  ErrorStats dbeta_errors;
  check_vector(host_dx, ref_dx, abs_tolerance, rel_tolerance, &dx_errors);
  check_vector(host_dgamma, ref_dgamma, abs_tolerance, rel_tolerance, &dgamma_errors);
  check_vector(host_dbeta, ref_dbeta, abs_tolerance, rel_tolerance, &dbeta_errors);

  const double max_abs_error =
      std::max(dx_errors.max_abs_error, std::max(dgamma_errors.max_abs_error, dbeta_errors.max_abs_error));
  const double max_rel_error =
      std::max(dx_errors.max_rel_error, std::max(dgamma_errors.max_rel_error, dbeta_errors.max_rel_error));
  const size_t bad_count = dx_errors.bad_count + dgamma_errors.bad_count + dbeta_errors.bad_count;

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double logical_bytes =
      4.0 * static_cast<double>(matrix_bytes) + 4.0 * static_cast<double>(vector_bytes) +
      2.0 * static_cast<double>(row_bytes);
  const double effective_gib_per_s =
      (logical_bytes / (1024.0 * 1024.0 * 1024.0)) / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"task_id\": \"normalization-backward\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"normalization-backward\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"mode\": \"" << mode_name(mode) << "\",\n"
            << "  \"dtype\": \"fp32\",\n"
            << "  \"reduction_policy\": \"baseline uses row-serial reductions plus atomic parameter gradients; optimized uses one CTA per row for dx plus one CTA per column for parameter gradients\",\n"
            << "  \"timer_scope\": \"HIP events include dgamma/dbeta reset and backward kernels; allocation and host-device setup are excluded\",\n"
            << "  \"device_name\": \"" << prop.name << "\",\n"
            << "  \"gfx_target\": \"" << prop.gcnArchName << "\",\n"
            << "  \"compute_capability\": \"" << prop.major << "." << prop.minor << "\",\n"
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
            << "  \"abs_tolerance\": " << abs_tolerance << ",\n"
            << "  \"rel_tolerance\": " << rel_tolerance << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_rel_error\": " << max_rel_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"component_errors\": {\n"
            << "    \"dx\": {\"max_abs_error\": " << dx_errors.max_abs_error
            << ", \"max_rel_error\": " << dx_errors.max_rel_error
            << ", \"bad_count\": " << dx_errors.bad_count << "},\n"
            << "    \"dgamma\": {\"max_abs_error\": " << dgamma_errors.max_abs_error
            << ", \"max_rel_error\": " << dgamma_errors.max_rel_error
            << ", \"bad_count\": " << dgamma_errors.bad_count << "},\n"
            << "    \"dbeta\": {\"max_abs_error\": " << dbeta_errors.max_abs_error
            << ", \"max_rel_error\": " << dbeta_errors.max_rel_error
            << ", \"bad_count\": " << dbeta_errors.bad_count << "}\n"
            << "  },\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_x);
  hipFree(device_dy);
  hipFree(device_gamma);
  hipFree(device_mean);
  hipFree(device_inv);
  hipFree(device_dx);
  hipFree(device_dgamma);
  hipFree(device_dbeta);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
