#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const std::uint8_t* packed_weights,
                                const float* x,
                                const float* scales,
                                const std::uint8_t* zero_points,
                                float* y,
                                int rows,
                                int cols,
                                int group_size,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const std::uint8_t* packed_weights,
                                 const float* x,
                                 const float* scales,
                                 const std::uint8_t* zero_points,
                                 float* y,
                                 int rows,
                                 int cols,
                                 int group_size,
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

void launch_variant(const std::uint8_t* packed_weights,
                    const float* x,
                    const float* scales,
                    const std::uint8_t* zero_points,
                    float* y,
                    int rows,
                    int cols,
                    int group_size,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(packed_weights, x, scales, zero_points, y, rows, cols, group_size, stream);
#else
  launch_optimized(packed_weights, x, scales, zero_points, y, rows, cols, group_size, stream);
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
  const auto low = static_cast<std::size_t>(std::floor(index));
  const auto high = static_cast<std::size_t>(std::ceil(index));
  if (low == high) {
    return values[low];
  }
  const double weight = index - static_cast<double>(low);
  return static_cast<float>((1.0 - weight) * values[low] + weight * values[high]);
}

float activation_value(int col) {
  const int mixed = (col * 37 + 11) % 257;
  return static_cast<float>(mixed - 128) * 0.0078125f;
}

int q4_value(int row, int col) {
  return (row * 13 + col * 7 + 5) & 0x0f;
}

std::uint8_t zero_point_value(int row, int group) {
  return static_cast<std::uint8_t>((row * 3 + group * 5 + 8) & 0x0f);
}

float scale_value(int row, int group) {
  const int mixed = (row * 17 + group * 19) % 9;
  return 0.015625f * static_cast<float>(mixed + 1);
}

void fill_inputs(std::vector<std::uint8_t>* packed_weights,
                 std::vector<float>* x,
                 std::vector<float>* scales,
                 std::vector<std::uint8_t>* zero_points,
                 int rows,
                 int cols,
                 int group_size) {
  const int packed_cols = (cols + 1) / 2;
  const int groups_per_row = (cols + group_size - 1) / group_size;

  for (int col = 0; col < cols; ++col) {
    (*x)[col] = activation_value(col);
  }

  for (int row = 0; row < rows; ++row) {
    for (int group = 0; group < groups_per_row; ++group) {
      const std::size_t meta_index = static_cast<std::size_t>(row) * groups_per_row + group;
      (*scales)[meta_index] = scale_value(row, group);
      (*zero_points)[meta_index] = zero_point_value(row, group);
    }

    for (int packed_col = 0; packed_col < packed_cols; ++packed_col) {
      const int col0 = packed_col * 2;
      const int low = q4_value(row, col0);
      const int high = col0 + 1 < cols ? q4_value(row, col0 + 1) : 0;
      (*packed_weights)[static_cast<std::size_t>(row) * packed_cols + packed_col] =
          static_cast<std::uint8_t>(low | (high << 4));
    }
  }
}

void cpu_reference(const std::vector<std::uint8_t>& packed_weights,
                   const std::vector<float>& x,
                   const std::vector<float>& scales,
                   const std::vector<std::uint8_t>& zero_points,
                   std::vector<float>* reference,
                   int rows,
                   int cols,
                   int group_size) {
  const int packed_cols = (cols + 1) / 2;
  const int groups_per_row = (cols + group_size - 1) / group_size;

  for (int row = 0; row < rows; ++row) {
    double acc = 0.0;
    for (int col = 0; col < cols; ++col) {
      const std::uint8_t packed = packed_weights[static_cast<std::size_t>(row) * packed_cols + (col >> 1)];
      const int q = (col & 1) == 0 ? static_cast<int>(packed & 0x0f)
                                  : static_cast<int>((packed >> 4) & 0x0f);
      const int group = col / group_size;
      const std::size_t meta_index = static_cast<std::size_t>(row) * groups_per_row + group;
      const int zp = static_cast<int>(zero_points[meta_index] & 0x0f);
      const double weight = static_cast<double>(q - zp) * static_cast<double>(scales[meta_index]);
      acc += weight * static_cast<double>(x[col]);
    }
    (*reference)[row] = static_cast<float>(acc);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int cols = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int group_size = argc > 3 ? std::atoi(argv[3]) : 128;
  const int warmup_iters = argc > 4 ? std::atoi(argv[4]) : 10;
  const int measure_iters = argc > 5 ? std::atoi(argv[5]) : 50;

  if (rows <= 0 || cols <= 0 || group_size <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [rows>0] [cols>0] [group_size>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const int packed_cols = (cols + 1) / 2;
  const int groups_per_row = (cols + group_size - 1) / group_size;
  const std::size_t packed_elems = static_cast<std::size_t>(rows) * packed_cols;
  const std::size_t x_elems = static_cast<std::size_t>(cols);
  const std::size_t meta_elems = static_cast<std::size_t>(rows) * groups_per_row;
  const std::size_t y_elems = static_cast<std::size_t>(rows);
  const std::size_t packed_bytes = packed_elems * sizeof(std::uint8_t);
  const std::size_t x_bytes = x_elems * sizeof(float);
  const std::size_t scale_bytes = meta_elems * sizeof(float);
  const std::size_t zero_point_bytes = meta_elems * sizeof(std::uint8_t);
  const std::size_t y_bytes = y_elems * sizeof(float);

  std::vector<std::uint8_t> host_packed_weights(packed_elems);
  std::vector<float> host_x(x_elems);
  std::vector<float> host_scales(meta_elems);
  std::vector<std::uint8_t> host_zero_points(meta_elems);
  std::vector<float> host_y(y_elems, 0.0f);
  std::vector<float> reference(y_elems, 0.0f);
  fill_inputs(&host_packed_weights, &host_x, &host_scales, &host_zero_points, rows, cols, group_size);
  cpu_reference(host_packed_weights, host_x, host_scales, host_zero_points, &reference, rows, cols, group_size);

  std::uint8_t* device_packed_weights = nullptr;
  float* device_x = nullptr;
  float* device_scales = nullptr;
  std::uint8_t* device_zero_points = nullptr;
  float* device_y = nullptr;
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
  check(hipMalloc(&device_packed_weights, packed_bytes), "hipMalloc(packed_weights)");
  check(hipMalloc(&device_x, x_bytes), "hipMalloc(x)");
  check(hipMalloc(&device_scales, scale_bytes), "hipMalloc(scales)");
  check(hipMalloc(&device_zero_points, zero_point_bytes), "hipMalloc(zero_points)");
  check(hipMalloc(&device_y, y_bytes), "hipMalloc(y)");
  check(hipMemcpyAsync(device_packed_weights,
                        host_packed_weights.data(),
                        packed_bytes,
                        hipMemcpyHostToDevice,
                        stream),
        "copy packed_weights");
  check(hipMemcpyAsync(device_x, host_x.data(), x_bytes, hipMemcpyHostToDevice, stream), "copy x");
  check(hipMemcpyAsync(device_scales, host_scales.data(), scale_bytes, hipMemcpyHostToDevice, stream),
        "copy scales");
  check(hipMemcpyAsync(device_zero_points,
                        host_zero_points.data(),
                        zero_point_bytes,
                        hipMemcpyHostToDevice,
                        stream),
        "copy zero_points");
  check(hipMemsetAsync(device_y, 0, y_bytes, stream), "clear y");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_packed_weights,
                   device_x,
                   device_scales,
                   device_zero_points,
                   device_y,
                   rows,
                   cols,
                   group_size,
                   stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<std::size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_packed_weights,
                   device_x,
                   device_scales,
                   device_zero_points,
                   device_y,
                   rows,
                   cols,
                   group_size,
                   stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_y.data(), device_y, y_bytes, hipMemcpyDeviceToHost, stream), "copy y");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::size_t bad_count = 0;
  constexpr double kAbsTolerance = 5.0e-3;
  constexpr double kRelTolerance = 5.0e-4;
  for (std::size_t i = 0; i < y_elems; ++i) {
    const double actual = static_cast<double>(host_y[i]);
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
  const double weighted_values = static_cast<double>(rows) * static_cast<double>(cols);
  const double weighted_values_per_s = weighted_values / (static_cast<double>(median_ms) / 1000.0);
  const double minimum_logical_gib =
      static_cast<double>(packed_bytes + x_bytes + scale_bytes + zero_point_bytes + y_bytes) /
      (1024.0 * 1024.0 * 1024.0);
  const double minimum_logical_gib_per_s = minimum_logical_gib / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"fused-int4-dequant-gemv\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"scope\": \"row-major packed uint4 weights, per-row per-group scales and zero points, FP32 vector, FP32 output\",\n"
            << "  \"device_name\": \"" << prop.name << "\",\n"
            << "  \"gfx_target\": \"" << prop.gcnArchName << "\",\n"
            << "  \"rows\": " << rows << ",\n"
            << "  \"cols\": " << cols << ",\n"
            << "  \"group_size\": " << group_size << ",\n"
            << "  \"packed_cols\": " << packed_cols << ",\n"
            << "  \"groups_per_row\": " << groups_per_row << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"weighted_values_per_s\": " << weighted_values_per_s << ",\n"
            << "  \"minimum_logical_gib_per_s\": " << minimum_logical_gib_per_s << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_rel_error\": " << max_rel_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_packed_weights);
  hipFree(device_x);
  hipFree(device_scales);
  hipFree(device_zero_points);
  hipFree(device_y);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
