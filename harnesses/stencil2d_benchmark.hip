#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const float* input,
                                float* output,
                                int width,
                                int height,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* input,
                                 float* output,
                                 int width,
                                 int height,
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
                    float* output,
                    int width,
                    int height,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(input, output, width, height, stream);
#else
  launch_optimized(input, output, width, height, stream);
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
  const int width = argc > 1 ? std::atoi(argv[1]) : 4096;
  const int height = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int warmup_iters = argc > 3 ? std::atoi(argv[3]) : 10;
  const int measure_iters = argc > 4 ? std::atoi(argv[4]) : 50;

  if (width <= 0 || height <= 0 || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [width>0] [height>0] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const size_t element_count = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t byte_count = element_count * sizeof(float);
  std::vector<float> host_input(element_count);
  std::vector<float> host_output(element_count, 0.0f);
  std::vector<float> expected(element_count, 0.0f);

  for (size_t i = 0; i < element_count; ++i) {
    host_input[i] = static_cast<float>((i % 251) - 125) * 0.03125f;
  }
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int idx = y * width + x;
      if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        expected[idx] = host_input[idx];
      } else {
        expected[idx] = 0.25f * host_input[idx] +
                        0.1875f * (host_input[idx - 1] + host_input[idx + 1] +
                                   host_input[idx - width] + host_input[idx + width]);
      }
    }
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
  check(hipMemsetAsync(device_output, 0, byte_count, stream), "clear output");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_input, device_output, width, height, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_input, device_output, width, height, stream);
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
  for (size_t i = 0; i < element_count; ++i) {
    const double error = std::abs(static_cast<double>(host_output[i] - expected[i]));
    max_abs_error = std::max(max_abs_error, error);
    if (error > 1e-5) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double gib = static_cast<double>(6 * byte_count) / (1024.0 * 1024.0 * 1024.0);
  const double effective_gib_per_s = gib / (static_cast<double>(median_ms) / 1000.0);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"stencil5\",\n"
            << "  \"width\": " << width << ",\n"
            << "  \"height\": " << height << ",\n"
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
  hipFree(device_output);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return bad_count == 0 ? 0 : 1;
}
