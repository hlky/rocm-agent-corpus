#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(VARIANT_BASELINE)
extern "C" void launch_baseline(const float* q,
                                const float* k,
                                const float* v,
                                float* output,
                                int batch_size,
                                int num_heads,
                                int query_length,
                                int key_length,
                                int head_dim,
                                float scale,
                                int causal,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* q,
                                 const float* k,
                                 const float* v,
                                 float* output,
                                 int batch_size,
                                 int num_heads,
                                 int query_length,
                                 int key_length,
                                 int head_dim,
                                 float scale,
                                 int causal,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

namespace {

constexpr double kAbsTolerance = 1.0e-3;
constexpr double kRelTolerance = 1.0e-3;

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void launch_variant(const float* q,
                    const float* k,
                    const float* v,
                    float* output,
                    int batch_size,
                    int num_heads,
                    int query_length,
                    int key_length,
                    int head_dim,
                    float scale,
                    int causal,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(q, k, v, output, batch_size, num_heads, query_length,
                  key_length, head_dim, scale, causal, stream);
#else
  launch_optimized(q, k, v, output, batch_size, num_heads, query_length,
                   key_length, head_dim, scale, causal, stream);
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
  const int mixed = static_cast<int>((index * 1103515245ull +
                                      static_cast<size_t>(salt) * 12345ull) %
                                     2047ull);
  return static_cast<float>(mixed - 1023) * 0.0009765625f;
}

int causal_query_offset(int query_length, int key_length) {
  return std::max(0, key_length - query_length);
}

bool key_is_visible(int query_index,
                    int key_index,
                    int query_length,
                    int key_length,
                    int causal) {
  if (!causal) {
    return true;
  }
  const int absolute_query = query_index + causal_query_offset(query_length, key_length);
  return key_index <= absolute_query;
}

void cpu_reference(const std::vector<float>& q,
                   const std::vector<float>& k,
                   const std::vector<float>& v,
                   std::vector<float>* reference,
                   int batch_size,
                   int num_heads,
                   int query_length,
                   int key_length,
                   int head_dim,
                   double scale,
                   int causal) {
  std::vector<double> scores(static_cast<size_t>(key_length));
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int head = 0; head < num_heads; ++head) {
      for (int query_index = 0; query_index < query_length; ++query_index) {
        const size_t q_base = ((static_cast<size_t>(batch) * num_heads + head) *
                               query_length + query_index) * head_dim;
        const size_t kv_base = (static_cast<size_t>(batch) * num_heads + head) *
                               key_length * head_dim;
        double row_max = -std::numeric_limits<double>::infinity();
        bool has_visible_key = false;

        for (int key_index = 0; key_index < key_length; ++key_index) {
          double score = -std::numeric_limits<double>::infinity();
          if (key_is_visible(query_index, key_index, query_length, key_length, causal)) {
            const size_t key_base = kv_base + static_cast<size_t>(key_index) * head_dim;
            double dot = 0.0;
            for (int dim = 0; dim < head_dim; ++dim) {
              dot += static_cast<double>(q[q_base + dim]) *
                     static_cast<double>(k[key_base + dim]);
            }
            score = dot * scale;
            row_max = std::max(row_max, score);
            has_visible_key = true;
          }
          scores[static_cast<size_t>(key_index)] = score;
        }

        const size_t output_base = q_base;
        if (!has_visible_key) {
          for (int dim = 0; dim < head_dim; ++dim) {
            (*reference)[output_base + dim] = 0.0f;
          }
          continue;
        }

        double denominator = 0.0;
        for (int key_index = 0; key_index < key_length; ++key_index) {
          if (scores[static_cast<size_t>(key_index)] !=
              -std::numeric_limits<double>::infinity()) {
            scores[static_cast<size_t>(key_index)] =
                std::exp(scores[static_cast<size_t>(key_index)] - row_max);
            denominator += scores[static_cast<size_t>(key_index)];
          } else {
            scores[static_cast<size_t>(key_index)] = 0.0;
          }
        }

        for (int dim = 0; dim < head_dim; ++dim) {
          double accumulator = 0.0;
          for (int key_index = 0; key_index < key_length; ++key_index) {
            const size_t value_index = kv_base + static_cast<size_t>(key_index) *
                                                 head_dim + dim;
            accumulator += scores[static_cast<size_t>(key_index)] *
                           static_cast<double>(v[value_index]);
          }
          (*reference)[output_base + dim] =
              static_cast<float>(accumulator / denominator);
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const int batch_size = argc > 1 ? std::atoi(argv[1]) : 1;
  const int num_heads = argc > 2 ? std::atoi(argv[2]) : 4;
  const int query_length = argc > 3 ? std::atoi(argv[3]) : 128;
  const int key_length = argc > 4 ? std::atoi(argv[4]) : query_length;
  const int head_dim = argc > 5 ? std::atoi(argv[5]) : 64;
  const int causal = argc > 6 ? std::atoi(argv[6]) : 1;
  const int warmup_iters = argc > 7 ? std::atoi(argv[7]) : 5;
  const int measure_iters = argc > 8 ? std::atoi(argv[8]) : 30;

  if (batch_size <= 0 || num_heads <= 0 || query_length <= 0 ||
      key_length <= 0 || head_dim <= 0 || head_dim > 128 ||
      warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [batch>0] [heads>0] [query>0] [key>0] "
              << "[head_dim 1..128] [causal 0|1] [warmup>=0] [iters>0]\n";
    return 2;
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const size_t q_elems = static_cast<size_t>(batch_size) * num_heads *
                         query_length * head_dim;
  const size_t kv_elems = static_cast<size_t>(batch_size) * num_heads *
                          key_length * head_dim;
  const size_t output_elems = q_elems;
  const size_t q_bytes = q_elems * sizeof(float);
  const size_t kv_bytes = kv_elems * sizeof(float);
  const size_t output_bytes = output_elems * sizeof(float);

  std::vector<float> host_q(q_elems);
  std::vector<float> host_k(kv_elems);
  std::vector<float> host_v(kv_elems);
  std::vector<float> host_output(output_elems, 0.0f);
  std::vector<float> reference(output_elems, 0.0f);
  for (size_t i = 0; i < q_elems; ++i) {
    host_q[i] = input_value(i, 17);
  }
  for (size_t i = 0; i < kv_elems; ++i) {
    host_k[i] = input_value(i, 29);
    host_v[i] = input_value(i, 41);
  }
  cpu_reference(host_q, host_k, host_v, &reference, batch_size, num_heads,
                query_length, key_length, head_dim, scale, causal);

  float* device_q = nullptr;
  float* device_k = nullptr;
  float* device_v = nullptr;
  float* device_output = nullptr;
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
  check(hipMalloc(&device_q, q_bytes), "hipMalloc(q)");
  check(hipMalloc(&device_k, kv_bytes), "hipMalloc(k)");
  check(hipMalloc(&device_v, kv_bytes), "hipMalloc(v)");
  check(hipMalloc(&device_output, output_bytes), "hipMalloc(output)");
  check(hipMemcpyAsync(device_q, host_q.data(), q_bytes, hipMemcpyHostToDevice, stream), "copy q");
  check(hipMemcpyAsync(device_k, host_k.data(), kv_bytes, hipMemcpyHostToDevice, stream), "copy k");
  check(hipMemcpyAsync(device_v, host_v.data(), kv_bytes, hipMemcpyHostToDevice, stream), "copy v");
  check(hipMemsetAsync(device_output, 0, output_bytes, stream), "clear output");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_q, device_k, device_v, device_output, batch_size,
                   num_heads, query_length, key_length, head_dim, scale, causal, stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_q, device_k, device_v, device_output, batch_size,
                   num_heads, query_length, key_length, head_dim, scale, causal, stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_output.data(), device_output, output_bytes,
                        hipMemcpyDeviceToHost, stream), "copy output");
  check(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t bad_count = 0;
  for (size_t i = 0; i < output_elems; ++i) {
    const double actual = static_cast<double>(host_output[i]);
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
  const double logical_flops = static_cast<double>(batch_size) * num_heads *
                               query_length * key_length * head_dim * 4.0;
  const double logical_gflop_per_s = logical_flops /
                                     (static_cast<double>(median_ms) * 1.0e6);
  const double minimum_logical_gib =
      static_cast<double>(q_bytes + 2 * kv_bytes + output_bytes) /
      (1024.0 * 1024.0 * 1024.0);
  const double minimum_logical_gib_per_s =
      minimum_logical_gib / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"task_id\": \"online-attention-forward\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"scaled-dot-product-attention-forward\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"scope\": \"FP32 contiguous B,H,S,D attention, one HIP launch per timed iteration\",\n"
            << "  \"device_name\": \"" << prop.name << "\",\n"
            << "  \"gfx_target\": \"" << prop.gcnArchName << "\",\n"
            << "  \"batch_size\": " << batch_size << ",\n"
            << "  \"num_heads\": " << num_heads << ",\n"
            << "  \"query_length\": " << query_length << ",\n"
            << "  \"key_length\": " << key_length << ",\n"
            << "  \"head_dim\": " << head_dim << ",\n"
            << "  \"causal\": " << (causal ? "true" : "false") << ",\n"
            << "  \"scale\": " << scale << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"logical_gflop_per_s\": " << logical_gflop_per_s << ",\n"
            << "  \"minimum_logical_gib_per_s\": " << minimum_logical_gib_per_s << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_rel_error\": " << max_rel_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_q);
  hipFree(device_k);
  hipFree(device_v);
  hipFree(device_output);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
