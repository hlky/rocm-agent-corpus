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
extern "C" void launch_baseline(const float* logits,
                                const float* uniforms,
                                int* sampled_indices,
                                int* topk_indices,
                                float* topk_values,
                                float* topk_probs,
                                int rows,
                                int vocab_size,
                                int k,
                                float temperature,
                                hipStream_t stream);
#elif defined(VARIANT_OPTIMIZED)
extern "C" void launch_optimized(const float* logits,
                                 const float* uniforms,
                                 int* sampled_indices,
                                 int* topk_indices,
                                 float* topk_values,
                                 float* topk_probs,
                                 int rows,
                                 int vocab_size,
                                 int k,
                                 float temperature,
                                 hipStream_t stream);
#else
#error "Define VARIANT_BASELINE or VARIANT_OPTIMIZED"
#endif

namespace {

constexpr int kMaxK = 8;

struct Reference {
  std::vector<int> sampled_indices;
  std::vector<int> topk_indices;
  std::vector<float> topk_values;
  std::vector<float> topk_probs;
};

void check(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void launch_variant(const float* logits,
                    const float* uniforms,
                    int* sampled_indices,
                    int* topk_indices,
                    float* topk_values,
                    float* topk_probs,
                    int rows,
                    int vocab_size,
                    int k,
                    float temperature,
                    hipStream_t stream) {
#if defined(VARIANT_BASELINE)
  launch_baseline(logits,
                  uniforms,
                  sampled_indices,
                  topk_indices,
                  topk_values,
                  topk_probs,
                  rows,
                  vocab_size,
                  k,
                  temperature,
                  stream);
#else
  launch_optimized(logits,
                   uniforms,
                   sampled_indices,
                   topk_indices,
                   topk_values,
                   topk_probs,
                   rows,
                   vocab_size,
                   k,
                   temperature,
                   stream);
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

bool better_candidate(float lhs_value, int lhs_index, float rhs_value, int rhs_index) {
  return lhs_value > rhs_value || (lhs_value == rhs_value && lhs_index < rhs_index);
}

void insert_candidate(float value,
                      int index,
                      std::vector<float>& values,
                      std::vector<int>& indices,
                      int k) {
  for (int slot = 0; slot < k; ++slot) {
    if (better_candidate(value, index, values[slot], indices[slot])) {
      for (int move = k - 1; move > slot; --move) {
        values[move] = values[move - 1];
        indices[move] = indices[move - 1];
      }
      values[slot] = value;
      indices[slot] = index;
      return;
    }
  }
}

void fill_inputs(std::vector<float>& logits,
                 std::vector<float>& uniforms,
                 int rows,
                 int vocab_size) {
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < vocab_size; ++col) {
      const int pattern = (row * 131 + col * 17 + (col / 97) * 23) % 509;
      const int tie_band = (col % 257 == 0) ? 24 : 0;
      logits[static_cast<size_t>(row) * vocab_size + col] =
          static_cast<float>(pattern - 254 + tie_band) * 0.03125f;
    }
    const int bucket = row & 3;
    uniforms[row] = bucket == 0 ? 0.05f : (bucket == 1 ? 0.35f : (bucket == 2 ? 0.65f : 0.95f));
  }
}

Reference cpu_reference(const std::vector<float>& logits,
                        const std::vector<float>& uniforms,
                        int rows,
                        int vocab_size,
                        int k,
                        float temperature) {
  Reference reference;
  reference.sampled_indices.assign(rows, -1);
  reference.topk_indices.assign(static_cast<size_t>(rows) * k, -1);
  reference.topk_values.assign(static_cast<size_t>(rows) * k, 0.0f);
  reference.topk_probs.assign(static_cast<size_t>(rows) * k, 0.0f);

  for (int row = 0; row < rows; ++row) {
    std::vector<float> values(k, -std::numeric_limits<float>::max());
    std::vector<int> indices(k, std::numeric_limits<int>::max());
    const int row_base = row * vocab_size;
    for (int col = 0; col < vocab_size; ++col) {
      insert_candidate(logits[row_base + col], col, values, indices, k);
    }

    const float row_max = values[0];
    double denominator = 0.0;
    for (int i = 0; i < k; ++i) {
      denominator += std::exp(static_cast<double>(values[i] - row_max) /
                              static_cast<double>(temperature));
    }

    float u = uniforms[row];
    u = std::max(0.0f, std::min(1.0f, u));
    double cumulative = 0.0;
    int chosen = indices[k - 1];
    const int output_base = row * k;
    for (int i = 0; i < k; ++i) {
      const double probability =
          std::exp(static_cast<double>(values[i] - row_max) /
                   static_cast<double>(temperature)) /
          denominator;
      reference.topk_indices[output_base + i] = indices[i];
      reference.topk_values[output_base + i] = values[i];
      reference.topk_probs[output_base + i] = static_cast<float>(probability);
      cumulative += probability;
      if (static_cast<double>(u) <= cumulative && chosen == indices[k - 1]) {
        chosen = indices[i];
      }
    }
    reference.sampled_indices[row] = chosen;
  }

  return reference;
}

}  // namespace

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 1024;
  const int vocab_size = argc > 2 ? std::atoi(argv[2]) : 32768;
  const int k = argc > 3 ? std::atoi(argv[3]) : 4;
  const float temperature = argc > 4 ? static_cast<float>(std::atof(argv[4])) : 0.8f;
  const int warmup_iters = argc > 5 ? std::atoi(argv[5]) : 5;
  const int measure_iters = argc > 6 ? std::atoi(argv[6]) : 30;

  if (rows <= 0 || vocab_size <= 0 || k <= 0 || k > kMaxK || k > vocab_size ||
      temperature <= 0.0f || warmup_iters < 0 || measure_iters <= 0) {
    std::cerr << "usage: benchmark [rows>0] [vocab_size>=k] [k 1..8] "
                 "[temperature>0] [warmup>=0] [iters>0]\n";
    return 2;
  }
  if (rows > std::numeric_limits<int>::max() / vocab_size) {
    std::cerr << "rows * vocab_size overflows int indexing\n";
    return 2;
  }

  const size_t logit_count = static_cast<size_t>(rows) * static_cast<size_t>(vocab_size);
  const size_t topk_count = static_cast<size_t>(rows) * static_cast<size_t>(k);
  const size_t logits_bytes = logit_count * sizeof(float);
  const size_t uniform_bytes = static_cast<size_t>(rows) * sizeof(float);
  const size_t sampled_bytes = static_cast<size_t>(rows) * sizeof(int);
  const size_t topk_index_bytes = topk_count * sizeof(int);
  const size_t topk_value_bytes = topk_count * sizeof(float);

  std::vector<float> host_logits(logit_count);
  std::vector<float> host_uniforms(rows);
  fill_inputs(host_logits, host_uniforms, rows, vocab_size);
  const Reference reference = cpu_reference(host_logits, host_uniforms, rows, vocab_size, k, temperature);

  std::vector<int> host_sampled(rows, -1);
  std::vector<int> host_topk_indices(topk_count, -1);
  std::vector<float> host_topk_values(topk_count, 0.0f);
  std::vector<float> host_topk_probs(topk_count, 0.0f);

  float* device_logits = nullptr;
  float* device_uniforms = nullptr;
  int* device_sampled = nullptr;
  int* device_topk_indices = nullptr;
  float* device_topk_values = nullptr;
  float* device_topk_probs = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;

  check(hipStreamCreate(&stream), "hipStreamCreate");
  check(hipEventCreate(&start), "hipEventCreate(start)");
  check(hipEventCreate(&stop), "hipEventCreate(stop)");
  check(hipMalloc(&device_logits, logits_bytes), "hipMalloc(logits)");
  check(hipMalloc(&device_uniforms, uniform_bytes), "hipMalloc(uniforms)");
  check(hipMalloc(&device_sampled, sampled_bytes), "hipMalloc(sampled)");
  check(hipMalloc(&device_topk_indices, topk_index_bytes), "hipMalloc(topk_indices)");
  check(hipMalloc(&device_topk_values, topk_value_bytes), "hipMalloc(topk_values)");
  check(hipMalloc(&device_topk_probs, topk_value_bytes), "hipMalloc(topk_probs)");
  check(hipMemcpyAsync(device_logits, host_logits.data(), logits_bytes, hipMemcpyHostToDevice, stream),
        "copy logits");
  check(hipMemcpyAsync(device_uniforms, host_uniforms.data(), uniform_bytes, hipMemcpyHostToDevice, stream),
        "copy uniforms");
  check(hipStreamSynchronize(stream), "setup sync");

  for (int i = 0; i < warmup_iters; ++i) {
    launch_variant(device_logits,
                   device_uniforms,
                   device_sampled,
                   device_topk_indices,
                   device_topk_values,
                   device_topk_probs,
                   rows,
                   vocab_size,
                   k,
                   temperature,
                   stream);
  }
  check(hipGetLastError(), "warmup launch");
  check(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check(hipEventRecord(start, stream), "record start");
    launch_variant(device_logits,
                   device_uniforms,
                   device_sampled,
                   device_topk_indices,
                   device_topk_values,
                   device_topk_probs,
                   rows,
                   vocab_size,
                   k,
                   temperature,
                   stream);
    check(hipEventRecord(stop, stream), "record stop");
    check(hipGetLastError(), "measured launch");
    check(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check(hipMemcpyAsync(host_sampled.data(), device_sampled, sampled_bytes, hipMemcpyDeviceToHost, stream),
        "copy sampled");
  check(hipMemcpyAsync(host_topk_indices.data(),
                        device_topk_indices,
                        topk_index_bytes,
                        hipMemcpyDeviceToHost,
                        stream),
        "copy topk_indices");
  check(hipMemcpyAsync(host_topk_values.data(),
                        device_topk_values,
                        topk_value_bytes,
                        hipMemcpyDeviceToHost,
                        stream),
        "copy topk_values");
  check(hipMemcpyAsync(host_topk_probs.data(),
                        device_topk_probs,
                        topk_value_bytes,
                        hipMemcpyDeviceToHost,
                        stream),
        "copy topk_probs");
  check(hipStreamSynchronize(stream), "final sync");

  size_t topk_index_mismatch_count = 0;
  size_t sample_mismatch_count = 0;
  double max_topk_value_abs_error = 0.0;
  double max_probability_abs_error = 0.0;
  for (size_t i = 0; i < topk_count; ++i) {
    if (host_topk_indices[i] != reference.topk_indices[i]) {
      ++topk_index_mismatch_count;
    }
    max_topk_value_abs_error =
        std::max(max_topk_value_abs_error,
                 std::abs(static_cast<double>(host_topk_values[i]) -
                          static_cast<double>(reference.topk_values[i])));
    max_probability_abs_error =
        std::max(max_probability_abs_error,
                 std::abs(static_cast<double>(host_topk_probs[i]) -
                          static_cast<double>(reference.topk_probs[i])));
  }
  for (int row = 0; row < rows; ++row) {
    if (host_sampled[row] != reference.sampled_indices[row]) {
      ++sample_mismatch_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const size_t logical_bytes =
      logits_bytes + uniform_bytes + sampled_bytes + topk_index_bytes + 2 * topk_value_bytes;
  const double logical_gib = static_cast<double>(logical_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double logical_gib_per_s = logical_gib / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = topk_index_mismatch_count == 0 && sample_mismatch_count == 0 &&
                      max_topk_value_abs_error <= 1e-6 && max_probability_abs_error <= 1e-4;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"task_id\": \"block-topk-sampling\",\n"
            << "  \"variant\": \"" << variant_name() << "\",\n"
            << "  \"operation\": \"top-k-temperature-sampling\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"rows\": " << rows << ",\n"
            << "  \"vocab_size\": " << vocab_size << ",\n"
            << "  \"k\": " << k << ",\n"
            << "  \"temperature\": " << temperature << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"logical_gib_per_s\": " << logical_gib_per_s << ",\n"
            << "  \"topk_index_mismatch_count\": " << topk_index_mismatch_count << ",\n"
            << "  \"sample_mismatch_count\": " << sample_mismatch_count << ",\n"
            << "  \"max_topk_value_abs_error\": " << max_topk_value_abs_error << ",\n"
            << "  \"max_probability_abs_error\": " << max_probability_abs_error << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  hipFree(device_logits);
  hipFree(device_uniforms);
  hipFree(device_sampled);
  hipFree(device_topk_indices);
  hipFree(device_topk_values);
  hipFree(device_topk_probs);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
