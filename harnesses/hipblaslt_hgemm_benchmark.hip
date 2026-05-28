#include <hipblasLt.h>
#include <hip/hip_fp16.h>
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

namespace {

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    std::cerr << what << ": " << hipGetErrorString(status) << "\n";
    std::exit(1);
  }
}

void check_hipblas(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    std::cerr << what << ": hipBLAS/rocBLAS status " << static_cast<int>(status) << "\n";
    std::exit(1);
  }
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
  const int mixed =
      static_cast<int>((index * 1315423911ull + static_cast<size_t>(salt) * 2654435761ull) % 1021ull);
  return static_cast<float>((mixed % 41) - 20) * 0.03125f;
}

void fill_half_inputs(std::vector<half>* values, int salt) {
  for (size_t i = 0; i < values->size(); ++i) {
    (*values)[i] = __float2half(input_value(i, salt));
  }
}

void cpu_reference(const std::vector<half>& a,
                   const std::vector<half>& b,
                   std::vector<float>* reference,
                   int m,
                   int n,
                   int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int kk = 0; kk < k; ++kk) {
        const float a_value = __half2float(a[row * k + kk]);
        const float b_value = __half2float(b[col * k + kk]);
        acc += a_value * b_value;
      }
      (*reference)[row * n + col] = acc;
    }
  }
}

struct LtPlan {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t op_desc = nullptr;
  hipblasLtMatrixLayout_t a_desc = nullptr;
  hipblasLtMatrixLayout_t b_desc = nullptr;
  hipblasLtMatrixLayout_t c_desc = nullptr;
  hipblasLtMatrixLayout_t d_desc = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  int heuristic_results_returned = 0;
};

void destroy_plan(LtPlan* plan) {
  if (plan->preference) {
    hipblasLtMatmulPreferenceDestroy(plan->preference);
  }
  if (plan->d_desc) {
    hipblasLtMatrixLayoutDestroy(plan->d_desc);
  }
  if (plan->c_desc) {
    hipblasLtMatrixLayoutDestroy(plan->c_desc);
  }
  if (plan->b_desc) {
    hipblasLtMatrixLayoutDestroy(plan->b_desc);
  }
  if (plan->a_desc) {
    hipblasLtMatrixLayoutDestroy(plan->a_desc);
  }
  if (plan->op_desc) {
    hipblasLtMatmulDescDestroy(plan->op_desc);
  }
  if (plan->handle) {
    hipblasLtDestroy(plan->handle);
  }
}

LtPlan create_plan(int m, int n, int k, size_t workspace_bytes, int requested_heuristics, hipStream_t stream) {
  LtPlan plan;
  check_hipblas(hipblasLtCreate(&plan.handle), "hipblasLtCreate");
  check_hipblas(hipblasLtMatmulDescCreate(&plan.op_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
               "hipblasLtMatmulDescCreate");

  const hipblasOperation_t transa = HIPBLAS_OP_T;
  const hipblasOperation_t transb = HIPBLAS_OP_N;
  check_hipblas(hipblasLtMatmulDescSetAttribute(plan.op_desc,
                                              HIPBLASLT_MATMUL_DESC_TRANSA,
                                              &transa,
                                              sizeof(transa)),
               "set transa");
  check_hipblas(hipblasLtMatmulDescSetAttribute(plan.op_desc,
                                              HIPBLASLT_MATMUL_DESC_TRANSB,
                                              &transb,
                                              sizeof(transb)),
               "set transb");

  // The task contract is A row-major [M,K], B column-major [K,N], C row-major [M,N].
  // hipBLASLt sees this as C^T[N,M] = B^T[N,K] * A^T[K,M] in default column-major layouts.
  check_hipblas(hipblasLtMatrixLayoutCreate(&plan.a_desc, HIP_R_16F, k, n, k), "create A desc from task B");
  check_hipblas(hipblasLtMatrixLayoutCreate(&plan.b_desc, HIP_R_16F, k, m, k), "create B desc from task A");
  check_hipblas(hipblasLtMatrixLayoutCreate(&plan.c_desc, HIP_R_32F, n, m, n), "create C desc");
  check_hipblas(hipblasLtMatrixLayoutCreate(&plan.d_desc, HIP_R_32F, n, m, n), "create D desc");

  check_hipblas(hipblasLtMatmulPreferenceCreate(&plan.preference), "hipblasLtMatmulPreferenceCreate");
  check_hipblas(hipblasLtMatmulPreferenceSetAttribute(plan.preference,
                                                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &workspace_bytes,
                                                    sizeof(workspace_bytes)),
               "set workspace preference");

  const int heuristic_count = std::max(1, requested_heuristics);
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_results(static_cast<size_t>(heuristic_count));
  check_hipblas(hipblasLtMatmulAlgoGetHeuristic(plan.handle,
                                              plan.op_desc,
                                              plan.a_desc,
                                              plan.b_desc,
                                              plan.c_desc,
                                              plan.d_desc,
                                              plan.preference,
                                              heuristic_count,
                                              heuristic_results.data(),
                                              &plan.heuristic_results_returned),
               "hipblasLtMatmulAlgoGetHeuristic");
  if (plan.heuristic_results_returned <= 0) {
    std::cerr << "No hipBLASLt heuristic algorithm returned for this contract.\n";
    destroy_plan(&plan);
    std::exit(1);
  }
  plan.algo = heuristic_results.front().algo;
  const hipblasPointerMode_t pointer_mode = HIPBLAS_POINTER_MODE_HOST;
  check_hipblas(hipblasLtMatmulDescSetAttribute(plan.op_desc,
                                              HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                              &pointer_mode,
                                              sizeof(pointer_mode)),
               "set pointer mode");
  return plan;
}

void run_matmul(LtPlan* plan,
                const half* task_a,
                const half* task_b,
                float* task_c,
                void* workspace,
                size_t workspace_bytes,
                hipStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_hipblas(hipblasLtMatmul(plan->handle,
                              plan->op_desc,
                              &alpha,
                              task_b,
                              plan->a_desc,
                              task_a,
                              plan->b_desc,
                              &beta,
                              task_c,
                              plan->c_desc,
                              task_c,
                              plan->d_desc,
                              &plan->algo,
                              workspace,
                              workspace_bytes,
                              stream),
               "hipblasLtMatmul");
}

}  // namespace

int main(int argc, char** argv) {
  const int m = argc > 1 ? std::atoi(argv[1]) : 256;
  const int n = argc > 2 ? std::atoi(argv[2]) : 256;
  const int k = argc > 3 ? std::atoi(argv[3]) : 256;
  const int warmup_iters = argc > 4 ? std::atoi(argv[4]) : 10;
  const int measure_iters = argc > 5 ? std::atoi(argv[5]) : 50;
  const size_t workspace_bytes = argc > 6 ? static_cast<size_t>(std::strtoull(argv[6], nullptr, 10)) : 33554432ull;
  const int heuristic_count = argc > 7 ? std::atoi(argv[7]) : 16;
  const double abs_tolerance = argc > 8 ? std::atof(argv[8]) : std::max(0.05, 0.0005 * static_cast<double>(k));
  const double rel_tolerance = argc > 9 ? std::atof(argv[9]) : 0.02;

  if (m <= 0 || n <= 0 || k <= 0 || warmup_iters < 0 || measure_iters <= 0 || heuristic_count <= 0) {
    std::cerr << "usage: hipblaslt_hgemm [M>0] [N>0] [K>0] [warmup>=0] [iters>0] "
              << "[workspace_bytes>=0] [heuristic_count>0] [abs_tol] [rel_tol]\n";
    return 2;
  }

  const size_t a_elems = static_cast<size_t>(m) * k;
  const size_t b_elems = static_cast<size_t>(n) * k;
  const size_t c_elems = static_cast<size_t>(m) * n;
  const size_t a_bytes = a_elems * sizeof(half);
  const size_t b_bytes = b_elems * sizeof(half);
  const size_t c_bytes = c_elems * sizeof(float);

  std::vector<half> host_a(a_elems);
  std::vector<half> host_b(b_elems);
  std::vector<float> host_c(c_elems, 0.0f);
  std::vector<float> reference(c_elems, 0.0f);
  fill_half_inputs(&host_a, 17);
  fill_half_inputs(&host_b, 29);
  cpu_reference(host_a, host_b, &reference, m, n, k);

  half* device_a = nullptr;
  half* device_b = nullptr;
  float* device_c = nullptr;
  void* workspace = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  hipDeviceProp_t prop{};

  int device = 0;
  check_hip(hipGetDevice(&device), "hipGetDevice");
  check_hip(hipGetDeviceProperties(&prop, device), "hipGetDeviceProperties");
  check_hip(hipStreamCreate(&stream), "hipStreamCreate");
  check_hip(hipEventCreate(&start), "hipEventCreate(start)");
  check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
  check_hip(hipMalloc(&device_a, a_bytes), "hipMalloc(a)");
  check_hip(hipMalloc(&device_b, b_bytes), "hipMalloc(b)");
  check_hip(hipMalloc(&device_c, c_bytes), "hipMalloc(c)");
  if (workspace_bytes > 0) {
    check_hip(hipMalloc(&workspace, workspace_bytes), "hipMalloc(workspace)");
  }
  check_hip(hipMemcpyAsync(device_a, host_a.data(), a_bytes, hipMemcpyHostToDevice, stream), "copy a");
  check_hip(hipMemcpyAsync(device_b, host_b.data(), b_bytes, hipMemcpyHostToDevice, stream), "copy b");
  check_hip(hipMemsetAsync(device_c, 0, c_bytes, stream), "clear c");
  check_hip(hipStreamSynchronize(stream), "setup sync");

  LtPlan plan = create_plan(m, n, k, workspace_bytes, heuristic_count, stream);

  for (int i = 0; i < warmup_iters; ++i) {
    run_matmul(&plan, device_a, device_b, device_c, workspace, workspace_bytes, stream);
  }
  check_hip(hipStreamSynchronize(stream), "warmup sync");

  std::vector<float> timings_ms;
  timings_ms.reserve(static_cast<size_t>(measure_iters));
  for (int i = 0; i < measure_iters; ++i) {
    check_hip(hipEventRecord(start, stream), "record start");
    run_matmul(&plan, device_a, device_b, device_c, workspace, workspace_bytes, stream);
    check_hip(hipEventRecord(stop, stream), "record stop");
    check_hip(hipEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "elapsed");
    timings_ms.push_back(elapsed_ms);
  }

  check_hip(hipMemcpyAsync(host_c.data(), device_c, c_bytes, hipMemcpyDeviceToHost, stream), "copy c");
  check_hip(hipStreamSynchronize(stream), "final sync");

  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  size_t bad_count = 0;
  for (size_t i = 0; i < c_elems; ++i) {
    const double actual = static_cast<double>(host_c[i]);
    const double expected = static_cast<double>(reference[i]);
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(1.0, std::abs(expected));
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
    if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
      ++bad_count;
    }
  }

  std::sort(timings_ms.begin(), timings_ms.end());
  const float median_ms = percentile(timings_ms, 0.5);
  const double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
  const double gflop_per_s = flops / (static_cast<double>(median_ms) * 1.0e6);
  const double minimum_logical_gib =
      static_cast<double>(a_bytes + b_bytes + c_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double minimum_logical_gib_per_s = minimum_logical_gib / (static_cast<double>(median_ms) / 1000.0);
  const bool passed = bad_count == 0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "{\n"
            << "  \"schema_version\": \"0.1.0\",\n"
            << "  \"task_id\": \"rocwmma-mfma-gemm\",\n"
            << "  \"baseline_id\": \"hipblaslt-hgemm\",\n"
            << "  \"variant\": \"hipblaslt\",\n"
            << "  \"operation\": \"rocwmma-hgemm-f32-accumulate\",\n"
            << "  \"evidence_label\": \"timing-only\",\n"
            << "  \"scope\": \"hipBLASLt steady-state matmul only; descriptors and heuristic search are setup costs\",\n"
            << "  \"layout_mapping\": \"task C row-major [M,N] is computed as column-major C^T[N,M] = task B^T * task A^T\",\n"
            << "  \"layout_a\": \"task A row-major fp16 [M,K] reinterpreted as hipBLASLt column-major [K,M] operand B\",\n"
            << "  \"layout_b\": \"task B column-major fp16 [K,N] used as hipBLASLt column-major [K,N] operand A with transa=T\",\n"
            << "  \"layout_c\": \"task C row-major fp32 [M,N] reinterpreted as hipBLASLt column-major [N,M]\",\n"
            << "  \"dtype_a\": \"fp16\",\n"
            << "  \"dtype_b\": \"fp16\",\n"
            << "  \"dtype_accumulator\": \"fp32\",\n"
            << "  \"dtype_output\": \"fp32\",\n"
            << "  \"math_mode\": \"HIPBLAS_COMPUTE_32F with HIP_R_32F scale type\",\n"
            << "  \"device_name\": \"" << prop.name << "\",\n"
            << "  \"gfx_target\": \"" << prop.gcnArchName << "\",\n"
            << "  \"m\": " << m << ",\n"
            << "  \"n\": " << n << ",\n"
            << "  \"k\": " << k << ",\n"
            << "  \"warmup_iters\": " << warmup_iters << ",\n"
            << "  \"measure_iters\": " << measure_iters << ",\n"
            << "  \"workspace_bytes\": " << workspace_bytes << ",\n"
            << "  \"heuristic_results_requested\": " << heuristic_count << ",\n"
            << "  \"heuristic_results_returned\": " << plan.heuristic_results_returned << ",\n"
            << "  \"median_ms\": " << median_ms << ",\n"
            << "  \"p10_ms\": " << percentile(timings_ms, 0.1) << ",\n"
            << "  \"p90_ms\": " << percentile(timings_ms, 0.9) << ",\n"
            << "  \"min_ms\": " << timings_ms.front() << ",\n"
            << "  \"max_ms\": " << timings_ms.back() << ",\n"
            << "  \"gflop_per_s\": " << gflop_per_s << ",\n"
            << "  \"minimum_logical_gib_per_s\": " << minimum_logical_gib_per_s << ",\n"
            << "  \"abs_tolerance\": " << abs_tolerance << ",\n"
            << "  \"rel_tolerance\": " << rel_tolerance << ",\n"
            << "  \"max_abs_error\": " << max_abs_error << ",\n"
            << "  \"max_rel_error\": " << max_rel_error << ",\n"
            << "  \"bad_count\": " << bad_count << ",\n"
            << "  \"passed\": " << (passed ? "true" : "false") << "\n"
            << "}\n";

  destroy_plan(&plan);
  hipFree(workspace);
  hipFree(device_a);
  hipFree(device_b);
  hipFree(device_c);
  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipStreamDestroy(stream);
  return passed ? 0 : 1;
}
