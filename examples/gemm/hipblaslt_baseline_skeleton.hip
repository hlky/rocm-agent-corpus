#include <hipblasLt.h>
#include <hip/hip_runtime.h>

// hipBLASLt baseline skeleton for GEMM competition tasks.
//
// This file is a descriptor map, not a complete benchmark. Fill it in when a
// task needs a fair hipBLASLt competitor for plain GEMM or a supported epilogue.

struct GemmLtBaselineMetadata {
  int m;
  int n;
  int k;
  int batch_count;
  int lda;
  int ldb;
  int ldc;
  int ldd;
  hipDataType a_type;
  hipDataType b_type;
  hipDataType c_type;
  hipDataType d_type;
  hipblasComputeType_t compute_type;
  size_t workspace_bytes;
  int heuristic_results_requested;
  int heuristic_results_returned;
  bool tf32_allowed;
  bool descriptor_creation_in_timing;
};

struct GemmLtBaselinePlan {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t op_desc = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatrixLayout_t d_layout = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulAlgo_t selected_algo{};
};

// Agent checklist:
// 1. Decide whether the task is expressed as native column-major hipBLASLt
//    matrices or a row-major wrapper that swaps operands/transposes.
// 2. Create `hipblasLtMatmulDesc_t` with the exact compute type and scale type.
// 3. Set transpose flags, pointer mode, and epilogue attributes.
// 4. Create A/B/C/D matrix layouts with explicit leading dimensions and batch
//    strides.
// 5. Set workspace preference and query more than one heuristic candidate.
// 6. Time steady-state `hipblasLtMatmul` separately from descriptor creation
//    unless end-to-end latency is the task.
// 7. Record selected algorithm metadata, workspace cap, TF32 policy, epilogue,
//    ROCm/hipBLASLt versions, and the exact correctness tolerance.

