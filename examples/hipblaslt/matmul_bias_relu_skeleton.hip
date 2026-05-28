#include <hip/hip_runtime.h>
#include <hipblasLt.h>

// Reference sketch: hipBLASLt matmul with an epilogue such as bias/ReLU.
// This file is intentionally not a complete benchmark yet. Use it as a map for
// the descriptors and metadata an agent should fill in before measuring.

struct LtMatmulPlan {
  hipblasLtHandle_t handle = nullptr;
  hipblasLtMatmulDesc_t op_desc = nullptr;
  hipblasLtMatrixLayout_t a_layout = nullptr;
  hipblasLtMatrixLayout_t b_layout = nullptr;
  hipblasLtMatrixLayout_t c_layout = nullptr;
  hipblasLtMatrixLayout_t d_layout = nullptr;
  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulAlgo_t algo{};
};

// Agent checklist:
// 1. Decide row-major wrapper versus native column-major dimensions.
// 2. Set compute type and scale type.
// 3. Set matrix layouts and leading dimensions.
// 4. Set epilogue attributes, e.g. bias pointer and ReLU where supported.
// 5. Query heuristics with workspace preference.
// 6. Benchmark multiple returned algorithms.
// 7. Record algorithm id, workspace, math mode, and epilogue.

