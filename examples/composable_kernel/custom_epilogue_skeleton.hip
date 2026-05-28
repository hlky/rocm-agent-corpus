// Reference sketch: Composable Kernel custom epilogue work starts by copying a nearby
// example from third_party/composable-kernel/examples and changing one axis at a time.
//
// Agent checklist:
// 1. Choose the closest example for architecture and datatype.
// 2. Keep the host reference and problem initialization simple.
// 3. First reproduce plain GEMM.
// 4. Swap the epilogue/output operator.
// 5. Add bias or activation broadcast with explicit axis documentation.
// 6. Benchmark against hipBLASLt when the epilogue is supported there.
// 7. Record tile shape, stages, schedule, arch, and Composable Kernel commit.
//
// Useful submodule locations:
// - third_party/composable-kernel/examples
// - third_party/composable-kernel/media/docs
// - third_party/composable-kernel/include/composable-kernel/epilogue

