# Baseline Plan

- Capture codegen regression evidence by recording LLVM IR / AMD GCN ISA, amdclang++ resource usage, register drift, spills, and instruction mix changes.
- Compile baseline and optimized kernels with verbose amdclang++ output and save build logs.
- Record environment and version metadata before comparing variants.
