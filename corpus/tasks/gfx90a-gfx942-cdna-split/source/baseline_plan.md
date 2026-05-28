# Baseline Plan

- Separate CDNA2 data-center gfx90a assumptions from consumer/pro gfx1030 behavior for global-to-LDS staging, TF32, cache, and library tactic comparisons.
- Compile portable CUDA and library baselines for each CDNA2 target with explicit math modes and LLVM IR / AMD GCN ISA flags.
- Record environment and version metadata before comparing variants.
