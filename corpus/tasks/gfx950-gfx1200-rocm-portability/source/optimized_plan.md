# Optimized Plan

- Add target-specific code paths only after the baseline compiles and passes correctness.
- Keep `gfx950` and `gfx1200` dispatch separate, including target-specific tile sizes, wave-size assumptions, and low-precision contracts.
- For each optimized variant, record the custom-kernel edge: fixed shape, fused epilogue, reduced memory traffic, lower dispatch overhead, or verified instruction use.
- Re-run library baselines after ROCm upgrades before preserving a custom win.
