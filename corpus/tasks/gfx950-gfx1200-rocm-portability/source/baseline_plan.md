# Baseline Plan

- Capture `rocminfo`, `hipcc --version`, and `hipcc --list-gpu-targets`.
- Build a portable HIP baseline for the task under test with the exact reported `gfx` target.
- Run the strongest library baseline available for the same contract: hipBLASLt/rocBLAS, Composable Kernel, hipCUB/rocPRIM/rocThrust, MIGraphX, Triton, or PyTorch/Inductor.
- Store compile logs and library versions before comparing gfx950 and gfx1200 behavior.
