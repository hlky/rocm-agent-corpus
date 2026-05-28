# Evidence Checklist

- Exact GPU name and reported `gfx` target from `rocminfo`.
- ROCm toolkit, driver/runtime, hipcc, and library versions.
- Full compile command, offload target, warnings, and resource output.
- Correctness result for the same shape, dtype, layout, and tolerance.
- Timing boundary: isolated kernel, library call, framework call, or end-to-end graph.
- Profiler status: no profiler, profiler attempted, or attached rocprofiler/rocprof counters.
- Disassembly status when claiming MFMA, WMMA, LDS staging, or vectorized memory behavior.
