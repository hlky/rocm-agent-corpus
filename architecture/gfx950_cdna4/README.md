# gfx950 CDNA4 Lab

Use this lab for `gfx950`-class CDNA4 targets. Because ROCm library and compiler
support can evolve quickly here, every record must carry exact toolkit and
library versions.

## Compile Profile

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx950 -lineinfo kernel.hip -o kernel_gfx950
hipcc --list-gpu-targets
```

If the local toolkit exposes suffix targets, record them as separate contracts
instead of merging them into generic `gfx950` evidence.

## Features To Verify

- MFMA/WMMA instruction families and any new low-precision modes exposed by the
  installed compiler and libraries.
- Composable Kernel/CK Tile collectives, hipBLASLt algorithms, and MIGraphX or
  vLLM on ROCm paths for the same workload.
- LDS staging depth, barriers, register pressure, occupancy, and memory
  transaction assumptions.
- Multi-GPU topology, RCCL version, and rocSHMEM behavior for overlap tasks.

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx950 | grep -Ei "v_mfma|v_wmma|ds_|s_waitcnt|s_barrier"
```

## Custom-Kernel Attack Surfaces

- Fixed-shape GEMM, grouped GEMM, custom CK epilogues, and fused low-precision
  scale/amax movement.
- Attention, MoE, paged KV-cache, logits, and sampling kernels where model
  shapes are fixed.
- RCCL/rocSHMEM overlap and persistent inference scheduling.

## Gotchas

- Re-run vendor baselines after ROCm or library upgrades.
- Do not infer support for an instruction or dtype from the GFX number alone.
- Mark HIP-event-only results as `timing-only` unless profiler counters are
  attached.
