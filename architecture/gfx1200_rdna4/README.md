# gfx1200 RDNA4 Lab

Use this lab for RDNA4 `gfx1200`-class consumer, workstation, or deployment
targets when the local ROCm stack supports them.

## Compile Profile

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx1200 -lineinfo kernel.hip -o kernel_gfx1200
hipcc --list-gpu-targets
```

Record exact GPU model, ROCm version, driver/runtime metadata, power and clock
state, and library versions. Consumer/workstation runs need temperature and
boost notes.

## Features To Verify

- WMMA/MFMA availability, low-precision support, and library coverage on the
  exact SKU.
- LDS tiling, vectorized memory access, cache locality, and occupancy limits.
- MIGraphX, hipBLASLt, Composable Kernel, Triton, and PyTorch/Inductor
  baselines for the same workload.

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx1200 | grep -Ei "v_mfma|v_wmma|ds_|global_load|global_store"
```

## Custom-Kernel Attack Surfaces

- Model-specific inference helpers: norms, RoPE, KV-cache transforms, logits
  processing, Top-K, top-p, sampling, and quant/dequant.
- Fused memory-bound chains that remove framework launches.
- Small batched GEMM, grouped GEMM, and low-precision GEMM-like kernels where
  one layout and shape family are fixed.

## Gotchas

- Keep RDNA4 workstation/deployment records separate from CDNA4 data-center
  records.
- Re-run baselines frequently; early custom wins can disappear as ROCm library
  support improves.
- Mark HIP-event-only measurements as `timing-only`.
