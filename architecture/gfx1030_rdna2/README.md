# gfx1030 RDNA2 Lab

Use this lab for RDNA2 `gfx1030` targets when ROCm support is available on the
measured host. This is mostly a memory, launch-overhead, and portability lab;
do not assume CDNA Matrix Core behavior.

## Compile Profile

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx1030 -lineinfo kernel.hip -o kernel_gfx1030
hipcc --list-gpu-targets
```

Attach GPU model, ROCm version, driver/runtime metadata, clocks, and thermal or
power notes to every measured record.

## Features To Verify

- Vectorized global loads/stores, alignment, cache behavior, and LDS tiling.
- Wave32 versus Wave64 assumptions when the kernel uses warp-like logic.
- hipCUB/rocPRIM behavior for reductions, scans, select, sort, and histogram.
- Any WMMA/MFMA-like path only after compiler output and disassembly prove it.

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx1030 | grep -Ei "v_mfma|v_wmma|ds_|global_load|global_store"
```

## Custom-Kernel Attack Surfaces

- Launch-overhead-limited kernels: short-row softmax, norms, small reductions,
  tiny GEMM-like transforms, and fused elementwise chains.
- Layout conversion, quant/dequant, packing, unpacking, and gather/scatter with
  strict alignment and shape contracts.
- Framework extension kernels where one HIP launch replaces several PyTorch or
  Triton launches.

## Gotchas

- RDNA2 results should not be generalized to CDNA2 data-center GPUs.
- If a library lacks a tuned path for the exact SKU, record that as part of the
  baseline rather than treating the custom kernel as architecture-general.
- Mark HIP-event-only measurements as `timing-only`.
