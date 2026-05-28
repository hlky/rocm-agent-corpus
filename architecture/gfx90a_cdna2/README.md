# gfx90a CDNA2 Lab

Use this lab for CDNA2 GPUs such as MI200/MI250-class devices. Treat the lab as
an architecture-specific tuning checklist, not as a general ROCm summary.

## Compile Profile

Prefer explicit offload targets for measured records:

```bash
hipcc -O3 --std=c++17 --offload-arch=gfx90a -lineinfo kernel.hip -o kernel_gfx90a
hipcc --version
hipcc --list-gpu-targets
```

Record the full command line, ROCm version, driver/runtime metadata, GPU model,
clock/power state, and compiler resource output with every result.

## Features To Verify

- MFMA Matrix Core paths for FP16, BF16, FP32, and INT8 where the exact library
  and compiler stack support them.
- LDS tiling, bank-conflict behavior, and software-pipelined global-to-LDS
  staging using `s_waitcnt`, barriers, and double buffering.
- Wave-level reductions, scans, and shuffle patterns used by hipCUB/rocPRIM.
- Occupancy limits from VGPRs, SGPRs, LDS bytes, and launch bounds.
- No CDNA3/CDNA4 or RDNA-specific behavior unless a separate target is built.

Disassemble only when the claim depends on instruction use:

```bash
llvm-objdump --disassemble --triple=amdgcn-amd-amdhsa ./kernel_gfx90a | grep -Ei "v_mfma|ds_|s_waitcnt|s_barrier"
```

## Custom-Kernel Attack Surfaces

- Fixed-shape GEMM or batched GEMM with a fused epilogue that hipBLASLt or CK
  must otherwise run as separate work.
- Attention, normalization, softmax, scan, histogram, and Top-K kernels with
  known row sizes, masks, and dtype contracts.
- Layout conversion and transpose kernels where LDS tiling removes uncoalesced
  traffic.
- MIGraphX or PyTorch extension kernels that remove framework dispatch or
  allocation overhead.

## Gotchas

- `gfx90a` evidence is not evidence for RDNA2/RDNA3 cards or for CDNA3.
- TF32, BF16, FP16, and FP32 are separate math contracts; record tolerances and
  accumulator type.
- HIP-event-only timings are `timing-only` until rocprofiler/rocprof counters
  are attached.
- Library baselines should include hipBLAS/rocBLAS/hipBLASLt, Composable
  Kernel, hipCUB/rocPRIM/rocThrust, MIGraphX, Triton, or framework-generated
  code as appropriate.
