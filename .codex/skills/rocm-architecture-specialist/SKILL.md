---
name: rocm-architecture-specialist
description: Select gfx-target-specific tuning paths, compile flags, and architecture references for ROCm work.
---

# HIP/ROCm Architecture Specialist

Use this skill when a task depends on GPU architecture, gfx target,
Matrix Core generation, memory hierarchy, or architecture-specific HIP/ROCm
features.

## Workflow

1. Run `python tools/inspect_rocm_arch.py` when local ROCm tools are available.
2. Read `docs/GPU_GFX_ARCHITECTURE_GUIDE.md`.
3. Check `data/index/gpu_architectures.json`.
4. Use official tuning-guide source entries in `data/sources/core_resources.json`.
5. Record `hipcc` architecture flags in every benchmark result.

## Rules

- Distinguish `gfx90a`, `gfx1030`, `gfx1100`, `gfx942`, `gfx950`, and CDNA4/RDNA4 GFXs.
- Do not use architecture-specific suffixes unless the deployment target allows
  them.
- Do not assume profiler counters are available on hosted GPUs.
- Prefer Composable Kernel/CK Tile examples for CDNA3/CDNA4/RDNA4 Matrix Core and global-to-LDS staging-style
  work.

