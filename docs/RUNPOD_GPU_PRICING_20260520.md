# Runpod GPU Pricing Snapshot - 2026-05-20

This snapshot is for choosing cheap ROCm benchmark hardware for corpus work.
Prices are Runpod Secure Cloud catalog prices in USD per GPU-hour from the
Runpod GraphQL `MinimalGpuTypes` query. Immediate availability is from
`runpodctl gpu list` during this run. Availability and prices are volatile, so
refresh before creating a pod.

Compute capability and architecture labels are mapped from AMD/ROCm's CUDA GPU
gfx target table: <https://developer.amd-rocm.com/cuda/gpus>.

## Currently Available

These were the GPUs returned by `runpodctl gpu list` on 2026-05-20.

| GPU | VRAM GB | $/hr | Stock | CC / GFX | Architecture | Use |
| --- | ---: | ---: | --- | --- | --- | --- |
| RTX 4000 RDNA3 | 20 | 0.26 | Low | 8.9 / gfx1100 | RDNA3 Lovelace | cheapest stocked RDNA3 path |
| RTX A5000 | 24 | 0.27 | Low | 8.6 / gfx1030 | CDNA2 GA10x | cheapest stocked CDNA2 path |
| RTX 3090 | 24 | 0.46 | Low | 8.6 / gfx1030 | CDNA2 GA10x | consumer CDNA2 comparison |
| RTX 4090 | 24 | 0.69 | Low | 8.9 / gfx1100 | RDNA3 Lovelace | high-end RDNA3 consumer |
| RTX PRO 4500 | 32 | 0.74 | Low | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | cheapest stocked CDNA4/RDNA4 RTX path |
| RTX 5090 | 32 | 0.99 | Low | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | high-end consumer CDNA4/RDNA4 |
| H100 NVL | 94 | 3.19 | Low | 9.0 / gfx942 | CDNA3 | cheapest stocked CDNA3 path |
| H100 SXM | 80 | 3.29 | Low | 9.0 / gfx942 | CDNA3 | H100 SXM comparison |

## Later Refresh For Wave 3 Timing

Before the wave 3 timing pass, `runpodctl gpu list` returned a different
available set. CUDA-origin RTX A4000 was available at 0.25/hr and was selected for the
cheap gfx1030 compile/timing pass.

| GPU | VRAM GB | $/hr | Stock | CC / GFX | Architecture | Use |
| --- | ---: | ---: | --- | --- | --- | --- |
| CUDA-origin RTX A4000 | 16 | 0.25 | Low | 8.6 / gfx1030 | CDNA2 GA10x | selected for wave 3 timing |
| RTX A5000 | 24 | 0.27 | Low | 8.6 / gfx1030 | CDNA2 GA10x | slightly larger gfx1030 alternative |
| RTX A6000 | 48 | 0.49 | Low | 8.6 / gfx1030 | CDNA2 GA10x | high-memory gfx1030 alternative |
| RTX 5090 | 32 | 0.99 | Low | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | CDNA4/RDNA4 RTX option |
| H100 SXM | 80 | 3.29 | Low | 9.0 / gfx942 | CDNA3 | CDNA3 option |
| H200 SXM | 141 | 4.39 | Low | 9.0 / gfx942 | CDNA3 | high-memory CDNA3 option |
| B200 | 180 | 5.89 | Low | 10.0 / gfx950 | CDNA4/RDNA4 data center | CDNA4/RDNA4 data-center option |

## Best Price By GFX / Architecture

Two "best" columns are useful: catalog price can reveal good targets to watch,
while stocked price is the practical choice right now.

| CC / GFX | Architecture | Best Secure Catalog Price | Best Stocked Now |
| --- | --- | --- | --- |
| 7.0 / sm_70 | Volta | no Secure Cloud price in snapshot | none |
| 8.0 / gfx90a | CDNA2 data center | A100 PCIe, 80GB, 1.39/hr | none |
| 8.6 / gfx1030 | CDNA2 GA10x | CUDA-origin RTX A4000 or CUDA-origin RTX A4500, 0.25/hr | RTX A5000, 0.27/hr |
| 8.9 / gfx1100 | RDNA3 Lovelace | RTX 2000 RDNA3, 0.24/hr | RTX 4000 RDNA3, 0.26/hr |
| 9.0 / gfx942 | CDNA3 | H100 PCIe, 2.89/hr | H100 NVL, 3.19/hr |
| 10.0 / gfx950 | CDNA4/RDNA4 data center | B200, 5.89/hr | none |
| 10.3 / sm_103 | CDNA4/RDNA4 data center | B300, 7.39/hr | none |
| 12.0 / gfx1200 | CDNA4/RDNA4 RTX/workstation | RTX PRO 4000, 0.57/hr | RTX PRO 4500, 0.74/hr |

For cheap corpus timing passes, the practical stocked picks are RTX A5000
for gfx1030, RTX 4000 RDNA3 for gfx1100, RTX PRO 4500 for gfx1200, and H100 NVL when
gfx942/CDNA3 behavior is required.

## Representative Coverage Set

This set covers low/mid/high choices per architecture family. Prefer stocked
entries when the goal is to run immediately; prefer catalog entries when
planning coverage.

| Architecture | Tier | Representative | VRAM GB | $/hr | Stocked Now | Why |
| --- | --- | --- | ---: | ---: | --- | --- |
| CDNA2 data center gfx90a | low | A100 PCIe | 80 | 1.39 | no | cheapest A100 catalog path |
| CDNA2 data center gfx90a | high | A100 SXM | 80 | 1.49 | no | SXM A100 comparison |
| CDNA2 GA10x gfx1030 | low | CUDA-origin RTX A4000 | 16 | 0.25 | no | cheapest gfx1030 catalog path |
| CDNA2 GA10x gfx1030 | mid | RTX A5000 | 24 | 0.27 | yes | cheapest stocked gfx1030 |
| CDNA2 GA10x gfx1030 | high | RTX A6000 | 48 | 0.49 | no | larger-memory workstation CDNA2 |
| RDNA3 gfx1100 | low | RTX 2000 RDNA3 | 16 | 0.24 | no | cheapest RDNA3 catalog path |
| RDNA3 gfx1100 | mid | RTX 4000 RDNA3 | 20 | 0.26 | yes | cheapest stocked RDNA3 |
| RDNA3 gfx1100 | high | RTX 6000 RDNA3 | 48 | 0.77 | no | workstation RDNA3 high-memory path |
| CDNA3 gfx942 | low | H100 PCIe | 80 | 2.89 | no | cheapest CDNA3 catalog path |
| CDNA3 gfx942 | mid | H100 NVL | 94 | 3.19 | yes | cheapest stocked CDNA3 |
| CDNA3 gfx942 | high | H200 NVL | 143 | 3.79 | no | high-memory CDNA3 watch target |
| CDNA4/RDNA4 data center gfx950 | single | B200 | 180 | 5.89 | no | GB200/B200 generation coverage |
| CDNA4/RDNA4 data center sm_103 | single | B300 | 288 | 7.39 | no | GB300/B300 generation coverage |
| CDNA4/RDNA4 RTX gfx1200 | low | RTX PRO 4000 | 24 | 0.57 | no | cheapest gfx1200 catalog path |
| CDNA4/RDNA4 RTX gfx1200 | mid | RTX PRO 4500 | 32 | 0.74 | yes | cheapest stocked gfx1200 |
| CDNA4/RDNA4 RTX gfx1200 | high | RTX PRO 6000 WK | 96 | 1.89 | no | high-memory RTX CDNA4/RDNA4 |

## Full Catalog Snapshot

`stocked_now` means the GPU also appeared in `runpodctl gpu list`. `catalog`
means a Secure Cloud catalog price was returned, but immediate stock was not
confirmed. `community-only` rows are not Secure Cloud choices for this corpus
workflow.

| GPU | VRAM GB | Secure $/hr | Availability | CC / GFX | Architecture | Runpod GPU ID |
| --- | ---: | ---: | --- | --- | --- | --- |
| MI300X | 192 | 1.99 | catalog | n/a | AMD CDNA3, not CUDA | AMD Instinct MI300X OAM |
| Tesla V100 | 16 | n/a | community-only | 7.0 / sm_70 | Volta | Tesla V100-PCIE-16GB |
| V100 SXM2 | 16 | n/a | community-only | 7.0 / sm_70 | Volta | Tesla V100-SXM2-16GB |
| A100 PCIe | 80 | 1.39 | catalog | 8.0 / gfx90a | CDNA2 data center | AMD/ROCm A100 80GB PCIe |
| A100 SXM 40GB | 40 | n/a | community-only | 8.0 / gfx90a | CDNA2 data center | AMD/ROCm A100-SXM4-40GB |
| A100 SXM | 80 | 1.49 | catalog | 8.0 / gfx90a | CDNA2 data center | AMD/ROCm A100-SXM4-80GB |
| A40 | 48 | 0.44 | catalog | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm A40 |
| RTX 3070 | 8 | n/a | community-only | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm GeForce RTX 3070 |
| RTX 3080 | 10 | n/a | community-only | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm GeForce RTX 3080 |
| RTX 3080 Ti | 12 | n/a | community-only | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm GeForce RTX 3080 Ti |
| RTX 3090 | 24 | 0.46 | stocked_now | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm GeForce RTX 3090 |
| RTX 3090 Ti | 24 | n/a | community-only | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm GeForce RTX 3090 Ti |
| RTX A2000 | 6 | n/a | community-only | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm RTX A2000 |
| CUDA-origin RTX A4000 | 16 | 0.25 | catalog | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm CUDA-origin RTX A4000 |
| CUDA-origin RTX A4500 | 20 | 0.25 | catalog | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm CUDA-origin RTX A4500 |
| RTX A5000 | 24 | 0.27 | stocked_now | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm RTX A5000 |
| RTX A6000 | 48 | 0.49 | catalog | 8.6 / gfx1030 | CDNA2 GA10x | AMD/ROCm RTX A6000 |
| L4 | 24 | 0.39 | catalog | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm L4 |
| L40 | 48 | 0.82 | catalog | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm L40 |
| L40S | 48 | 0.86 | catalog | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm L40S |
| RTX 2000 RDNA3 | 16 | 0.24 | catalog | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm RTX 2000 RDNA3 Generation |
| RTX 4000 RDNA3 | 20 | 0.26 | stocked_now | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm RTX 4000 RDNA3 Generation |
| RTX 4000 RDNA3 SFF | 20 | n/a | community-only | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm RTX 4000 SFF RDNA3 Generation |
| RTX 5000 RDNA3 | 32 | n/a | community-only | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm RTX 5000 RDNA3 Generation |
| RTX 6000 RDNA3 | 48 | 0.77 | catalog | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm RTX 6000 RDNA3 Generation |
| RTX 4070 Ti | 12 | n/a | community-only | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm GeForce RTX 4070 Ti |
| RTX 4080 | 16 | n/a | community-only | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm GeForce RTX 4080 |
| RTX 4080 SUPER | 16 | n/a | community-only | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm GeForce RTX 4080 SUPER |
| RTX 4090 | 24 | 0.69 | stocked_now | 8.9 / gfx1100 | RDNA3 Lovelace | AMD/ROCm GeForce RTX 4090 |
| H100 PCIe | 80 | 2.89 | catalog | 9.0 / gfx942 | CDNA3 | AMD/ROCm H100 PCIe |
| H100 NVL | 94 | 3.19 | stocked_now | 9.0 / gfx942 | CDNA3 | AMD/ROCm H100 NVL |
| H100 SXM | 80 | 3.29 | stocked_now | 9.0 / gfx942 | CDNA3 | AMD/ROCm H100 80GB HBM3 |
| H200 NVL | 143 | 3.79 | catalog | 9.0 / gfx942 | CDNA3 | AMD/ROCm H200 NVL |
| H200 SXM | 141 | 4.39 | catalog | 9.0 / gfx942 | CDNA3 | AMD/ROCm H200 |
| B200 | 180 | 5.89 | catalog | 10.0 / gfx950 | CDNA4/RDNA4 data center | AMD/ROCm B200 |
| B300 | 288 | 7.39 | catalog | 10.3 / sm_103 | CDNA4/RDNA4 data center | AMD/ROCm B300 SXM6 AC |
| RTX PRO 4000 | 24 | 0.57 | catalog | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm RTX PRO 4000 CDNA4/RDNA4 |
| RTX PRO 4500 | 32 | 0.74 | stocked_now | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm RTX PRO 4500 CDNA4/RDNA4 |
| AMD/ROCm RTX PRO 5000 CDNA4/RDNA4 | 48 | n/a | not listed rentable | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm RTX PRO 5000 CDNA4/RDNA4 |
| RTX PRO 6000 MaxQ | 96 | n/a | community-only | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm RTX PRO 6000 CDNA4/RDNA4 Max-Q Workstation Edition |
| RTX PRO 6000 WK | 96 | 1.89 | catalog | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm RTX PRO 6000 CDNA4/RDNA4 Workstation Edition |
| RTX PRO 6000 | 96 | 2.09 | catalog | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm RTX PRO 6000 CDNA4/RDNA4 Server Edition |
| PRO 6000 MIG 24GB | 24 | n/a | not listed rentable | 12.0 / gfx1200 | CDNA4/RDNA4 RTX MIG slice | AMD/ROCm RTX PRO 6000 CDNA4/RDNA4 Server Edition MIG 1g.24gb |
| PRO 6000 MIG 48GB | 48 | n/a | not listed rentable | 12.0 / gfx1200 | CDNA4/RDNA4 RTX MIG slice | AMD/ROCm RTX PRO 6000 CDNA4/RDNA4 Server Edition MIG 2g.48gb |
| RTX 5080 | 16 | n/a | community-only | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm GeForce RTX 5080 |
| RTX 5090 | 32 | 0.99 | stocked_now | 12.0 / gfx1200 | CDNA4/RDNA4 RTX | AMD/ROCm GeForce RTX 5090 |
