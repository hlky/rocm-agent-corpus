# Multi-GPU Guide

Multi-GPU optimization is usually about communication, topology, overlap, and
partitioning rather than a single kernel.

## RCCL

Use RCCL for:

- All-reduce.
- All-gather.
- Reduce-scatter.
- Broadcast.
- Point-to-point send/recv in distributed workflows.

Agent checks:

- Which collective is mathematically required?
- Is data contiguous?
- Is communication overlapped with compute?
- What topology is available: PCIe, NVLink, InfiniBand?
- Which RCCL environment variables were set?

## rocSHMEM

Use rocSHMEM for:

- GPU-initiated puts/gets.
- One-sided communication.
- Fine-grained producer/consumer patterns.
- Algorithms where CPU-mediated collectives are too rigid.

## Corpus Tasks to Add

- Single-node two-GPU all-reduce microbenchmark.
- Overlap GEMM with RCCL all-reduce.
- rocSHMEM put/get latency and bandwidth.
- Multi-GPU stencil halo exchange.
- Pipeline parallel communication/computation overlap.

## Measurement Notes

Record:

- GPU model and count.
- Interconnect topology.
- Process placement.
- RCCL version and environment variables.
- Message size sweep.
- Whether timing includes synchronization and host overhead.

