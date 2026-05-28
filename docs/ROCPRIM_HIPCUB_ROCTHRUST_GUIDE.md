# rocPRIM, hipCUB, rocThrust, and libcudacxx Guide

rocPRIM/hipCUB/rocThrust is the practical default for many non-GEMM ROCm primitives. Agents should
prefer these primitives over hand-written kernels unless fusion, unusual
operators, or fixed-shape specialization justify custom code.

## hipCUB

Use hipCUB for:

- Device reductions.
- Device scans.
- Segmented reductions.
- Radix sort.
- Select, unique, partition.
- Histogram.
- Block and warp primitives inside custom kernels.

Agent workflow:

1. Find the matching `hipcub::Device*`, `hipcub::Block*`, or `hipcub::Warp*` primitive.
2. Compare against the custom kernel with identical inputs and tolerances.
3. Record temporary storage size and stream behavior.
4. For custom kernels, use hipCUB block/warp primitives before writing raw shuffle
   logic unless the task is explicitly educational.

## rocThrust

Use rocThrust for:

- Rapid prototypes.
- Transform, reduce, scan, sort, gather/scatter compositions.
- Iterator-based expression of simple pipelines.

Risks:

- Temporary allocations may dominate small workloads.
- Chained algorithms may launch multiple kernels.
- Custom execution policies and allocators may be needed for production.

## libcudacxx

Use libcudacxx for:

- GPU-aware standard-library-like facilities.
- Barriers, atomics, and synchronization primitives.
- Type utilities and device-compatible building blocks.

## Corpus Tasks to Add

- hipCUB DeviceReduce versus custom block reduction.
- hipCUB DeviceScan versus custom scan.
- hipCUB radix sort versus rocThrust sort.
- rocThrust transform-reduce versus fused custom kernel.
- BlockReduce inside a custom softmax/layernorm kernel.

