"""Triton row-wise softmax skeleton.

This is a navigation example, not a locked benchmark. Agents should compare a
filled-in Triton implementation against corpus/tasks/rowwise-softmax and record
compile time, runtime, meta-parameters, and generated code evidence.
"""


def checklist() -> list[str]:
    return [
        "choose BLOCK_SIZE as power-of-two >= cols",
        "record num_warps and num_stages",
        "warm up separately from timing",
        "compare against HIP C++ and framework baselines",
        "inspect generated LLVM IR / AMD GCN ISA before claiming architecture-specific behavior",
    ]

