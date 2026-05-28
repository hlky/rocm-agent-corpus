"""Template-only Python baseline for pytorch-inductor-generated-kernel.

Summary: Capture the workflow for comparing handwritten HIP against PyTorch eager, torch.compile/Inductor, nvFuser-like fusion, and Triton-generated kernels.
Baseline strategy: Run eager PyTorch and torch.compile/Inductor, collect generated kernel/source metadata, and time with synchronized HIP events or benchmark utilities.

Replace this with the concrete framework/compiler baseline before recording
correctness or timing evidence.
"""


def run_baseline(*args, **kwargs):
    raise NotImplementedError("template-only baseline; no measurement claim")
