"""Template-only MIOpen fusion baseline sketch.

Fill this with a real cudnn-frontend Python or C++ graph construction before
recording correctness or timing evidence. Keep plan build time separate from
steady-state execution latency.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionCase:
    name: str
    shape: str
    dtype: str
    layout: str
    baseline: str


CASES = [
    FusionCase("conv_bias_relu", "N=32,H=56,W=56,C=64", "fp16", "NHWC", "MIOpen operation graph"),
    FusionCase("ln_activation", "B=16,S=2048,H=4096", "bf16", "row-major", "MIOpen pointwise/reduction graph"),
]


def build_cudnn_frontend_plan(case: FusionCase) -> None:
    """TODO: construct tensors, operations, heuristics, and execution plan."""
    raise NotImplementedError(f"Add MIOpen graph construction for {case.name}")


if __name__ == "__main__":
    for case in CASES:
        print(case)
