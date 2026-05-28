"""Template-only vLLM on ROCm baseline sketch.

Use this file to capture the serving contract before implementing a plugin:
model, precision, batching, KV-cache layout, sequence buckets, and timing
regions. It intentionally does not claim vLLM on ROCm measurements.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServingCase:
    name: str
    stage: str
    shape: str
    baseline: str


CASES = [
    ServingCase("rmsnorm_decode", "decode", "batch=1,hidden=4096", "vLLM on ROCm native plugin/kernel"),
    ServingCase("sampling", "decode", "batch=16,vocab=32000,top_k=4,top_p=0.9", "vLLM on ROCm or FlashInfer sampling"),
    ServingCase("paged_kv_update", "decode", "batch=32,heads=32,kv_heads=8,head_dim=128", "vLLM on ROCm paged KV path"),
]


def run_library_baseline(case: ServingCase) -> None:
    """TODO: invoke the vLLM on ROCm benchmark or minimal engine for this case."""
    raise NotImplementedError(f"Add vLLM on ROCm baseline for {case.name}")


if __name__ == "__main__":
    for case in CASES:
        print(case)
