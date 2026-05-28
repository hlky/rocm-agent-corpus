#!/usr/bin/env python3
"""Benchmark a PyTorch eager baseline for block top-k temperature sampling."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "block-topk-sampling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--tie-epsilon", type=float, default=1.0e-9)
    parser.add_argument("--hardware-record", default="")
    parser.add_argument("--write-result", action="store_true")
    return parser.parse_args()


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def load_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this baseline runner") from exc
    return torch


def make_inputs(torch: Any, rows: int, vocab_size: int, device: Any) -> tuple[Any, Any]:
    row_ids = torch.arange(rows, device=device, dtype=torch.int64).unsqueeze(1)
    col_ids = torch.arange(vocab_size, device=device, dtype=torch.int64).unsqueeze(0)
    pattern = (row_ids * 131 + col_ids * 17 + torch.div(col_ids, 97, rounding_mode="floor") * 23) % 509
    tie_band = torch.where((col_ids % 257) == 0, 24, 0)
    logits = (pattern - 254 + tie_band).to(torch.float32) * 0.03125
    buckets = torch.arange(rows, device=device, dtype=torch.int64) & 3
    uniforms = torch.where(
        buckets == 0,
        torch.full((rows,), 0.05, device=device, dtype=torch.float32),
        torch.where(
            buckets == 1,
            torch.full((rows,), 0.35, device=device, dtype=torch.float32),
            torch.where(
                buckets == 2,
                torch.full((rows,), 0.65, device=device, dtype=torch.float32),
                torch.full((rows,), 0.95, device=device, dtype=torch.float32),
            ),
        ),
    )
    return logits, uniforms


def pytorch_topk_sample(torch: Any, logits: Any, uniforms: Any, k: int, temperature: float, tie_epsilon: float) -> tuple[Any, Any, Any, Any]:
    vocab_size = logits.shape[1]
    tie_bias = torch.arange(vocab_size, device=logits.device, dtype=torch.float64) * tie_epsilon
    adjusted = logits.to(torch.float64) - tie_bias.unsqueeze(0)
    _, topk_indices = torch.topk(adjusted, k=k, dim=1, largest=True, sorted=True)
    topk_values = torch.gather(logits, 1, topk_indices)
    topk_probs = torch.softmax(topk_values / temperature, dim=1)
    cdf = torch.cumsum(topk_probs, dim=1)
    positions = torch.sum(uniforms.unsqueeze(1) > cdf, dim=1).clamp(max=k - 1)
    sampled = torch.gather(topk_indices, 1, positions.unsqueeze(1)).squeeze(1)
    return sampled, topk_indices, topk_values, topk_probs


def make_result(args: argparse.Namespace) -> dict[str, Any]:
    torch = load_torch()
    if not torch.cuda.is_available():
        raise SystemExit("A PyTorch CUDA/HIP backend is required for PyTorch baseline timing")
    if args.rows <= 0 or args.vocab_size <= 0 or args.k <= 0 or args.k > args.vocab_size:
        raise SystemExit("Require rows > 0, vocab_size > 0, and 0 < k <= vocab_size")
    if args.temperature <= 0.0 or args.warmup < 0 or args.iters <= 0:
        raise SystemExit("Require temperature > 0, warmup >= 0, and iters > 0")

    device = torch.device("cuda")
    logits, uniforms = make_inputs(torch, args.rows, args.vocab_size, device)

    timings: list[float] = []
    outputs: tuple[Any, Any, Any, Any] | None = None
    for iteration in range(args.warmup + args.iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        outputs = pytorch_topk_sample(torch, logits, uniforms, args.k, args.temperature, args.tie_epsilon)
        stop.record()
        stop.synchronize()
        elapsed = float(start.elapsed_time(stop))
        if iteration >= args.warmup:
            timings.append(elapsed)

    assert outputs is not None
    sampled, topk_indices, topk_values, topk_probs = outputs

    cpu_logits, cpu_uniforms = make_inputs(torch, args.rows, args.vocab_size, torch.device("cpu"))
    cpu_sampled, cpu_topk_indices, cpu_topk_values, cpu_topk_probs = pytorch_topk_sample(
        torch,
        cpu_logits,
        cpu_uniforms,
        args.k,
        args.temperature,
        args.tie_epsilon,
    )
    torch.cuda.synchronize()
    topk_indices_cpu = topk_indices.cpu()
    sampled_cpu = sampled.cpu()
    topk_values_cpu = topk_values.cpu()
    topk_probs_cpu = topk_probs.cpu()

    topk_index_mismatch_count = int((topk_indices_cpu != cpu_topk_indices).sum().item())
    sample_mismatch_count = int((sampled_cpu != cpu_sampled).sum().item())
    max_topk_value_abs_error = float((topk_values_cpu - cpu_topk_values).abs().max().item())
    max_probability_abs_error = float((topk_probs_cpu - cpu_topk_probs).abs().max().item())
    passed = (
        topk_index_mismatch_count == 0
        and sample_mismatch_count == 0
        and max_topk_value_abs_error <= 1.0e-6
        and max_probability_abs_error <= 1.0e-4
    )

    sorted_timings = sorted(timings)
    logical_bytes = (
        args.rows * args.vocab_size * 4
        + args.rows * 4
        + args.rows * 4
        + args.rows * args.k * 4
        + 2 * args.rows * args.k * 4
    )
    logical_gib = logical_bytes / (1024.0 * 1024.0 * 1024.0)
    median_ms = percentile(sorted_timings, 0.5)
    props = torch.cuda.get_device_properties(device)
    torch_rocm_version = getattr(torch.version, "hip", None)
    runtime_version = torch_rocm_version or torch.version.cuda or ""

    result: dict[str, Any] = {
        "schema_version": "0.1.0",
        "task_id": TASK_ID,
        "variant": "pytorch-eager",
        "baseline_id": "pytorch-eager-topk-softmax-sample-tie-adjusted",
        "operation": "top-k-temperature-sampling",
        "evidence_label": "timing-only",
        "rows": args.rows,
        "vocab_size": args.vocab_size,
        "k": args.k,
        "temperature": args.temperature,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "median_ms": median_ms,
        "p10_ms": percentile(sorted_timings, 0.1),
        "p90_ms": percentile(sorted_timings, 0.9),
        "min_ms": min(sorted_timings),
        "max_ms": max(sorted_timings),
        "logical_gib_per_s": logical_gib / (median_ms / 1000.0),
        "topk_index_mismatch_count": topk_index_mismatch_count,
        "sample_mismatch_count": sample_mismatch_count,
        "max_topk_value_abs_error": max_topk_value_abs_error,
        "max_probability_abs_error": max_probability_abs_error,
        "passed": passed,
        "timer_scope": (
            "HIP events around eager PyTorch tensor operations for tie-adjusted topk, gather, "
            "softmax, cumsum, and sample selection; input construction and CPU reference are excluded"
        ),
        "tie_policy": (
            "The seed requires lower token id to win equal logits. This PyTorch baseline subtracts "
            "index * tie_epsilon from a float64 copy before torch.topk, then gathers original fp32 logits."
        ),
        "framework": {
            "name": "pytorch",
            "torch_version": torch.__version__,
            "torch_backend": "rocm" if torch_rocm_version else "cuda",
            "torch_rocm_version": torch_rocm_version,
            "torch_cuda_version": torch.version.cuda,
            "rocm_or_cuda_runtime_version": runtime_version,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "device_name": props.name,
        "gfx_target": getattr(props, "gcnArchName", ""),
        "compute_capability": f"{props.major}.{props.minor}",
        "rocm_or_cuda_runtime_version": runtime_version,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": "tools/run_topk_sampling_pytorch_baseline.py",
        "run_command": [sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
    }
    if args.hardware_record:
        result["hardware_record"] = args.hardware_record
    return result


def main() -> int:
    args = parse_args()
    result = make_result(args)
    print(json.dumps(result, indent=2))
    if args.write_result:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results_dir = ROOT / "corpus" / "tasks" / TASK_ID / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / (
            f"pytorch-eager-{args.rows}x{args.vocab_size}-k{args.k}-t{args.temperature}-{stamp}.json"
        )
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
