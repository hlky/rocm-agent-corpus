#!/usr/bin/env python3
"""Benchmark PyTorch eager/autograd baselines for normalization backward."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["layernorm", "rmsnorm"], required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--dtype", choices=["fp32"], default="fp32")
    parser.add_argument("--seed", type=int, default=20260525)
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
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit("PyTorch is required for this baseline runner") from exc
    return torch, F


def make_result(args: argparse.Namespace) -> dict[str, Any]:
    torch, F = load_torch()
    if not torch.cuda.is_available():
        raise SystemExit("A PyTorch CUDA/HIP backend is required for PyTorch baseline timing")

    if args.dtype != "fp32":
        raise SystemExit("Only fp32 is currently implemented")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32, requires_grad=True)
    dy = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32)
    gamma = torch.randn(args.cols, device=device, dtype=torch.float32, requires_grad=True)
    beta = None
    if args.mode == "layernorm":
        beta = torch.randn(args.cols, device=device, dtype=torch.float32, requires_grad=True)

    def clear_grads() -> None:
        x.grad = None
        gamma.grad = None
        if beta is not None:
            beta.grad = None

    def forward() -> Any:
        if args.mode == "layernorm":
            return F.layer_norm(x, (args.cols,), gamma, beta, args.epsilon)
        inv_rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + args.epsilon)
        return x * inv_rms * gamma

    timings: list[float] = []
    total_iters = args.warmup + args.iters
    for iteration in range(total_iters):
        clear_grads()
        y = forward()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        y.backward(dy)
        stop.record()
        stop.synchronize()
        elapsed = float(start.elapsed_time(stop))
        if iteration >= args.warmup:
            timings.append(elapsed)

    torch.cuda.synchronize()
    sorted_timings = sorted(timings)
    dx = x.grad
    dgamma = gamma.grad
    dbeta = beta.grad if beta is not None else None
    finite_dx = bool(torch.isfinite(dx).all().item()) if dx is not None else False
    finite_dgamma = bool(torch.isfinite(dgamma).all().item()) if dgamma is not None else False
    finite_dbeta = True if dbeta is None else bool(torch.isfinite(dbeta).all().item())
    passed = finite_dx and finite_dgamma and finite_dbeta

    props = torch.cuda.get_device_properties(device)
    torch_rocm_version = getattr(torch.version, "hip", None)
    runtime_version = torch_rocm_version or torch.version.cuda or ""
    baseline_id = (
        "pytorch-eager-layernorm-autograd"
        if args.mode == "layernorm"
        else "pytorch-eager-composite-rmsnorm-autograd"
    )
    result: dict[str, Any] = {
        "schema_version": "0.1.0",
        "task_id": "normalization-backward",
        "variant": "pytorch-eager-backward",
        "baseline_id": baseline_id,
        "operation": "normalization-backward",
        "evidence_label": "timing-only",
        "mode": args.mode,
        "dtype": args.dtype,
        "rows": args.rows,
        "cols": args.cols,
        "epsilon": args.epsilon,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "median_ms": percentile(sorted_timings, 0.5),
        "p10_ms": percentile(sorted_timings, 0.1),
        "p90_ms": percentile(sorted_timings, 0.9),
        "min_ms": min(sorted_timings),
        "max_ms": max(sorted_timings),
        "timer_scope": (
            "HIP events around backward() only; forward graph construction and input allocation "
            "are outside the timed region"
        ),
        "saved_statistics_policy": (
            "PyTorch autograd uses tensors saved by the eager forward; this result does not force "
            "a saved-statistics ABI matching the custom HIP harness"
        ),
        "passed": passed,
        "correctness_check": "finite dx/dgamma/dbeta gradients after the final measured iteration",
        "finite_gradients": {
            "dx": finite_dx,
            "dgamma": finite_dgamma,
            "dbeta": finite_dbeta,
        },
        "framework": {
            "name": "pytorch",
            "torch_version": torch.__version__,
            "torch_backend": "rocm" if torch_rocm_version else "cuda",
            "torch_rocm_version": torch_rocm_version,
            "torch_cuda_version": torch.version.cuda,
            "rocm_or_cuda_runtime_version": runtime_version,
            "cudnn_version": torch.backends.cudnn.version(),
            "backend": baseline_id,
        },
        "device_name": props.name,
        "gfx_target": getattr(props, "gcnArchName", ""),
        "compute_capability": f"{props.major}.{props.minor}",
        "rocm_or_cuda_runtime_version": runtime_version,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": "tools/run_normalization_pytorch_baseline.py",
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
        task_dir = ROOT / "corpus" / "tasks" / "normalization-backward"
        results_dir = task_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        path = results_dir / f"pytorch-eager-{args.rows}x{args.cols}-{args.mode}-{stamp}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
