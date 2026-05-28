#!/usr/bin/env python3
"""Benchmark PyTorch eager row-wise softmax for the rowwise-softmax task."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "rowwise-softmax"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
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


def make_input(torch: Any, rows: int, cols: int, device: Any) -> Any:
    idx = torch.arange(rows * cols, device=device, dtype=torch.int64)
    values = ((idx % 257).to(torch.float32) - 128.0) * 0.03125
    return values.reshape(rows, cols)


def make_result(args: argparse.Namespace) -> dict[str, Any]:
    torch = load_torch()
    if not torch.cuda.is_available():
        raise SystemExit("A PyTorch CUDA/HIP backend is required for PyTorch baseline timing")
    if args.rows <= 0 or args.cols <= 0 or args.warmup < 0 or args.iters <= 0:
        raise SystemExit("Require rows > 0, cols > 0, warmup >= 0, and iters > 0")

    device = torch.device("cuda")
    x = make_input(torch, args.rows, args.cols, device)

    timings: list[float] = []
    y = None
    for iteration in range(args.warmup + args.iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        y = torch.softmax(x, dim=1)
        stop.record()
        stop.synchronize()
        elapsed = float(start.elapsed_time(stop))
        if iteration >= args.warmup:
            timings.append(elapsed)

    assert y is not None
    cpu_x = make_input(torch, args.rows, args.cols, torch.device("cpu"))
    cpu_y = torch.softmax(cpu_x, dim=1)
    torch.cuda.synchronize()
    y_cpu = y.cpu()
    diff = (y_cpu - cpu_y).abs()
    row_sums = y_cpu.sum(dim=1)
    max_abs_error = float(diff.max().item())
    max_row_sum_error = float((row_sums - 1.0).abs().max().item())
    bad_count = int((diff > 1.0e-5).sum().item())
    passed = bad_count == 0 and max_row_sum_error <= 1.0e-4

    sorted_timings = sorted(timings)
    median_ms = percentile(sorted_timings, 0.5)
    byte_count = args.rows * args.cols * 4
    effective_gib_per_s = (2.0 * byte_count / (1024.0 * 1024.0 * 1024.0)) / (median_ms / 1000.0)
    props = torch.cuda.get_device_properties(device)
    torch_rocm_version = getattr(torch.version, "hip", None)
    runtime_version = torch_rocm_version or torch.version.cuda or ""

    result: dict[str, Any] = {
        "schema_version": "0.1.0",
        "task_id": TASK_ID,
        "variant": "pytorch-eager",
        "baseline_id": "pytorch-eager-softmax",
        "operation": "rowwise-softmax",
        "evidence_label": "timing-only",
        "rows": args.rows,
        "cols": args.cols,
        "warmup_iters": args.warmup,
        "measure_iters": args.iters,
        "median_ms": median_ms,
        "p10_ms": percentile(sorted_timings, 0.1),
        "p90_ms": percentile(sorted_timings, 0.9),
        "min_ms": min(sorted_timings),
        "max_ms": max(sorted_timings),
        "effective_gib_per_s": effective_gib_per_s,
        "max_abs_error": max_abs_error,
        "max_row_sum_error": max_row_sum_error,
        "bad_count": bad_count,
        "passed": passed,
        "timer_scope": "HIP events around torch.softmax(x, dim=1); input construction and CPU reference are excluded",
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
        "source_artifact": "tools/run_rowwise_softmax_pytorch_baseline.py",
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
        path = results_dir / f"pytorch-eager-{args.rows}x{args.cols}-{stamp}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
