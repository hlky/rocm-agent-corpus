#!/usr/bin/env python3
"""Benchmark a Triton normalization-backward baseline for LayerNorm/RMSNorm."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODE_RMSNORM = 0
MODE_LAYERNORM = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["layernorm", "rmsnorm"], required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--dtype", choices=["fp32"], default="fp32")
    parser.add_argument("--num-warps-row", type=int, default=8)
    parser.add_argument("--num-warps-param", type=int, default=8)
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


def next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def load_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn.functional as F
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise SystemExit("PyTorch and Triton are required for this baseline runner") from exc
    return torch, F, triton


torch, F, triton = load_modules()
import triton.language as tl


@triton.jit
def _norm_backward_dx_kernel(x_ptr,
                             dy_ptr,
                             gamma_ptr,
                             mean_ptr,
                             inv_ptr,
                             dx_ptr,
                             cols: tl.constexpr,
                             mode: tl.constexpr,
                             block_cols: tl.constexpr) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, block_cols)
    mask = offsets < cols
    base = row * cols + offsets
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)
    inv = tl.load(inv_ptr + row)
    dy_gamma = dy * gamma

    if mode == 1:
        mean = tl.load(mean_ptr + row)
        normalized = (x - mean) * inv
        sum_dy_gamma = tl.sum(tl.where(mask, dy_gamma, 0.0), axis=0)
        sum_dy_gamma_norm = tl.sum(tl.where(mask, dy_gamma * normalized, 0.0), axis=0)
        inv_cols = 1.0 / cols
        dx = (dy_gamma - sum_dy_gamma * inv_cols - normalized * sum_dy_gamma_norm * inv_cols) * inv
    else:
        dot = tl.sum(tl.where(mask, dy_gamma * x, 0.0), axis=0)
        scale = inv * inv * inv * dot / cols
        dx = dy_gamma * inv - x * scale

    tl.store(dx_ptr + base, dx, mask=mask)


@triton.jit
def _norm_backward_param_kernel(x_ptr,
                                dy_ptr,
                                mean_ptr,
                                inv_ptr,
                                dgamma_ptr,
                                dbeta_ptr,
                                rows: tl.constexpr,
                                cols: tl.constexpr,
                                mode: tl.constexpr,
                                block_rows: tl.constexpr) -> None:
    col = tl.program_id(0)
    offsets = tl.arange(0, block_rows)
    mask = offsets < rows
    base = offsets * cols + col
    x = tl.load(x_ptr + base, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + base, mask=mask, other=0.0)
    inv = tl.load(inv_ptr + offsets, mask=mask, other=0.0)

    if mode == 1:
        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0)
        normalized = (x - mean) * inv
        dbeta = tl.sum(tl.where(mask, dy, 0.0), axis=0)
    else:
        normalized = x * inv
        dbeta = 0.0

    dgamma = tl.sum(tl.where(mask, dy * normalized, 0.0), axis=0)
    tl.store(dgamma_ptr + col, dgamma)
    tl.store(dbeta_ptr + col, dbeta)


def error_stats(got: Any, expected: Any, abs_tol: float, rel_tol: float) -> dict[str, Any]:
    abs_error = torch.abs(got - expected)
    rel_error = abs_error / torch.clamp(torch.abs(expected), min=1.0e-12)
    bad = (abs_error > abs_tol) & (rel_error > rel_tol)
    return {
        "max_abs_error": float(torch.max(abs_error).item()),
        "max_rel_error": float(torch.max(rel_error).item()),
        "bad_count": int(torch.sum(bad).item()),
    }


def make_reference(x: Any, dy: Any, gamma: Any, beta: Any | None, mode: str, epsilon: float) -> tuple[Any, Any, Any]:
    x_ref = x.detach().clone().requires_grad_(True)
    gamma_ref = gamma.detach().clone().requires_grad_(True)
    beta_ref = None if beta is None else beta.detach().clone().requires_grad_(True)
    if mode == "layernorm":
        y = F.layer_norm(x_ref, (x.shape[1],), gamma_ref, beta_ref, epsilon)
    else:
        inv_rms = torch.rsqrt(torch.mean(x_ref * x_ref, dim=-1, keepdim=True) + epsilon)
        y = x_ref * inv_rms * gamma_ref
    y.backward(dy)
    dbeta = torch.zeros_like(gamma_ref) if beta_ref is None else beta_ref.grad.detach()
    return x_ref.grad.detach(), gamma_ref.grad.detach(), dbeta


def make_result(args: argparse.Namespace) -> dict[str, Any]:
    torch_rocm_version = getattr(torch.version, "hip", None)
    if not torch.cuda.is_available() or not torch_rocm_version:
        raise SystemExit("A PyTorch ROCm/HIP backend exposed through torch.cuda is required for Triton baseline timing")
    if args.dtype != "fp32":
        raise SystemExit("Only fp32 is currently implemented")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    x = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32)
    dy = torch.randn(args.rows, args.cols, device=device, dtype=torch.float32)
    gamma = torch.randn(args.cols, device=device, dtype=torch.float32)
    beta = torch.randn(args.cols, device=device, dtype=torch.float32) if args.mode == "layernorm" else None
    if args.mode == "layernorm":
        saved_mean = torch.mean(x, dim=1)
        variance = torch.mean((x - saved_mean[:, None]) * (x - saved_mean[:, None]), dim=1)
        saved_inv = torch.rsqrt(variance + args.epsilon)
        mode_id = MODE_LAYERNORM
    else:
        saved_mean = torch.zeros(args.rows, device=device, dtype=torch.float32)
        saved_inv = torch.rsqrt(torch.mean(x * x, dim=1) + args.epsilon)
        mode_id = MODE_RMSNORM

    ref_dx, ref_dgamma, ref_dbeta = make_reference(x, dy, gamma, beta, args.mode, args.epsilon)
    dx = torch.empty_like(x)
    dgamma = torch.empty_like(gamma)
    dbeta = torch.empty_like(gamma)
    block_cols = next_power_of_2(args.cols)
    block_rows = next_power_of_2(args.rows)

    timings: list[float] = []
    total_iters = args.warmup + args.iters
    for iteration in range(total_iters):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        _norm_backward_dx_kernel[(args.rows,)](
            x, dy, gamma, saved_mean, saved_inv, dx, args.cols, mode_id, block_cols,
            num_warps=args.num_warps_row,
        )
        _norm_backward_param_kernel[(args.cols,)](
            x, dy, saved_mean, saved_inv, dgamma, dbeta, args.rows, args.cols, mode_id, block_rows,
            num_warps=args.num_warps_param,
        )
        stop.record()
        stop.synchronize()
        elapsed = float(start.elapsed_time(stop))
        if iteration >= args.warmup:
            timings.append(elapsed)

    torch.cuda.synchronize()
    sorted_timings = sorted(timings)
    dx_errors = error_stats(dx, ref_dx, 0.005, 0.005)
    dgamma_errors = error_stats(dgamma, ref_dgamma, 0.005, 0.005)
    dbeta_errors = error_stats(dbeta, ref_dbeta, 0.005, 0.005)
    bad_count = dx_errors["bad_count"] + dgamma_errors["bad_count"] + dbeta_errors["bad_count"]
    max_abs_error = max(dx_errors["max_abs_error"], dgamma_errors["max_abs_error"], dbeta_errors["max_abs_error"])
    max_rel_error = max(dx_errors["max_rel_error"], dgamma_errors["max_rel_error"], dbeta_errors["max_rel_error"])

    props = torch.cuda.get_device_properties(device)
    runtime_version = torch_rocm_version or ""
    result: dict[str, Any] = {
        "schema_version": "0.1.0",
        "task_id": "normalization-backward",
        "variant": "triton-baseline-backward",
        "baseline_id": "triton-row-column-normalization-backward",
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
        "timer_scope": "HIP events around two Triton backward kernels; input allocation and saved-stat computation are excluded",
        "reduction_policy": "one Triton program per row for dx plus one Triton program per column for dgamma/dbeta",
        "triton": {
            "version": triton.__version__,
            "block_cols": block_cols,
            "block_rows": block_rows,
            "num_warps_row": args.num_warps_row,
            "num_warps_param": args.num_warps_param,
        },
        "framework": {
            "torch_version": torch.__version__,
            "torch_backend": "rocm",
            "torch_rocm_version": torch_rocm_version,
            "rocm_runtime_version": runtime_version,
        },
        "device_name": props.name,
        "gfx_target": getattr(props, "gcnArchName", ""),
        "rocm_runtime_version": runtime_version,
        "abs_tolerance": 0.005,
        "rel_tolerance": 0.005,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "bad_count": bad_count,
        "component_errors": {
            "dx": dx_errors,
            "dgamma": dgamma_errors,
            "dbeta": dbeta_errors,
        },
        "passed": bad_count == 0,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": "tools/run_normalization_triton_baseline.py",
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
        path = results_dir / f"triton-{args.rows}x{args.cols}-{args.mode}-{stamp}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
