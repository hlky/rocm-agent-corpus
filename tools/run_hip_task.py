#!/usr/bin/env python3
"""Build and run one of the HIP/ROCm seed benchmark tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TASK_CONFIG = {
    "vectorized-saxpy": {
        "harness": "harnesses/vector_saxpy_benchmark.hip",
        "defines": [],
        "default_args": ["16777216", "10", "50"],
        "shape_label": lambda args: f"{args[0]}",
    },
    "block-reduction-sum": {
        "harness": "harnesses/reduction_sum_benchmark.hip",
        "defines": [],
        "default_args": ["16777216", "5", "30"],
        "shape_label": lambda args: f"{args[0]}",
    },
    "rowwise-softmax": {
        "harness": "harnesses/rowwise_softmax_benchmark.hip",
        "defines": [],
        "default_args": ["4096", "1024", "5", "30"],
        "shape_label": lambda args: f"{args[0]}x{args[1]}",
    },
    "rowwise-layernorm-rmsnorm": {
        "harness": "harnesses/layernorm_rmsnorm_benchmark.hip",
        "defines": [],
        "default_args": ["4096", "1024", "rmsnorm", "5", "30"],
        "shape_label": lambda args: f"{args[0]}x{args[1]}-{args[2]}",
    },
    "block-prefix-scan": {
        "harness": "harnesses/prefix_scan_benchmark.hip",
        "defines": [],
        "default_args": ["65536", "256", "5", "30"],
        "shape_label": lambda args: f"{args[0]}x{args[1]}",
    },
    "histogram-privatized-atomics": {
        "harness": "harnesses/histogram_benchmark.hip",
        "defines": [],
        "default_args": ["16777216", "5", "30", "skewed"],
        "shape_label": lambda args: f"{args[0]}-{args[3]}",
    },
    "small-fixed-gemm": {
        "harness": "harnesses/small_gemm_benchmark.hip",
        "defines": [],
        "default_args": ["16", "16", "16", "4096", "10", "50"],
        "shape_label": lambda args: f"{args[3]}x{args[0]}x{args[1]}x{args[2]}",
    },
    "online-attention-forward": {
        "harness": "harnesses/attention_forward_benchmark.hip",
        "defines": [],
        "default_args": ["1", "4", "128", "128", "64", "1", "5", "30"],
        "shape_label": lambda args: (
            f"b{args[0]}h{args[1]}q{args[2]}k{args[3]}d{args[4]}-causal{args[5]}"
        ),
    },
    "block-topk-sampling": {
        "harness": "harnesses/topk_sampling_benchmark.hip",
        "defines": [],
        "default_args": ["1024", "32768", "4", "0.8", "5", "30"],
        "shape_label": lambda args: f"{args[0]}x{args[1]}-k{args[2]}-t{args[3]}",
    },
    "select-filter-compact": {
        "harness": "harnesses/select_filter_compact_benchmark.hip",
        "defines": [],
        "default_args": ["1048576", "medium", "5", "30"],
        "shape_label": lambda args: f"{args[0]}-{args[1]}",
    },
    "fused-int4-dequant-gemv": {
        "harness": "harnesses/int4_dequant_gemv_benchmark.hip",
        "defines": [],
        "default_args": ["4096", "4096", "128", "10", "50"],
        "shape_label": lambda args: f"{args[0]}x{args[1]}-g{args[2]}",
    },
    "rocwmma-mfma-gemm": {
        "harness": "harnesses/rocwmma_gemm_benchmark.hip",
        "defines": [],
        "default_args": ["256", "256", "256", "10", "50"],
        "shape_label": lambda args: f"{args[0]}x{args[1]}x{args[2]}",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", choices=sorted(TASK_CONFIG))
    parser.add_argument("variant", choices=["baseline", "optimized"])
    parser.add_argument("benchmark_args", nargs="*", help="Arguments forwarded to the benchmark binary")
    parser.add_argument("--arch", default="", help="Optional hipcc arch, e.g. gfx1030")
    parser.add_argument("--write-result", action="store_true")
    return parser.parse_args()


def load_task(task_id: str) -> tuple[Path, dict]:
    task_dir = ROOT / "corpus" / "tasks" / task_id
    task_json = task_dir / "task.json"
    if not task_json.exists():
        raise SystemExit(f"Unknown task: {task_id}")
    return task_dir, json.loads(task_json.read_text(encoding="utf-8"))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main() -> int:
    args = parse_args()
    if shutil.which("hipcc") is None:
        raise SystemExit("hipcc not found. Run this on a ROCm toolkit machine or ROCm GPU host.")

    config = TASK_CONFIG[args.task_id]
    bench_args = args.benchmark_args or config["default_args"]
    task_dir, task = load_task(args.task_id)
    source_rel = task["artifacts"][args.variant]
    source_path = task_dir / source_rel
    if not source_path.exists():
        raise SystemExit(f"Missing source artifact: {source_path}")

    out_dir = ROOT / "out" / args.task_id / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "benchmark.exe" if sys.platform.startswith("win") else "benchmark"
    exe_path = out_dir / exe_name

    variant_define = "-DVARIANT_BASELINE" if args.variant == "baseline" else "-DVARIANT_OPTIMIZED"
    compile_cmd = ["hipcc", "-std=c++17", "-O3", "-lineinfo", variant_define]
    compile_cmd.extend(config["defines"])
    if args.arch:
        compile_cmd.append(f"--offload-arch={args.arch}")
    compile_cmd.extend([str(source_path), str(ROOT / config["harness"]), "-o", str(exe_path)])

    compiled = run(compile_cmd)
    if compiled.stdout:
        print(compiled.stdout)
    if compiled.stderr:
        print(compiled.stderr, file=sys.stderr)
    if compiled.returncode != 0:
        return compiled.returncode

    bench = run([str(exe_path), *bench_args])
    if bench.stderr:
        print(bench.stderr, file=sys.stderr)
    print(bench.stdout)
    if bench.returncode != 0:
        return bench.returncode

    if args.write_result:
        results_dir = task_dir / "results"
        results_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        shape = config["shape_label"](bench_args)
        result_path = results_dir / f"{args.variant}-{shape}-{stamp}.json"
        parsed = json.loads(bench.stdout)
        parsed["task_id"] = args.task_id
        parsed["captured_at"] = datetime.now(timezone.utc).isoformat()
        parsed["source_artifact"] = source_rel
        parsed["build_command"] = compile_cmd
        result_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
