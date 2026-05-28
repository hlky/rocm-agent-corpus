#!/usr/bin/env python3
"""Build and run library baselines for corpus tasks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def compiler_debug_flags() -> list[str]:
    if sys.platform.startswith("win"):
        return ["-gline-tables-only"]
    return ["-lineinfo"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseline",
        choices=["rocwmma-mfma-gemm:hipblaslt-hgemm"],
        help="Library baseline id to build and run",
    )
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--workspace-bytes", type=int, default=33_554_432)
    parser.add_argument("--heuristics", type=int, default=16)
    parser.add_argument("--arch", default="", help="Optional hipcc arch, e.g. gfx1030")
    parser.add_argument("--hardware-id", default="", help="Optional hardware metadata id to attach to result JSON")
    parser.add_argument("--evidence-label", default="timing-only", help="Evidence label for written result JSON")
    parser.add_argument("--write-result", action="store_true", help="Write JSON output under the task results directory")
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_json_stdout(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        raise json.JSONDecodeError("no JSON object found in benchmark stdout", stdout, 0)
    return json.loads(stdout[start : end + 1])


def main() -> int:
    args = parse_args()
    if shutil.which("hipcc") is None:
        raise SystemExit("hipcc not found. Run this on a ROCm toolkit machine or a ROCm GPU host.")

    task_id, baseline_id = args.baseline.split(":", 1)
    task_dir = ROOT / "corpus" / "tasks" / task_id
    if not (task_dir / "task.json").exists():
        raise SystemExit(f"Unknown task: {task_id}")

    out_dir = ROOT / "out" / "library_baselines" / baseline_id
    out_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "benchmark.exe" if sys.platform.startswith("win") else "benchmark"
    exe_path = out_dir / exe_name

    compile_cmd = [
        "hipcc",
        "-std=c++17",
        "-O3",
        *compiler_debug_flags(),
    ]
    if args.arch:
        compile_cmd.append(f"--offload-arch={args.arch}")
    compile_cmd.extend(
        [
            str(ROOT / "harnesses" / "hipblaslt_hgemm_benchmark.hip"),
            "-lhipblasLt",
            "-lhipblas",
            "-o",
            str(exe_path),
        ]
    )

    compiled = run(compile_cmd)
    if compiled.stdout:
        print(compiled.stdout)
    if compiled.stderr:
        print(compiled.stderr, file=sys.stderr)
    if compiled.returncode != 0:
        return compiled.returncode
    if not exe_path.exists():
        print(f"Expected benchmark executable was not created: {exe_path}", file=sys.stderr)
        return 1

    bench_cmd = [
        str(exe_path),
        str(args.m),
        str(args.n),
        str(args.k),
        str(args.warmup),
        str(args.iters),
        str(args.workspace_bytes),
        str(args.heuristics),
    ]
    bench = run(bench_cmd)
    if bench.stderr:
        print(bench.stderr, file=sys.stderr)
    print(bench.stdout)
    if bench.returncode != 0:
        return bench.returncode

    if args.write_result:
        results_dir = task_dir / "results"
        results_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result_path = results_dir / f"hipblaslt-{args.m}x{args.n}x{args.k}-{stamp}.json"
        parsed = parse_json_stdout(bench.stdout)
        parsed["evidence_label"] = args.evidence_label
        if args.arch:
            parsed["gfx_target"] = args.arch
        if args.hardware_id:
            parsed["hardware_id"] = args.hardware_id
        parsed.setdefault(
            "timer_scope",
            "HIP events reported by the library baseline harness; allocation and host reference work are excluded",
        )
        parsed["captured_at"] = datetime.now(timezone.utc).isoformat()
        parsed["source_artifact"] = "harnesses/hipblaslt_hgemm_benchmark.hip"
        parsed["build_command"] = compile_cmd
        parsed["run_command"] = bench_cmd
        parsed["baseline_id"] = baseline_id
        result_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
