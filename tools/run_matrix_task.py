#!/usr/bin/env python3
"""Build and run a seed matrix HIP/ROCm task."""

from __future__ import annotations

import argparse
import json
import site
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="Task id under corpus/tasks")
    parser.add_argument("variant", choices=["baseline", "optimized"], help="Variant to build and run")
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--arch", default="", help="Optional hipcc arch, e.g. gfx1100")
    parser.add_argument("--hardware-id", default="", help="Optional hardware metadata id to attach to result JSON")
    parser.add_argument("--evidence-label", default="timing-only", help="Evidence label for written result JSON")
    parser.add_argument("--write-result", action="store_true", help="Write JSON output under the task results directory")
    parser.add_argument("--profile", action="store_true", help="Run with rocprof MemoryWorkloadAnalysis_Tables after timing")
    return parser.parse_args()


def load_task(task_id: str) -> tuple[Path, dict]:
    task_dir = ROOT / "corpus" / "tasks" / task_id
    task_json = task_dir / "task.json"
    if not task_json.exists():
        raise SystemExit(f"Unknown task: {task_id}")
    return task_dir, json.loads(task_json.read_text(encoding="utf-8"))


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


def compiler_debug_flags() -> list[str]:
    if sys.platform.startswith("win"):
        return ["-gline-tables-only"]
    return ["-lineinfo"]


def bundled_rocm_include_flags() -> list[str]:
    flags: list[str] = []
    for site_dir in site.getsitepackages():
        include_dir = Path(site_dir) / "_rocm_sdk_devel" / "include"
        if include_dir.exists():
            flags.append(f"-I{include_dir}")
    return flags


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

    task_dir, task = load_task(args.task_id)
    operation = task.get("benchmarking", {}).get("operation")
    if operation not in {"copy", "transpose"}:
        raise SystemExit(f"Task {args.task_id} does not declare benchmarking.operation copy|transpose")

    artifact_key = args.variant
    source_rel = task["artifacts"][artifact_key]
    source_path = task_dir / source_rel
    if not source_path.exists():
        raise SystemExit(f"Missing source artifact: {source_path}")

    out_dir = ROOT / "out" / args.task_id / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    exe_name = "benchmark.exe" if sys.platform.startswith("win") else "benchmark"
    exe_path = out_dir / exe_name

    variant_define = "-DVARIANT_BASELINE" if args.variant == "baseline" else "-DVARIANT_OPTIMIZED"
    op_define = "-DOP_COPY" if operation == "copy" else "-DOP_TRANSPOSE"
    compile_cmd = [
        "hipcc",
        "-std=c++17",
        "-O3",
        *compiler_debug_flags(),
        *bundled_rocm_include_flags(),
        variant_define,
        op_define,
    ]
    if args.arch:
        compile_cmd.append(f"--offload-arch={args.arch}")
    compile_cmd.extend([
        str(source_path),
        str(ROOT / "harnesses" / "matrix_transform_benchmark.hip"),
        "-o",
        str(exe_path),
    ])

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
        str(args.rows),
        str(args.cols),
        str(args.warmup),
        str(args.iters),
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
        result_path = results_dir / f"{args.variant}-{args.rows}x{args.cols}-{stamp}.json"
        parsed = parse_json_stdout(bench.stdout)
        parsed["task_id"] = args.task_id
        parsed["evidence_label"] = args.evidence_label
        if args.arch:
            parsed["gfx_target"] = args.arch
        if args.hardware_id:
            parsed["hardware_id"] = args.hardware_id
        parsed.setdefault(
            "timer_scope",
            "HIP events around the benchmark function call for the selected variant; allocation and host reference work are excluded",
        )
        parsed["captured_at"] = datetime.now(timezone.utc).isoformat()
        parsed["source_artifact"] = source_rel
        parsed["build_command"] = compile_cmd
        with result_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(parsed, indent=2) + "\n")
        print(f"Wrote {result_path}")

    if args.profile:
        if shutil.which("rocprof") is None:
            raise SystemExit("rocprof not found; timing succeeded but profiling cannot run.")
        profile_dir = task_dir / "profiles"
        profile_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        profile_path = profile_dir / f"{args.variant}-{args.rows}x{args.cols}-{stamp}.txt"
        rocprof_cmd = [
            "rocprof",
            "--section",
            "MemoryWorkloadAnalysis_Tables",
            "--print-details=all",
            str(exe_path),
            str(args.rows),
            str(args.cols),
            "2",
            "5",
        ]
        profiled = run(rocprof_cmd)
        profile_path.write_text(profiled.stdout + profiled.stderr, encoding="utf-8")
        print(f"Wrote {profile_path}")
        return profiled.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

