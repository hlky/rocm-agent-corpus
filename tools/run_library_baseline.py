#!/usr/bin/env python3
"""Build and run library baselines for corpus tasks."""

from __future__ import annotations

import argparse
import json
import os
import site
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


def bundled_rocm_sdk_roots() -> list[Path]:
    roots: list[Path] = []
    for site_dir in site.getsitepackages():
        root = Path(site_dir) / "_rocm_sdk_devel"
        if root.exists():
            roots.append(root)
    return roots


def bundled_rocm_include_flags() -> list[str]:
    flags: list[str] = []
    for root in bundled_rocm_sdk_roots():
        include_dir = root / "include"
        if include_dir.exists():
            flags.append(f"-I{include_dir}")
    return flags


def bundled_rocm_library_flags() -> list[str]:
    flags: list[str] = []
    for root in bundled_rocm_sdk_roots():
        lib_dir = root / "lib"
        if lib_dir.exists():
            flags.append(f"-L{lib_dir}")
    return flags


def bundled_rocm_library_link_args(names: list[str]) -> list[str]:
    if sys.platform.startswith("win"):
        return [f"-l{name}" for name in names]

    args: list[str] = []
    roots = bundled_rocm_sdk_roots()
    for name in names:
        selected = None
        for root in roots:
            lib_dir = root / "lib"
            candidates = [
                lib_dir / f"{name}.lib",
                lib_dir / f"{name.lower()}.lib",
            ]
            if not sys.platform.startswith("win"):
                candidates.extend(
                    [
                        lib_dir / f"lib{name}.dll.a",
                        lib_dir / f"lib{name.lower()}.dll.a",
                    ]
                )
            selected = next((candidate for candidate in candidates if candidate.exists()), None)
            if selected is not None:
                break
        args.append(str(selected) if selected is not None else f"-l{name}")
    return args


def expose_bundled_rocm_runtime_bins() -> None:
    bins = [root / "bin" for root in bundled_rocm_sdk_roots() if (root / "bin").exists()]
    if not bins:
        return
    os.environ["PATH"] = os.pathsep.join(str(path) for path in bins) + os.pathsep + os.environ.get("PATH", "")


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
    expose_bundled_rocm_runtime_bins()
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
        *bundled_rocm_include_flags(),
    ]
    if args.arch:
        compile_cmd.append(f"--offload-arch={args.arch}")
    compile_cmd.extend(
        [
            str(ROOT / "harnesses" / "hipblaslt_hgemm_benchmark.hip"),
            *bundled_rocm_library_flags(),
            *bundled_rocm_library_link_args(["hipblasLt", "hipblas"]),
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
        with result_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(parsed, indent=2) + "\n")
        print(f"Wrote {result_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
