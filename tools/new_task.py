#!/usr/bin/env python3
"""Create a new HIP/ROCm optimization task skeleton."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


BASELINE_TEMPLATE = """#include <hip/hip_runtime.h>

__global__ void baseline_kernel() {
  // TODO: add baseline implementation.
}

extern "C" void launch_baseline(hipStream_t stream) {
  baseline_kernel<<<1, 1, 0, stream>>>();
}
"""


OPTIMIZED_TEMPLATE = """#include <hip/hip_runtime.h>

__global__ void optimized_kernel() {
  // TODO: add optimized implementation.
}

extern "C" void launch_optimized(hipStream_t stream) {
  optimized_kernel<<<1, 1, 0, stream>>>();
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", help="Stable kebab-case task id")
    parser.add_argument("title", help="Human-readable title")
    parser.add_argument("--domain", default="memory", help="Domain tag, e.g. memory, gemm, reduction")
    parser.add_argument("--language", default="hip-cpp", help="Implementation language")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_dir = ROOT / "corpus" / "tasks" / args.task_id
    source_dir = task_dir / "source"

    if task_dir.exists():
        raise SystemExit(f"Task already exists: {task_dir}")

    source_dir.mkdir(parents=True)

    tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]
    task = {
        "schema_version": "0.1.0",
        "id": args.task_id,
        "title": args.title,
        "domain": args.domain,
        "language": args.language,
        "tags": tags,
        "source_refs": [],
        "problem": {
            "summary": "TODO: describe the operation and optimization opportunity.",
            "inputs": [],
            "outputs": [],
            "constraints": []
        },
        "artifacts": {
            "baseline": "source/baseline.hip",
            "optimized": "source/optimized.hip"
        },
        "verification": {
            "status": "todo",
            "method": "TODO: describe correctness checks.",
            "tolerances": {}
        },
        "benchmarking": {
            "status": "todo",
            "required_metrics": [],
            "recommended_shapes": []
        },
        "license": "TODO",
        "curation": {
            "status": "draft",
            "notes": "TODO"
        }
    }

    (task_dir / "task.json").write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
    (source_dir / "baseline.hip").write_text(BASELINE_TEMPLATE, encoding="utf-8")
    (source_dir / "optimized.hip").write_text(OPTIMIZED_TEMPLATE, encoding="utf-8")

    print(f"Created {task_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

