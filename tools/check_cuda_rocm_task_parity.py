#!/usr/bin/env python3
"""Check task-directory parity against the CUDA source corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("H:/cuda-agent-corpus")
DEFAULT_MAP = ROOT / "data" / "index" / "cuda_rocm_task_parity_map.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="CUDA corpus checkout")
    parser.add_argument("--target", type=Path, default=ROOT, help="ROCm corpus checkout")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="CUDA-to-ROCm parity map JSON")
    return parser.parse_args()


def task_ids(repo: Path) -> set[str]:
    task_root = repo / "corpus" / "tasks"
    if not task_root.is_dir():
        raise SystemExit(f"Missing task directory: {task_root}")
    return {path.name for path in task_root.iterdir() if path.is_dir()}


def load_renames(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    renames = data.get("renamed_equivalents", {})
    if not isinstance(renames, dict):
        raise SystemExit(f"Invalid renamed_equivalents in {path}")
    return {str(key): str(value) for key, value in renames.items()}


def main() -> int:
    args = parse_args()
    cuda_tasks = task_ids(args.source)
    rocm_tasks = task_ids(args.target)
    renames = load_renames(args.map)

    duplicate_targets = sorted({value for value in renames.values() if list(renames.values()).count(value) > 1})
    missing_source_keys = sorted(set(renames) - cuda_tasks)
    missing_target_values = sorted(set(renames.values()) - rocm_tasks)

    missing_in_rocm = []
    for cuda_id in sorted(cuda_tasks):
        expected_rocm_id = renames.get(cuda_id, cuda_id)
        if expected_rocm_id not in rocm_tasks:
            missing_in_rocm.append((cuda_id, expected_rocm_id))

    reverse = {value: key for key, value in renames.items()}
    extra_in_rocm = []
    for rocm_id in sorted(rocm_tasks):
        source_id = reverse.get(rocm_id, rocm_id)
        if source_id not in cuda_tasks:
            extra_in_rocm.append((rocm_id, source_id))

    print("CUDA to ROCm task parity")
    print(f"  cuda_tasks: {len(cuda_tasks)}")
    print(f"  rocm_tasks: {len(rocm_tasks)}")
    print(f"  renamed_equivalents: {len(renames)}")

    failures = []
    if duplicate_targets:
        failures.append(("duplicate ROCm rename targets", duplicate_targets))
    if missing_source_keys:
        failures.append(("rename keys missing from CUDA tasks", missing_source_keys))
    if missing_target_values:
        failures.append(("rename values missing from ROCm tasks", missing_target_values))
    if missing_in_rocm:
        failures.append(("CUDA tasks without ROCm equivalent", missing_in_rocm))
    if extra_in_rocm:
        failures.append(("ROCm tasks without CUDA source equivalent", extra_in_rocm))

    if failures:
        for title, values in failures:
            print(f"\n{title}:")
            for value in values:
                print(f"  {value}")
        return 1

    print("  status: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
