#!/usr/bin/env python3
"""Summarize measured optimization records."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iter_records():
    for path in sorted((ROOT / "data" / "records").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def main() -> int:
    print("Measured HIP/ROCm optimization records")
    print("task_id,status,baseline_ms,optimized_ms,speedup")
    for _path, record in iter_records():
        measurement = record.get("measurement")
        if not measurement:
            continue
        print(
            f"{record['task_id']},"
            f"{record['status']},"
            f"{measurement['baseline_ms']:.6f},"
            f"{measurement['optimized_ms']:.6f},"
            f"{measurement['speedup']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

