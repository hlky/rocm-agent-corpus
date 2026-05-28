#!/usr/bin/env python3
"""Inspect local ROCm GPU architecture information."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> dict:
    if shutil.which(command[0]) is None:
        return {
            "available": False,
            "command": command,
            "stdout": "",
            "stderr": f"{command[0]} not found",
            "returncode": None,
        }
    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {
        "available": True,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def parse_gfx_targets(rocminfo_text: str) -> list[str]:
    targets: list[str] = []
    for line in rocminfo_text.splitlines():
        line = line.strip()
        if line.startswith("Name:") and "gfx" in line:
            target = line.split(":", 1)[1].strip()
            if target not in targets:
                targets.append(target)
    return targets


def main() -> int:
    smi = run(["rocm-smi"])
    rocminfo = run(["rocminfo"])
    hipcc_targets = run(["hipcc", "--list-gpu-targets"])
    hipcc_help = run(["hipcc", "--help"])

    gfx_targets = parse_gfx_targets(rocminfo["stdout"]) if rocminfo["returncode"] == 0 else []

    result = {
        "schema_version": "0.1.0",
        "gfx_targets": gfx_targets,
        "commands": {
            "rocm_smi": smi,
            "rocminfo": rocminfo,
            "hipcc_list_gpu_targets": hipcc_targets,
            "hipcc_help": hipcc_help,
        },
        "architecture_index": str(ROOT / "data" / "index" / "gpu_architectures.json"),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

