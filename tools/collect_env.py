#!/usr/bin/env python3
"""Capture local ROCm environment metadata as JSON."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


def run_command(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "available": False,
            "command": command,
            "stdout": "",
            "stderr": f"{command[0]} not found",
            "returncode": None,
        }

    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "available": True,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "returncode": completed.returncode,
    }


def extract_gfx_targets(rocminfo_output: str) -> list[str]:
    targets: list[str] = []
    for line in rocminfo_output.splitlines():
        line = line.strip()
        if line.startswith("Name:") and "gfx" in line:
            target = line.split(":", 1)[1].strip()
            if target not in targets:
                targets.append(target)
    return targets


def main() -> int:
    commands = {
        "rocm_smi": run_command(["rocm-smi"]),
        "rocminfo": run_command(["rocminfo"]),
        "hipcc_version": run_command(["hipcc", "--version"]),
        "rocprof_version": run_command(["rocprof", "--version"]),
        "rocprofv3_version": run_command(["rocprofv3", "--version"]),
    }

    gfx_targets: list[str] = []
    rocminfo = commands["rocminfo"]
    if rocminfo["available"] and rocminfo["returncode"] == 0:
        gfx_targets = extract_gfx_targets(rocminfo["stdout"])

    env = {
        "schema_version": "0.1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "commands": commands,
        "gfx_targets": gfx_targets,
    }
    print(json.dumps(env, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

