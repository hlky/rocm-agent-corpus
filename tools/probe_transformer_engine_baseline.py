#!/usr/bin/env python3
"""Probe Transformer Engine availability for normalization-backward baselines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-record", default="")
    parser.add_argument("--install-attempt-status", default="not-attempted")
    parser.add_argument("--install-attempt-note", default="")
    parser.add_argument("--write-result", action="store_true")
    return parser.parse_args()


def run_python_probe(code: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "command": [sys.executable, "-c", code],
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def make_result(args: argparse.Namespace) -> dict[str, Any]:
    probes = {
        "transformer_engine": module_available("transformer_engine"),
        "transformer_engine_torch": module_available("transformer_engine_torch"),
        "transformer_engine.pytorch": module_available("transformer_engine.pytorch"),
    }
    import_probe = run_python_probe(
        "import transformer_engine, transformer_engine.pytorch as te; "
        "print(getattr(transformer_engine, '__version__', 'unknown')); "
        "print(hasattr(te, 'LayerNorm'), hasattr(te, 'RMSNorm'))"
    )
    available = all(probes.values()) and import_probe["returncode"] == 0
    result: dict[str, Any] = {
        "schema_version": "0.1.0",
        "task_id": "normalization-backward",
        "variant": "transformer-engine-availability-probe",
        "operation": "normalization-backward",
        "evidence_label": "provider-unavailable" if not available else "provider-available-unmeasured",
        "provider": "Transformer Engine",
        "provider_modules": probes,
        "import_probe": import_probe,
        "install_attempt": {
            "status": args.install_attempt_status,
            "note": args.install_attempt_note,
        },
        "baseline_status": "blocked" if not available else "available-unmeasured",
        "baseline_scope": "LayerNorm/RMSNorm backward via transformer_engine.pytorch LayerNorm/RMSNorm modules or lower-level transformer_engine_torch bindings",
        "timer_scope": "No timing recorded by this probe",
        "os": platform.platform(),
        "python": sys.version,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": "tools/probe_transformer_engine_baseline.py",
        "run_command": [sys.executable, str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]],
        "passed": available,
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
        path = results_dir / f"transformer-engine-probe-{stamp}.json"
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
