#!/usr/bin/env python3
"""Validate the HIP/ROCm Agent Corpus scaffold.

The validator intentionally uses only the Python standard library so it can run
in fresh ROCm workspaces without dependency installation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


SOURCE_REQUIRED = {
    "id": str,
    "name": str,
    "url": str,
    "kind": str,
    "license": str,
    "use_policy": str,
    "priority": int,
    "topics": list,
}

TASK_REQUIRED = {
    "schema_version": str,
    "id": str,
    "title": str,
    "domain": str,
    "language": str,
    "tags": list,
    "source_refs": list,
    "problem": dict,
    "artifacts": dict,
    "verification": dict,
    "benchmarking": dict,
    "license": str,
    "curation": dict,
}

RECORD_REQUIRED = {
    "schema_version": str,
    "id": str,
    "task_id": str,
    "status": str,
    "optimization_tags": list,
    "baseline_artifact": str,
    "optimized_artifact": str,
    "claim": str,
    "expected_profiler_signals": list,
    "correctness_status": str,
    "curation_note": str,
}


class Validation:
    def __init__(self) -> None:
      self.errors: list[str] = []
      self.counts = {
          "source_files": 0,
          "sources": 0,
          "tasks": 0,
          "records": 0,
      }

    def error(self, message: str) -> None:
        self.errors.append(message)


def load_json(path: Path, validation: Validation) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        validation.error(f"{path}: invalid JSON: {exc}")
    except OSError as exc:
        validation.error(f"{path}: could not read file: {exc}")
    return None


def check_required(path: Path, obj: dict[str, Any], required: dict[str, type], validation: Validation) -> None:
    for key, expected_type in required.items():
        if key not in obj:
            validation.error(f"{path}: missing required field '{key}'")
            continue
        if not isinstance(obj[key], expected_type):
            validation.error(
                f"{path}: field '{key}' should be {expected_type.__name__}, "
                f"got {type(obj[key]).__name__}"
            )


def validate_sources(validation: Validation) -> set[str]:
    source_ids: set[str] = set()
    for path in sorted((ROOT / "data" / "sources").glob("*.json")):
        validation.counts["source_files"] += 1
        doc = load_json(path, validation)
        if not isinstance(doc, dict):
            validation.error(f"{path}: expected top-level object")
            continue
        if not isinstance(doc.get("sources"), list):
            validation.error(f"{path}: expected 'sources' array")
            continue
        for source in doc["sources"]:
            if not isinstance(source, dict):
                validation.error(f"{path}: source entries must be objects")
                continue
            check_required(path, source, SOURCE_REQUIRED, validation)
            source_id = source.get("id")
            if isinstance(source_id, str):
                if source_id in source_ids:
                    validation.error(f"{path}: duplicate source id '{source_id}'")
                source_ids.add(source_id)
                validation.counts["sources"] += 1
            topics = source.get("topics")
            if isinstance(topics, list) and not all(isinstance(topic, str) for topic in topics):
                validation.error(f"{path}: source '{source_id}' topics must all be strings")
    return source_ids


def validate_tasks(source_ids: set[str], validation: Validation) -> set[str]:
    task_ids: set[str] = set()
    task_root = ROOT / "corpus" / "tasks"
    for path in sorted(task_root.glob("*/task.json")):
        doc = load_json(path, validation)
        if not isinstance(doc, dict):
            validation.error(f"{path}: expected top-level object")
            continue
        check_required(path, doc, TASK_REQUIRED, validation)

        task_id = doc.get("id")
        if isinstance(task_id, str):
            if task_id in task_ids:
                validation.error(f"{path}: duplicate task id '{task_id}'")
            task_ids.add(task_id)
            validation.counts["tasks"] += 1

        for source_ref in doc.get("source_refs", []):
            if source_ref not in source_ids:
                validation.error(f"{path}: unknown source_ref '{source_ref}'")

        artifacts = doc.get("artifacts", {})
        if isinstance(artifacts, dict):
            artifact_values: list[str] = []
            for key in ("baseline", "optimized", "harness"):
                value = artifacts.get(key)
                if isinstance(value, str):
                    artifact_values.append(value)
            for key in ("profiles", "results"):
                value = artifacts.get(key)
                if isinstance(value, list):
                    artifact_values.extend(item for item in value if isinstance(item, str))
            for artifact in artifact_values:
                artifact_path = path.parent / artifact
                if not artifact_path.exists():
                    validation.error(f"{path}: artifact '{artifact}' does not exist")
    return task_ids


def validate_records(task_ids: set[str], validation: Validation) -> None:
    records_root = ROOT / "data" / "records"
    for path in sorted(records_root.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    validation.error(f"{path}:{line_no}: invalid JSONL record: {exc}")
                    continue
                if not isinstance(record, dict):
                    validation.error(f"{path}:{line_no}: record must be an object")
                    continue
                check_required(path, record, RECORD_REQUIRED, validation)
                task_id = record.get("task_id")
                if isinstance(task_id, str) and task_id not in task_ids:
                    validation.error(f"{path}:{line_no}: unknown task_id '{task_id}'")
                validation.counts["records"] += 1


def main() -> int:
    validation = Validation()
    source_ids = validate_sources(validation)
    task_ids = validate_tasks(source_ids, validation)
    validate_records(task_ids, validation)

    print("HIP/ROCm Agent Corpus validation")
    for key, value in validation.counts.items():
        print(f"  {key}: {value}")

    if validation.errors:
        print("\nErrors:")
        for error in validation.errors:
            print(f"  - {error}")
        return 1

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

