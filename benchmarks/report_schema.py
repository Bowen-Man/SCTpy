from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SUPPORTED_CORRECTNESS_REPORT_VERSIONS = frozenset({1})
REQUIRED_STAGES = frozenset(
    {"input", "selection", "raw_fitting", "regularization", "residual", "hvg"}
)


def validate_correctness_report(report: Mapping[str, Any]) -> None:
    """Validate the stable envelope without adding a benchmark runtime dependency."""
    version = report.get("schema_version")
    if type(version) is not int or version not in SUPPORTED_CORRECTNESS_REPORT_VERSIONS:
        raise ValueError(f"unsupported correctness report schema {version!r}")
    if type(report.get("native_gate_passed")) is not bool:
        raise ValueError("correctness report must contain boolean native_gate_passed")
    stages = report.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("correctness report must contain a stages object")
    missing = REQUIRED_STAGES.difference(stages)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"correctness report is missing stages: {names}")


def load_correctness_report(path: str | Path) -> dict[str, Any]:
    """Load a supported report while tolerating additive fields within its version."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("correctness report root must be an object")
    validate_correctness_report(value)
    return value
