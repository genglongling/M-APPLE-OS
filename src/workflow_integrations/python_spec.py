"""Python verification workflow spec (default path)."""

from __future__ import annotations

import json
from typing import Any


def export_python_verification_spec(
    workflow: dict[str, Any], out_path: str
) -> str:
    """
    Write a JSON sidecar describing Python-based verification for this workflow.

    Execution still uses src/utils/validation_tools.py and MAPLE self_validate();
    this file documents step → validator mapping for tooling and CI.
    """
    spec = {
        "format": "alas-python-verification-v1",
        "workflow_name": workflow.get("name"),
        "default_verifier": "ValidationTools.validate_schedule",
        "maple_self_validate": "ExecutionManager.self_validate",
        "comprehensive_script": "validate_all_initial_schedules_comprehensive.py",
        "steps": [],
    }
    for step in workflow.get("steps", []):
        kind = step.get("kind", "agent")
        verifier = (
            "ValidationTools.comprehensive_validation"
            if kind == "python" or "Validation" in step.get("agent", "")
            else "agent_output_review"
        )
        spec["steps"].append(
            {
                "id": step["id"],
                "agent": step.get("agent"),
                "verifier": verifier,
                "depends_on": step.get("depends_on", []),
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    return out_path
