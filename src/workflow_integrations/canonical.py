"""Canonical workflow graph (language-neutral) for optional exporters."""

from __future__ import annotations

import re
from typing import Any


def slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return s or "step"


def task_spec_to_canonical(task_spec: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an ALAS/MAPLE task_spec (nodes with agent + dependencies) to a
    JSON-serializable workflow graph.
    """
    steps: list[dict[str, Any]] = []
    for i, node in enumerate(task_spec.get("nodes", [])):
        agent = node.get("agent")
        if isinstance(agent, str):
            agent_name = agent
        elif agent is not None and hasattr(agent, "name"):
            agent_name = str(agent.name)
        else:
            agent_name = node.get("id") or node.get("name") or f"step_{i}"

        raw_deps = node.get("dependencies", [])
        depends_on = [slug(str(d)) for d in raw_deps]

        steps.append(
            {
                "id": node.get("id") or slug(agent_name),
                "agent": agent_name,
                "depends_on": depends_on,
                "kind": node.get("kind", "agent"),
                "optional": bool(node.get("optional", False)),
            }
        )

    return {
        "name": task_spec.get("name", "alas-workflow"),
        "description": task_spec.get("description", ""),
        "verification": "python",
        "steps": steps,
    }


def alas_maple_jssp_workflow(variant: str = "full") -> dict[str, Any]:
    """
    Default MAPLE–ALAS JSSP pipeline (matches maple_workflow_diagram.txt).

    variant: full | no_validation | no_optimization | no_repair
    """
    validate = variant != "no_validation"
    optimize = variant not in ("no_optimization", "no_repair")

    steps: list[dict[str, Any]] = [
        {
            "id": "generate_schedule",
            "agent": "MAPLEJSSPQueryAgent",
            "depends_on": [],
            "kind": "llm",
        },
    ]
    if validate:
        steps.append(
            {
                "id": "validate_schedule",
                "agent": "ValidationTools",
                "depends_on": ["generate_schedule"],
                "kind": "python",
            }
        )
    if optimize:
        prev = "validate_schedule" if validate else "generate_schedule"
        steps.append(
            {
                "id": "optimize_schedule",
                "agent": "OptimizationTools",
                "depends_on": [prev],
                "kind": "python",
                "optional": True,
            }
        )
        steps.append(
            {
                "id": "persist_schedule",
                "agent": "FileStorageTools",
                "depends_on": ["optimize_schedule"],
                "kind": "python",
            }
        )

    return {
        "name": f"maple-jssp-{variant}",
        "description": f"ALAS MAPLE JSSP workflow ({variant})",
        "verification": "python",
        "variant": variant,
        "steps": steps,
    }
