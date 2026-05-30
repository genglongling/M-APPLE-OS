"""Export Amazon States Language (ASL) state machines (optional)."""

from __future__ import annotations

import json
from typing import Any


def export_asl_state_machine(workflow: dict[str, Any], out_path: str) -> str:
    """
    Emit ASL JSON. Task states invoke the same Python step stub as Argo
    (Lambda, ECS, or Step Functions Activity in your deployment).

    Reference: Amazon States Language for JSON state machines [optional].
    """
    steps = workflow.get("steps", [])
    if not steps:
        raise ValueError("Workflow has no steps")

    states: dict[str, Any] = {}
    step_ids = [s["id"] for s in steps]
    terminal = step_ids[-1]

    for i, step in enumerate(steps):
        sid = step["id"]
        resource = (
            "arn:aws:states:::lambda:invoke"
            if not step.get("optional")
            else "arn:aws:states:::lambda:invoke.waitForTaskToken"
        )
        next_id = step_ids[i + 1] if i + 1 < len(steps) else None
        state: dict[str, Any] = {
            "Type": "Task",
            "Resource": resource,
            "Parameters": {
                "FunctionName": "${AlasWorkflowStepFunction}",
                "Payload": {
                    "step_id": sid,
                    "agent.$": "$.agent",
                    "workflow.$": "$.workflow",
                },
            },
            "ResultPath": f"$.results.{sid}",
        }
        if next_id:
            state["Next"] = next_id
        else:
            state["End"] = True
        states[sid] = state

    machine = {
        "Comment": workflow.get("description") or workflow.get("name"),
        "StartAt": step_ids[0],
        "States": states,
        "Metadata": {
            "alas_workflow": workflow.get("name"),
            "verification": "python",
            "terminal_step": terminal,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(machine, f, indent=2)
    return out_path
