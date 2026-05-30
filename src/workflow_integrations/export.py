"""Orchestrate optional workflow artifact export."""

from __future__ import annotations

import os
from typing import Any

from workflow_integrations.config import IntegrationConfig
from workflow_integrations.python_spec import export_python_verification_spec

try:
    from workflow_integrations.argo_export import export_argo_workflow
except ImportError:
    export_argo_workflow = None  # type: ignore

try:
    from workflow_integrations.asl_export import export_asl_state_machine
except ImportError:
    export_asl_state_machine = None  # type: ignore

try:
    from workflow_integrations.bpmn_export import export_bpmn_diagram
except ImportError:
    export_bpmn_diagram = None  # type: ignore


def export_workflow_artifacts(
    workflow: dict[str, Any],
    out_dir: str,
    config: IntegrationConfig | None = None,
    *,
    argo_namespace: str = "default",
    argo_image: str = "python:3.11-slim",
) -> dict[str, str]:
    """
    Write optional integration files under out_dir.

    Returns mapping integration name → file path.
    """
    config = config or IntegrationConfig()
    os.makedirs(out_dir, exist_ok=True)
    name = workflow.get("name", "workflow").replace(" ", "_")
    written: dict[str, str] = {}

    if config.enable_python_spec:
        path = os.path.join(out_dir, f"{name}_python_verification.json")
        export_python_verification_spec(workflow, path)
        written["python"] = path

    if config.enable_argo and export_argo_workflow:
        path = os.path.join(out_dir, f"{name}_argo_workflow.yaml")
        export_argo_workflow(
            workflow,
            path,
            namespace=argo_namespace,
            image=argo_image,
        )
        written["argo"] = path

    if config.enable_asl and export_asl_state_machine:
        path = os.path.join(out_dir, f"{name}_asl.json")
        export_asl_state_machine(workflow, path)
        written["asl"] = path

    if config.enable_bpmn and export_bpmn_diagram:
        path = os.path.join(out_dir, f"{name}_bpmn20.xml")
        export_bpmn_diagram(workflow, path)
        written["bpmn"] = path

    return written
