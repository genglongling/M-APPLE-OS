#!/usr/bin/env python3
"""Export optional workflow artifacts (Argo, ASL, BPMN) from an ALAS workflow spec."""

from __future__ import annotations

import argparse
import json
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from workflow_integrations import (  # noqa: E402
    IntegrationConfig,
    alas_maple_jssp_workflow,
    export_workflow_artifacts,
    task_spec_to_canonical,
)


def load_workflow(path: str | None, variant: str) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "steps" in data:
            return data
        return task_spec_to_canonical(data)
    return alas_maple_jssp_workflow(variant)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export optional Argo / ASL / BPMN workflow artifacts"
    )
    parser.add_argument(
        "--workflow",
        help="JSON task_spec or canonical workflow (default: built-in MAPLE JSSP)",
    )
    parser.add_argument(
        "--variant",
        default="full",
        choices=["full", "no_validation", "no_optimization", "no_repair"],
        help="MAPLE preset when --workflow is omitted",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(project_root, "artifacts", "workflow_integrations"),
    )
    parser.add_argument("--enable-argo", action="store_true")
    parser.add_argument("--enable-asl", action="store_true")
    parser.add_argument("--enable-bpmn", action="store_true")
    parser.add_argument(
        "--no-python-spec",
        action="store_true",
        help="Skip python_verification.json sidecar",
    )
    parser.add_argument("--argo-namespace", default="default")
    parser.add_argument("--argo-image", default="python:3.11-slim")
    args = parser.parse_args()

    workflow = load_workflow(args.workflow, args.variant)
    config = IntegrationConfig(
        enable_argo=args.enable_argo or os.getenv("ALAS_ENABLE_ARGO", "").lower() in ("1", "true", "yes"),
        enable_asl=args.enable_asl or os.getenv("ALAS_ENABLE_ASL", "").lower() in ("1", "true", "yes"),
        enable_bpmn=args.enable_bpmn or os.getenv("ALAS_ENABLE_BPMN", "").lower() in ("1", "true", "yes"),
        enable_python_spec=not args.no_python_spec,
    )

    if not config.enable_python_spec and not config.any_enabled:
        print(
            "No integrations selected. Use --enable-argo, --enable-asl, --enable-bpmn, "
            "or omit --no-python-spec (Python spec is on by default)."
        )
        sys.exit(1)

    written = export_workflow_artifacts(
        workflow,
        args.out_dir,
        config,
        argo_namespace=args.argo_namespace,
        argo_image=args.argo_image,
    )
    print(f"Workflow: {workflow.get('name')}")
    print(f"Output directory: {args.out_dir}")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
