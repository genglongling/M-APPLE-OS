"""Export Argo Workflows CRD YAML for Kubernetes execution (optional)."""

from __future__ import annotations

from typing import Any


def export_argo_workflow(
    workflow: dict[str, Any],
    out_path: str,
    *,
    namespace: str = "default",
    image: str = "python:3.11-slim",
    command_module: str = "applications.run_workflow_step",
) -> str:
    """
    Emit a Workflow manifest. Each step runs a Python entrypoint stub that
    delegates to ALAS/MAPLE when wired in your cluster (ConfigMap / image build).

    Requires: Argo Workflows controller on Kubernetes [optional integration].
    """
    wf_name = workflow.get("name", "alas-workflow").replace("_", "-")[:63]
    templates: list[str] = []
    dag_tasks: list[str] = []

    for step in workflow.get("steps", []):
        sid = step["id"]
        templates.append(
            _container_template(sid, image, command_module, step.get("agent", sid))
        )
        deps = step.get("depends_on") or []
        dep_clause = ""
        if deps:
            dep_items = "\n".join(
                f'            - name: {d}' for d in deps
            )
            dep_clause = f"\n        dependencies:\n{dep_items}"
        dag_tasks.append(
            f"""      - name: {sid}
        template: {sid}{dep_clause}"""
        )

    body = f"""apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  generateName: {wf_name}-
  namespace: {namespace}
  labels:
    app.kubernetes.io/name: alas
    alas.workflow/name: {wf_name}
spec:
  entrypoint: main-dag
  templates:
  - name: main-dag
    dag:
      tasks:
{chr(10).join(dag_tasks)}
{"".join(templates)}
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(body)
    return out_path


def _container_template(
    step_id: str, image: str, command_module: str, agent: str
) -> str:
    return f"""
  - name: {step_id}
    container:
      image: {image}
      command: [python, -m, {command_module}]
      args: ["--step-id", "{step_id}", "--agent", "{agent}"]
      env:
        - name: ALAS_WORKFLOW_STEP
          value: "{step_id}"
"""
