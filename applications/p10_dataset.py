"""Load P10 (GPU supply chain) instances and build agent prompts."""

from __future__ import annotations

import json
import os
from typing import Any

P10_DIR = os.path.join(os.path.dirname(__file__), "P10", "custom")


def list_p10_instances() -> list[str]:
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(P10_DIR)
        if f.startswith("p10_instance_") and f.endswith(".json")
    )


def load_p10_instance(instance_id: str) -> dict[str, Any]:
    path = os.path.join(P10_DIR, f"{instance_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_p10_query(instance: dict[str, Any]) -> str:
    instance_id = instance.get("instance_id", "unknown")
    suppliers = instance.get("suppliers", [])
    components = instance.get("components", [])
    facilities = instance.get("facilities", [])
    budget = instance.get("budget")
    deadlines = instance.get("delivery_deadlines", {})
    dependencies = instance.get("dependencies", [])
    disruptions = instance.get("disruption_scenarios", [])

    query = f"""
GPU Supply Chain Planning (P10) - Instance: {instance_id}

Problem Description:
- Plan procurement and assembly for GPU supply chain components
- Respect supplier capacity, lead times, facility capacity, and budget
- Satisfy component delivery deadlines and process dependencies
- Mitigate disruption scenarios (shortages, machine breakdowns)
- Goal: Minimize total cost while meeting all deadlines within budget

Components: {', '.join(components)}
Budget: {budget}

Suppliers:
"""
    for s in suppliers:
        query += (
            f"\n  - {s.get('supplier_id')}: location={s.get('location')}, "
            f"capacity={s.get('capacity')}, lead_time={s.get('lead_time')}, "
            f"cost_multiplier={s.get('cost_multiplier')}"
        )

    query += "\n\nFacilities:\n"
    for f in facilities:
        query += (
            f"\n  - {f.get('facility_id')}: location={f.get('location')}, "
            f"capacity={f.get('capacity')}, cost_per_unit={f.get('cost_per_unit')}"
        )

    query += "\n\nDelivery Deadlines (days):\n"
    for comp, days in deadlines.items():
        query += f"\n  - {comp}: {days}"

    query += "\n\nDependencies (prerequisite -> dependent):\n"
    for dep in dependencies:
        query += f"\n  - {dep}"

    query += "\n\nDisruption Scenarios:\n"
    for dis in disruptions:
        query += f"\n  - {dis}"

    query += """

REQUIRED OUTPUT FORMAT:
1. FINAL TOTAL COST: [numeric value]
2. BUDGET UTILIZATION: [percentage]
3. PROCUREMENT PLAN:
   - Component: [name], Supplier: [id], Quantity: [n], Lead time: [days], Cost: [value]
4. FACILITY PLAN:
   - Facility: [id], Throughput: [n], Cost: [value]
5. DISRUPTION MITIGATION:
   - Scenario: [type], Action: [strategy], Residual risk: [description]

Provide a feasible supply plan within budget that meets deadlines and dependencies.
"""
    return query.strip()


def instance_path(instance_id: str) -> str:
    return os.path.join(P10_DIR, f"{instance_id}.json")
