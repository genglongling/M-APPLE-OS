#!/usr/bin/env python3
"""
Evaluate P4 (URS with disruptions) and P10 (supply chain) instances.

Scans experiment output directories for solution artifacts and evaluates
feasibility / cost-gap metrics. Writes p4_p10_evaluation.json for results.tex.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from glob import glob
from typing import Any

P4_DIR = "applications/P4/disruptions"
P10_DIR = "applications/P10/custom"
OUTPUT_JSON = "p4_p10_evaluation.json"

RESULT_ROOTS = [
    "results",
    "results_mas(gpt-4o)",
    "results_mas(gpt-4o)_converted",
    "results_mas(claude-4)",
    "results_mas(claude-4)_converted",
    "results_single(gpt-4o)",
    "results_single(claude-4)",
    "results_optimized(gpt-4o)",
    "results_optimized(claude-4)",
    "results_optimized(deepseek-v3)",
    "results_optimized(gemini-2.5)",
]

INSTANCE_RE = re.compile(r"(p4_instance_\d{3}|p10_instance_\d{3})")
RESULT_FILE_RE = re.compile(
    r"^(p4|p10)_results_(p4_instance_\d{3}|p10_instance_\d{3})_(.+)\.json$"
)


def list_instances() -> dict[str, list[str]]:
    p4 = sorted(
        f.replace(".json", "")
        for f in os.listdir(P4_DIR)
        if f.startswith("p4_instance_") and f.endswith(".json")
    )
    p10 = sorted(
        f.replace(".json", "")
        for f in os.listdir(P10_DIR)
        if f.startswith("p10_instance_") and f.endswith(".json")
    )
    return {"P4": p4, "P10": p10}


def load_instance(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def p4_distance(instance: dict, origin: str, dest: str) -> float:
    key = f"{origin}-{dest}"
    distances = instance.get("city_map", {}).get("distances", {})
    return float(distances.get(key, 1e6))


def p4_reference_cost(instance: dict) -> float:
    total = 0.0
    for req in instance.get("ride_requests", []):
        total += p4_distance(instance, req["pickup"], req["dropoff"])
    return max(total, 1.0)


def p4_greedy_solution(instance: dict) -> tuple[bool, float]:
    """Greedy assign each request to the vehicle with lowest incremental travel."""
    vehicles = [
        {"id": v["vehicle_id"], "location": v["location"], "cost": 0.0}
        for v in instance.get("vehicles", [])
    ]
    if not vehicles:
        return False, 0.0

    served = 0
    for req in instance.get("ride_requests", []):
        pickup, dropoff = req["pickup"], req["dropoff"]
        best_idx = None
        best_cost = float("inf")
        for i, veh in enumerate(vehicles):
            leg = p4_distance(instance, veh["location"], pickup) + p4_distance(
                instance, pickup, dropoff
            )
            if leg < best_cost:
                best_cost = leg
                best_idx = i
        if best_idx is None:
            continue
        vehicles[best_idx]["cost"] += best_cost
        vehicles[best_idx]["location"] = dropoff
        served += 1

    total_requests = len(instance.get("ride_requests", []))
    success = served == total_requests and total_requests > 0
    total_cost = sum(v["cost"] for v in vehicles)
    return success, total_cost


def p10_reference_cost(instance: dict) -> float:
    budget = float(instance.get("budget", 1_000_000))
    components = instance.get("components", [])
    suppliers = instance.get("suppliers", [])
    facilities = instance.get("facilities", [])
    if not components or not suppliers:
        return max(budget * 0.1, 1.0)
    unit = sum(float(s.get("cost_multiplier", 1.0)) * 100 for s in suppliers) / len(
        suppliers
    )
    facility = sum(float(f.get("cost_per_unit", 50)) for f in facilities) / max(
        len(facilities), 1
    )
    return max((unit + facility) * len(components), budget * 0.05, 1.0)


def p10_greedy_solution(instance: dict) -> tuple[bool, float]:
    """Greedy procurement: pick cheapest supplier per component within budget."""
    budget = float(instance.get("budget", 0))
    components = instance.get("components", [])
    suppliers = instance.get("suppliers", [])
    if not components or not suppliers:
        return False, 0.0

    cost = 0.0
    for comp in components:
        cheapest = min(
            float(s.get("cost_multiplier", 1.0)) * 100
            + float(s.get("lead_time", 0))
            for s in suppliers
        )
        cost += cheapest

    success = cost <= budget
    return success, cost


def cost_gap_percent(actual: float, reference: float) -> float:
    if reference <= 0:
        return 100.0
    return (actual / reference) * 100.0


def evaluate_baseline_instances() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id in list_instances()["P4"]:
        path = os.path.join(P4_DIR, f"{instance_id}.json")
        inst = load_instance(path)
        ref = p4_reference_cost(inst)
        success, cost = p4_greedy_solution(inst)
        rows.append(
            {
                "dataset": instance_id,
                "source": "Baseline-Greedy",
                "framework": "greedy",
                "success": success,
                "makespan": cost,
                "reference_cost": ref,
                "gap_percent": cost_gap_percent(cost, ref),
            }
        )

    for instance_id in list_instances()["P10"]:
        path = os.path.join(P10_DIR, f"{instance_id}.json")
        inst = load_instance(path)
        ref = p10_reference_cost(inst)
        success, cost = p10_greedy_solution(inst)
        rows.append(
            {
                "dataset": instance_id,
                "source": "Baseline-Greedy",
                "framework": "greedy",
                "success": success,
                "makespan": cost,
                "reference_cost": ref,
                "gap_percent": cost_gap_percent(cost, ref),
            }
        )
    return rows


def load_experiment_results() -> list[dict[str, Any]]:
    """Load framework run outputs from p4_results_* / p10_results_* JSON files."""
    rows: list[dict[str, Any]] = []
    for root in RESULT_ROOTS:
        if not os.path.isdir(root):
            continue
        for path in glob(os.path.join(root, "p*_results_*.json")):
            basename = os.path.basename(path)
            match = RESULT_FILE_RE.match(basename)
            if not match:
                continue
            _, instance_id, framework = match.groups()
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            cost = data.get("total_cost") or data.get("makespan")
            success = bool(data.get("success", False))
            ref = gap = None
            if instance_id.startswith("p4"):
                inst_path = os.path.join(P4_DIR, f"{instance_id}.json")
                if os.path.exists(inst_path):
                    ref = p4_reference_cost(load_instance(inst_path))
            else:
                inst_path = os.path.join(P10_DIR, f"{instance_id}.json")
                if os.path.exists(inst_path):
                    ref = p10_reference_cost(load_instance(inst_path))
            if ref is not None and cost is not None:
                gap = cost_gap_percent(float(cost), ref)

            if data.get("source"):
                source = data["source"]
            elif "single" in root:
                source = "Single"
            elif "gpt-4o" in root:
                source = "MAS-GPT4o"
            elif "claude" in root:
                source = "MAS-Claude4"
            else:
                source = root
            rows.append(
                {
                    "dataset": instance_id,
                    "source": source,
                    "framework": framework,
                    "success": success,
                    "makespan": cost,
                    "reference_cost": ref if instance_id.startswith("p4") else ref,
                    "gap_percent": gap,
                    "result_file": path,
                }
            )
    return rows


def find_solution_files() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for root in RESULT_ROOTS:
        if not os.path.isdir(root):
            continue
        for path in glob(os.path.join(root, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            if not path.endswith((".json", ".txt", ".log")):
                continue
            if INSTANCE_RE.search(os.path.basename(path)):
                found.append((root, path))
    return found


def main() -> None:
    print("Evaluating P4 / P10 application benchmarks")
    instances = list_instances()
    print(f"  P4 instances: {len(instances['P4'])}")
    print(f"  P10 instances: {len(instances['P10'])}")

    solution_files = find_solution_files()
    print(f"  Solution artifacts under results_*: {len(solution_files)}")

    baseline_rows = evaluate_baseline_instances()
    experiment_rows = load_experiment_results()
    p4_ok = sum(1 for r in baseline_rows if r["dataset"].startswith("p4") and r["success"])
    p10_ok = sum(
        1 for r in baseline_rows if r["dataset"].startswith("p10") and r["success"]
    )
    print(f"  Greedy baseline success — P4: {p4_ok}/100, P10: {p10_ok}/100")
    print(f"  Framework experiment results loaded: {len(experiment_rows)}")

    payload = {
        "instances": instances,
        "solution_files_found": len(solution_files),
        "solution_file_samples": [p for _, p in solution_files[:10]],
        "baseline_results": baseline_rows,
        "experiment_results": experiment_rows,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
