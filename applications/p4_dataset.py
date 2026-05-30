"""Load P4 (URS with disruptions) instances and build agent prompts."""

from __future__ import annotations

import json
import os
from typing import Any

P4_DIR = os.path.join(os.path.dirname(__file__), "P4", "disruptions")


def list_p4_instances() -> list[str]:
    return sorted(
        f.replace(".json", "")
        for f in os.listdir(P4_DIR)
        if f.startswith("p4_instance_") and f.endswith(".json")
    )


def load_p4_instance(instance_id: str) -> dict[str, Any]:
    path = os.path.join(P4_DIR, f"{instance_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_p4_query(instance: dict[str, Any]) -> str:
    instance_id = instance.get("instance_id", "unknown")
    locations = instance.get("city_map", {}).get("locations", [])
    distances = instance.get("city_map", {}).get("distances", {})
    requests = instance.get("ride_requests", [])
    vehicles = instance.get("vehicles", [])
    disruptions = instance.get("disruptions", [])

    query = f"""
Urban Ride-Sharing with Disruptions (URS / P4) - Instance: {instance_id}

Problem Description:
- Serve all ride requests using available vehicles
- Respect pickup/dropoff locations, time windows, and passenger deadlines
- City locations: {', '.join(locations)}
- Handle traffic disruptions (accidents, construction) affecting routes
- Goal: Maximize served requests; minimize total travel cost and violations

Ride Requests:
"""
    for req in requests:
        tw = req.get("time_window", [])
        query += (
            f"\n  - {req.get('passenger_id')}: "
            f"{req.get('pickup')} -> {req.get('dropoff')}, "
            f"time_window={tw}, deadline={req.get('deadline')}"
        )

    query += "\n\nVehicles:\n"
    for veh in vehicles:
        query += (
            f"\n  - {veh.get('vehicle_id')}: at {veh.get('location')}, "
            f"capacity={veh.get('capacity')}, fuel={veh.get('fuel')}"
        )

    query += "\n\nDisruptions:\n"
    for dis in disruptions:
        query += f"\n  - {dis}"

    if distances:
        sample_keys = list(distances.keys())[:12]
        query += "\n\nSample travel distances (location pairs):\n"
        for key in sample_keys:
            query += f"\n  - {key}: {distances[key]}"

    query += """

REQUIRED OUTPUT FORMAT:
1. FINAL TOTAL COST: [numeric value]
2. REQUESTS SERVED: [count] / [total]
3. VEHICLE ASSIGNMENTS:
   - Vehicle: [id], Route: [ordered stops], Cost: [value]
4. DISRUPTION HANDLING:
   - For each disruption: mitigation strategy and impact on schedule

Provide a feasible assignment plan that serves as many requests as possible
within deadlines and respects vehicle capacity and disruptions.
"""
    return query.strip()


def instance_path(instance_id: str) -> str:
    return os.path.join(P4_DIR, f"{instance_id}.json")
