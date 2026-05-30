#!/usr/bin/env python3
"""Run multi-agent frameworks on P4 (URS with disruptions) instances."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from mas_backbone_paths import mas_output_dir, mas_source_tag, normalize_backbone
from p4_dataset import build_p4_query, instance_path, list_p4_instances, load_p4_instance

try:
    from agent_frameworks_p4.autogen_multi_agent.router import run_autogen_agents
except ImportError:
    run_autogen_agents = None

try:
    from agent_frameworks_p4.crewai_multi_agent.router import run_crewai
except ImportError:
    run_crewai = None

try:
    from agent_frameworks_p4.openai_swarm_agent.router import run_swarm_agents
except ImportError:
    run_swarm_agents = None

try:
    from agent_frameworks_p4.langgraph.router import run_agent as run_langgraph
except ImportError:
    run_langgraph = None

FRAMEWORK_RUNNERS = {
    "AutoGen": run_autogen_agents,
    "CrewAI": run_crewai,
    "OpenAI_Swarm": run_swarm_agents,
    "LangGraph": run_langgraph,
}


def extract_total_cost(response: str) -> float | None:
    patterns = [
        r"FINAL TOTAL COST:\s*\$?\s*([\d,.]+)",
        r"TOTAL COST:\s*\$?\s*([\d,.]+)",
        r"total cost[:\s]*\$?\s*([\d,.]+)",
        r"Total Cost[:\s]*\$?\s*([\d,.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def run_framework(framework: str, query: str) -> dict[str, Any]:
    runner = FRAMEWORK_RUNNERS.get(framework)
    if runner is None:
        return {"success": False, "error": f"Framework {framework} not available"}

    start = time.time()
    try:
        response = runner(query)
        cost = None
        try:
            cost = extract_total_cost(str(response))
        except (TypeError, ValueError):
            pass
        return {
            "success": True,
            "response": str(response),
            "total_cost": cost,
            "execution_time": time.time() - start,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "execution_time": time.time() - start,
        }


def save_results(
    instance_id: str,
    framework: str,
    payload: dict[str, Any],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"p4_results_{instance_id}_{framework}.json")
    backbone = normalize_backbone(os.getenv("MAS_BACKBONE", "claude-4"))
    record = {
        "dataset": instance_id,
        "framework": framework,
        "source": mas_source_tag(backbone),
        "mas_backbone": backbone,
        "timestamp": datetime.now().isoformat(),
        **payload,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="P4 multi-agent framework comparison")
    parser.add_argument("--instance", type=str, help="p4_instance_001, etc.")
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["AutoGen", "CrewAI", "OpenAI_Swarm", "LangGraph"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=mas_output_dir(project_root, os.getenv("MAS_BACKBONE", "claude-4")),
    )
    parser.add_argument(
        "--mas-backbone",
        choices=["claude-4", "gpt-4o"],
        default=os.getenv("MAS_BACKBONE", "claude-4"),
        help="LLM backbone for all MAS frameworks (default: claude-4)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max instances (0=all)")
    args = parser.parse_args()
    os.environ["MAS_BACKBONE"] = normalize_backbone(args.mas_backbone)
    args.output_dir = args.output_dir or mas_output_dir(project_root, args.mas_backbone)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print(
            "No API keys found. Add ANTHROPIC_API_KEY (and/or OPENAI_API_KEY) to "
            f"{os.path.join(project_root, '.env')} — see .env.example"
        )
        sys.exit(1)

    instances = [args.instance] if args.instance else list_p4_instances()
    if args.limit > 0:
        instances = instances[: args.limit]

    for instance_id in instances:
        if not os.path.exists(instance_path(instance_id)):
            print(f"Missing instance: {instance_id}")
            continue

        instance = load_p4_instance(instance_id)
        query = build_p4_query(instance)
        print(f"\n{'=' * 60}\nP4 instance: {instance_id}\n{'=' * 60}")

        for framework in args.frameworks:
            print(f"  Running {framework}...")
            result = run_framework(framework, query)
            path = save_results(instance_id, framework, result, args.output_dir)
            status = "ok" if result.get("success") else "fail"
            print(f"    {status} -> {path}")


if __name__ == "__main__":
    main()
