#!/usr/bin/env python3
"""Run single-agent LLMs on all P4/P10 instances (fills Single rows in results.tex)."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Callable

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from p4_dataset import build_p4_query, list_p4_instances, load_p4_instance
from p10_dataset import build_p10_query, list_p10_instances, load_p10_instance

MODELS = ["GPT-4o", "Claude-Sonnet-4", "Gemini-2.5", "DeepSeek-V3"]
LOG_DIR = os.path.join(project_root, "logs")
LOG_FILE = os.path.join(LOG_DIR, "p4_p10_single_agent.log")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_total_cost(response: str) -> float | None:
    patterns = [
        r"FINAL TOTAL COST:\s*\$?\s*([\d,.]+)",
        r"TOTAL COST:\s*\$?\s*([\d,.]+)",
        r"total cost[:\s]*\$?\s*([\d,.]+)",
        r"Total Cost[:\s]*\$?\s*([\d,.]+)",
        r"total procurement cost[:\s]*\$?\s*([\d,.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def call_gpt4o(query: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert optimization solver. Provide a feasible plan and state FINAL TOTAL COST.",
            },
            {"role": "user", "content": query},
        ],
        max_tokens=8192,
        temperature=0.1,
    )
    return r.choices[0].message.content or ""


def call_claude(query: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        temperature=0.1,
        system="You are an expert optimization solver. Provide a feasible plan and state FINAL TOTAL COST.",
        messages=[{"role": "user", "content": query}],
    )
    return r.content[0].text


def call_gemini(query: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    r = model.generate_content(
        f"You are an expert optimization solver.\n\n{query}\n\nState FINAL TOTAL COST.",
        generation_config=genai.types.GenerationConfig(max_output_tokens=8192, temperature=0.1),
    )
    return r.text


def call_deepseek(query: str) -> str:
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com/anthropic",
    )
    r = client.messages.create(
        model="deepseek-chat",
        max_tokens=8192,
        temperature=0.1,
        system="You are an expert optimization solver. Provide a feasible plan and state FINAL TOTAL COST.",
        messages=[{"role": "user", "content": query}],
    )
    return r.content[0].text


RUNNERS: dict[str, Callable[[str], str]] = {
    "GPT-4o": call_gpt4o,
    "Claude-Sonnet-4": call_claude,
    "Gemini-2.5": call_gemini,
    "DeepSeek-V3": call_deepseek,
}


def result_path(output_dir: str, prefix: str, instance_id: str, model: str) -> str:
    safe = model.replace(" ", "_")
    return os.path.join(output_dir, f"{prefix}_results_{instance_id}_{safe}.json")


def should_skip(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("success"))
    except (json.JSONDecodeError, OSError):
        return False


def run_one(
    output_dir: str,
    prefix: str,
    instance_id: str,
    model: str,
    query: str,
) -> None:
    out = result_path(output_dir, prefix, instance_id, model)
    if should_skip(out):
        log(f"SKIP {prefix} {instance_id} {model}")
        return
    log(f"RUN {prefix} {instance_id} {model}")
    t0 = time.time()
    try:
        text = RUNNERS[model](query)
        cost = extract_total_cost(text)
        payload = {
            "dataset": instance_id,
            "framework": model,
            "source": "Single",
            "success": True,
            "response": text,
            "total_cost": cost,
            "makespan": cost,
            "execution_time": time.time() - t0,
            "timestamp": datetime.now().isoformat(),
        }
        status = "ok"
    except Exception as exc:
        payload = {
            "dataset": instance_id,
            "framework": model,
            "source": "Single",
            "success": False,
            "error": str(exc),
            "execution_time": time.time() - t0,
            "timestamp": datetime.now().isoformat(),
        }
        status = "fail"
    os.makedirs(output_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    log(f"DONE {status} {out} ({payload['execution_time']:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="P4/P10 single-agent batch")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "results_single(gpt-4o)"),
    )
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--p4-only", action="store_true")
    parser.add_argument("--p10-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    p4 = list_p4_instances()
    p10 = list_p10_instances()
    if args.limit > 0:
        p4, p10 = p4[: args.limit], p10[: args.limit]

    log(f"Single-agent batch — models={args.models} output={args.output_dir}")

    if not args.p10_only:
        for iid in p4:
            inst = load_p4_instance(iid)
            query = build_p4_query(inst)
            for model in args.models:
                if args.skip_existing and should_skip(
                    result_path(args.output_dir, "p4", iid, model)
                ):
                    continue
                run_one(args.output_dir, "p4", iid, model, query)

    if not args.p4_only:
        for iid in p10:
            inst = load_p10_instance(iid)
            query = build_p10_query(inst)
            for model in args.models:
                if args.skip_existing and should_skip(
                    result_path(args.output_dir, "p10", iid, model)
                ):
                    continue
                run_one(args.output_dir, "p10", iid, model, query)

    log("Single-agent batch complete")
    import evaluate_p4_p10_benchmarks
    import generate_results_tex

    evaluate_p4_p10_benchmarks.main()
    generate_results_tex.main()


if __name__ == "__main__":
    main()
