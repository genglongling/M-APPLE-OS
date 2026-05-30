#!/usr/bin/env python3
"""
Full P4 + P10 batch: all instances × all frameworks.
Resumes from existing result JSON files. Logs progress for monitoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv(os.path.join(project_root, ".env"))

from mas_backbone_paths import mas_output_dir, normalize_backbone
from p4_dataset import build_p4_query, list_p4_instances, load_p4_instance, instance_path as p4_path
from p10_dataset import build_p10_query, list_p10_instances, load_p10_instance, instance_path as p10_path

# Import runners from comparison scripts
from run_p4_framework_comparison import (
    FRAMEWORK_RUNNERS as P4_RUNNERS,
    run_framework as run_p4_fw,
    save_results as save_p4,
)
from run_p10_framework_comparison import (
    FRAMEWORK_RUNNERS as P10_RUNNERS,
    run_framework as run_p10_fw,
    save_results as save_p10,
)

DEFAULT_FRAMEWORKS = ["AutoGen", "CrewAI", "OpenAI_Swarm", "LangGraph"]
LOG_DIR = os.path.join(project_root, "logs")
PROGRESS_FILE = os.path.join(LOG_DIR, "p4_p10_progress.json")
LOG_FILE = os.path.join(LOG_DIR, "p4_p10_batch.log")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def result_path(output_dir: str, prefix: str, instance_id: str, framework: str) -> str:
    return os.path.join(output_dir, f"{prefix}_results_{instance_id}_{framework}.json")


def should_skip(path: str, force: bool) -> bool:
    if force or not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("success"))
    except (json.JSONDecodeError, OSError):
        return False


def write_progress(state: dict[str, Any]) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def run_dataset(
    dataset: str,
    instances: list[str],
    frameworks: list[str],
    output_dir: str,
    build_query,
    load_instance,
    inst_path_fn,
    run_fw,
    save_fn,
    prefix: str,
    skip_existing: bool,
    force: bool,
    state: dict[str, Any],
) -> None:
    total = len(instances) * len(frameworks)
    done = 0
    for i, instance_id in enumerate(instances):
        if not os.path.exists(inst_path_fn(instance_id)):
            log(f"SKIP missing {dataset} {instance_id}")
            continue
        query = build_query(load_instance(instance_id))
        for framework in frameworks:
            out = result_path(output_dir, prefix, instance_id, framework)
            if skip_existing and should_skip(out, force):
                log(f"SKIP {dataset} {instance_id} {framework} (exists)")
                state["skipped"] = state.get("skipped", 0) + 1
                done += 1
                continue

            if P4_RUNNERS.get(framework) is None and dataset == "P4":
                log(f"SKIP {dataset} {framework} (import unavailable)")
                state["failed"] = state.get("failed", 0) + 1
                done += 1
                continue

            log(f"RUN {dataset} {instance_id} {framework} ({done + 1}/{total})")
            state["current"] = {
                "dataset": dataset,
                "instance": instance_id,
                "framework": framework,
                "index": done + 1,
                "total": total,
            }
            write_progress(state)

            t0 = time.time()
            try:
                result = run_fw(framework, query)
                path = save_fn(instance_id, framework, result, output_dir)
                ok = result.get("success", False)
                status = "ok" if ok else "fail"
                log(f"DONE {status} {path} ({time.time() - t0:.1f}s)")
                if ok:
                    state["succeeded"] = state.get("succeeded", 0) + 1
                else:
                    state["failed"] = state.get("failed", 0) + 1
            except Exception as exc:
                log(f"ERROR {dataset} {instance_id} {framework}: {exc}")
                traceback.print_exc()
                save_fn(
                    instance_id,
                    framework,
                    {"success": False, "error": str(exc), "execution_time": time.time() - t0},
                    output_dir,
                )
                state["failed"] = state.get("failed", 0) + 1

            done += 1
            state["completed_jobs"] = done
            write_progress(state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full P4+P10 batch with resume")
    parser.add_argument(
        "--mas-backbone",
        choices=["claude-4", "gpt-4o"],
        default=os.getenv("MAS_BACKBONE", "claude-4"),
        help="LLM backbone: claude-4 (default) or gpt-4o for MAS (GPT-4o) table rows",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override results dir (default: results_mas(<backbone>))",
    )
    parser.add_argument("--frameworks", nargs="+", default=DEFAULT_FRAMEWORKS)
    parser.add_argument("--p4-only", action="store_true")
    parser.add_argument("--p10-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--force", action="store_true", help="Re-run even if success JSON exists")
    parser.add_argument("--limit-p4", type=int, default=0)
    parser.add_argument("--limit-p10", type=int, default=0)
    args = parser.parse_args()
    os.environ["MAS_BACKBONE"] = normalize_backbone(args.mas_backbone)
    if not args.output_dir:
        args.output_dir = mas_output_dir(project_root, args.mas_backbone)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        log("ERROR: No API keys in .env")
        sys.exit(1)

    unavailable = [f for f in args.frameworks if P4_RUNNERS.get(f) is None]
    if unavailable:
        log(f"WARNING: unavailable frameworks (not imported): {unavailable}")

    state: dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "output_dir": args.output_dir,
        "mas_backbone": args.mas_backbone,
        "frameworks": args.frameworks,
        "succeeded": 0,
        "failed": 0,
        "skipped": 0,
        "completed_jobs": 0,
        "status": "running",
    }
    write_progress(state)
    log(
        f"Batch start — backbone={args.mas_backbone} "
        f"frameworks={args.frameworks} output={args.output_dir}"
    )

    try:
        if not args.p10_only:
            p4_list = list_p4_instances()
            if args.limit_p4 > 0:
                p4_list = p4_list[: args.limit_p4]
            run_dataset(
                "P4",
                p4_list,
                args.frameworks,
                args.output_dir,
                build_p4_query,
                load_p4_instance,
                p4_path,
                run_p4_fw,
                save_p4,
                "p4",
                args.skip_existing,
                args.force,
                state,
            )

        if not args.p4_only:
            p10_list = list_p10_instances()
            if args.limit_p10 > 0:
                p10_list = p10_list[: args.limit_p10]
            run_dataset(
                "P10",
                p10_list,
                args.frameworks,
                args.output_dir,
                build_p10_query,
                load_p10_instance,
                p10_path,
                run_p10_fw,
                save_p10,
                "p10",
                args.skip_existing,
                args.force,
                state,
            )

        state["status"] = "completed"
        state["finished_at"] = datetime.now().isoformat()
        write_progress(state)
        log(
            f"Batch complete — ok={state.get('succeeded', 0)} "
            f"fail={state.get('failed', 0)} skip={state.get('skipped', 0)}"
        )

        log("Running evaluate_p4_p10_benchmarks.py ...")
        import evaluate_p4_p10_benchmarks

        evaluate_p4_p10_benchmarks.main()

        log("Running generate_results_tex.py ...")
        import generate_results_tex

        generate_results_tex.main()
        log("Tables updated: results.tex")
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["finished_at"] = datetime.now().isoformat()
        write_progress(state)
        log("Interrupted — progress saved; re-run to resume (--skip-existing)")
        sys.exit(130)
    except Exception as exc:
        state["status"] = "error"
        state["error"] = str(exc)
        write_progress(state)
        log(f"Fatal error: {exc}")
        raise


if __name__ == "__main__":
    main()
