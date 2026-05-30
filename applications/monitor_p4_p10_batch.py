#!/usr/bin/env python3
"""Print live P4/P10 batch progress from logs/p4_p10_progress.json."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS = os.path.join(project_root, "logs", "p4_p10_progress.json")
LOG = os.path.join(project_root, "logs", "p4_p10_batch.log")


def count_results(output_dir: str) -> dict[str, int]:
    counts = {"p4_ok": 0, "p4_fail": 0, "p10_ok": 0, "p10_fail": 0}
    if not os.path.isdir(output_dir):
        return counts
    for name in os.listdir(output_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(output_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            ok = d.get("success", False)
        except (json.JSONDecodeError, OSError):
            ok = False
        if name.startswith("p4_results_"):
            counts["p4_ok" if ok else "p4_fail"] += 1
        elif name.startswith("p10_results_"):
            counts["p10_ok" if ok else "p10_fail"] += 1
    return counts


def main() -> None:
    watch = "--watch" in sys.argv
    output_dir = os.path.join(project_root, "results_mas(gpt-4o)")

    while True:
        print(f"\n--- {datetime.now().strftime('%H:%M:%S')} ---")
        if os.path.isfile(PROGRESS):
            with open(PROGRESS, "r", encoding="utf-8") as f:
                p = json.load(f)
            cur = p.get("current", {})
            print(f"Status: {p.get('status')} | jobs: {p.get('completed_jobs', 0)}")
            print(
                f"  ok={p.get('succeeded', 0)} fail={p.get('failed', 0)} "
                f"skip={p.get('skipped', 0)}"
            )
            if cur:
                print(
                    f"  Current: {cur.get('dataset')} {cur.get('instance')} "
                    f"{cur.get('framework')} ({cur.get('index')}/{cur.get('total')})"
                )
        else:
            print("No progress file yet (batch not started?)")

        c = count_results(output_dir)
        print(
            f"Result files: P4 ok={c['p4_ok']} fail={c['p4_fail']} | "
            f"P10 ok={c['p10_ok']} fail={c['p10_fail']} (max 400 each)"
        )
        if os.path.isfile(LOG):
            with open(LOG, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines[-5:]:
                print(f"  log: {line.rstrip()}")

        if not watch:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
