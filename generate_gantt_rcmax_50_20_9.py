#!/usr/bin/env python3
"""
Generate Gantt chart PNGs for rcmax_50_20_9:
- Initial plan: run1 from maple-multiple
- Final plan: best run (min makespan) from maple-multiple
- Overall plan and plans for machine 1, 2, and 20 (Machine1, Machine2, Machine19)

Data source: results_baselines_dmu/maple-multiple
- Initial: rcmax_50_20_9_maple_run1.csv
- Final:   rcmax_50_20_9_maple_run{N}.csv where run N has minimum makespan (from validation_summary.csv)
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MAPLE_MULTIPLE_DIR = os.path.join(PROJECT_ROOT, "results_baselines_dmu", "maple-multiple")
VALIDATION_SUMMARY = os.path.join(MAPLE_MULTIPLE_DIR, "validation_summary.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "plots", "gantt_rcmax_50_20_9")

DATASET = "rcmax_50_20_9"


def get_best_run_for_dataset(dataset: str):
    """Return (filename, makespan) for the run with minimum makespan for dataset, or (None, None)."""
    if not os.path.exists(VALIDATION_SUMMARY):
        return None, None
    best_file, best_makespan = None, None
    with open(VALIDATION_SUMMARY, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("dataset", "").strip() != dataset:
                continue
            try:
                makespan = int(row.get("makespan", 0))
                fname = (row.get("file") or "").strip()
                if fname and (best_makespan is None or makespan < best_makespan):
                    best_makespan = makespan
                    best_file = fname
            except (ValueError, TypeError):
                continue
    return best_file, best_makespan


def load_schedule_from_maple_csv(csv_filename: str, makespan_override=None):
    """Load schedule from a CSV in maple-multiple (job, step, machine, start, end)."""
    path = os.path.join(MAPLE_MULTIPLE_DIR, csv_filename)
    if not os.path.exists(path):
        return None, None
    schedule = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row["start"])
                end = int(row["end"])
                schedule.append({
                    "job": row.get("job", ""),
                    "step": int(row.get("step", 0)),
                    "machine": row.get("machine", ""),
                    "start": start,
                    "end": end,
                    "duration": end - start,
                })
            except (ValueError, KeyError):
                continue
    makespan = makespan_override
    if makespan is None and schedule:
        makespan = max(op["end"] for op in schedule)
    return schedule, makespan or 0


def schedule_to_machine_ops(schedule):
    """Group schedule by machine for Gantt (machine -> list of (start, end, job, step))."""
    by_machine = {}
    for op in schedule:
        m = op.get("machine", "")
        if m not in by_machine:
            by_machine[m] = []
        by_machine[m].append({
            "start": op["start"],
            "end": op["end"],
            "job": op.get("job", ""),
            "step": op.get("step", 0),
        })
    for m in by_machine:
        by_machine[m].sort(key=lambda x: x["start"])
    return by_machine


def job_to_color(job_str, index_fallback, colors=None):
    """Consistent color from job name (Job1 -> color index)."""
    if colors is None:
        colors = plt.cm.tab20.colors
    job_num = (job_str or "").replace("Job", "").strip()
    if job_num.isdigit():
        return colors[int(job_num) % len(colors)]
    return colors[index_fallback % len(colors)]


def plot_gantt_overall(schedule, makespan, title, outpath, max_machines=20):
    """One figure: all machines (overall plan). Y = machine, X = time."""
    by_machine = schedule_to_machine_ops(schedule)
    machine_order = [f"Machine{i}" for i in range(max_machines)]
    machines = [m for m in machine_order if m in by_machine]
    if not machines:
        machines = sorted(by_machine.keys(), key=lambda x: (int(x.replace("Machine", "") or 0)))

    n_machines = len(machines)
    fig_height = max(6, n_machines * 0.45)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    colors = plt.cm.tab20.colors
    bar_height = 0.75

    for i, m in enumerate(machines):
        ops = by_machine.get(m, [])
        y_pos = i
        for j, op in enumerate(ops):
            left = op["start"]
            width = max(op["end"] - op["start"], 0.5)
            color = job_to_color(op["job"], j, colors)
            ax.barh(y_pos, width, left=left, height=bar_height, color=color, edgecolor="gray", linewidth=0.4)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([m.replace("Machine", "M") for m in machines], fontsize=9)
    x_max = (makespan * 1.03 + 10) if makespan else 2000
    ax.set_xlim(0, x_max)
    ax.set_ylabel("Machine", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time", fontsize=10, fontweight="bold")
    ax.set_title(f"{title} — Makespan: {makespan}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    fig.tight_layout(pad=1.0)
    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def plot_gantt_single_machine(schedule, machine_name, makespan, title, outpath):
    """One figure: one machine. Y = machine (single row), X = time."""
    ops = [o for o in schedule if o.get("machine") == machine_name]
    ops.sort(key=lambda x: x["start"])

    if not ops:
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, (makespan * 1.03 + 10) if makespan else 100)
        ax.set_ylabel("Machine", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time", fontsize=10, fontweight="bold")
        ax.set_yticks([0])
        ax.set_yticklabels([machine_name.replace("Machine", "M")], fontsize=10)
        ax.set_title(f"{title} — Makespan: {makespan} (no operations)", fontsize=11, fontweight="bold")
        out_dir = os.path.dirname(outpath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved (empty): {outpath}")
        return

    fig, ax = plt.subplots(figsize=(12, 2.8))
    colors = plt.cm.tab20.colors
    y_pos = 0
    x_max = makespan or max(o["end"] for o in ops)
    min_bar_frac = 0.015  # only label bars wider than 1.5% of x range
    for i, op in enumerate(ops):
        left = op["start"]
        width = max(op["end"] - op["start"], 0.5)
        job = op.get("job", "")
        step = op.get("step", "")
        color = job_to_color(job, i, colors)
        ax.barh(y_pos, width, left=left, height=0.75, color=color, edgecolor="gray", linewidth=0.4)
        if width >= x_max * min_bar_frac:
            label = f"{job} s{step}"
            ax.text(left + width / 2, y_pos, label, ha="center", va="center", fontsize=7)

    ax.set_yticks([0])
    ax.set_yticklabels([machine_name.replace("Machine", "M")], fontsize=11)
    ax.set_xlim(0, x_max * 1.03 + 10)
    ax.set_xlabel("Time", fontsize=10, fontweight="bold")
    ax.set_ylabel("Machine", fontsize=10, fontweight="bold")
    ax.set_title(f"{title} — Makespan: {makespan}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    fig.tight_layout(pad=1.0)
    out_dir = os.path.dirname(outpath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initial = run1, Final = best run (min makespan) from maple-multiple
    initial_schedule, initial_makespan = load_schedule_from_maple_csv(f"{DATASET}_maple_run1.csv")
    best_file, best_makespan_val = get_best_run_for_dataset(DATASET)
    final_schedule, final_makespan = None, None
    if best_file:
        final_schedule, final_makespan = load_schedule_from_maple_csv(best_file, makespan_override=best_makespan_val)

    if not final_schedule:
        print("Could not load final plan. Expected maple-multiple validation_summary.csv and best run CSV.")
        print(f"  {MAPLE_MULTIPLE_DIR}")
        return

    # Extract run label for titles (e.g. run14 -> "run14")
    final_run_label = (best_file or "").replace(".csv", "").replace(f"{DATASET}_maple_", "") if best_file else "best"

    # 20 machines: machine 1 = Machine1, machine 2 = Machine2, machine 20 = Machine19 (0-indexed)
    machine_1 = "Machine1"
    machine_2 = "Machine2"
    machine_20 = "Machine19"

    # ---- Initial plan (run1) ----
    if initial_schedule:
        plot_gantt_overall(
            initial_schedule,
            initial_makespan,
            f"rcmax_50_20_9 — Initial plan (run1) — Overall [maple-multiple]",
            os.path.join(OUTPUT_DIR, "rcmax_50_20_9_initial_overall.png"),
        )
        plot_gantt_single_machine(
            initial_schedule,
            machine_1,
            initial_makespan,
            f"rcmax_50_20_9 — Initial plan (run1) — Machine 1 [maple-multiple]",
            os.path.join(OUTPUT_DIR, "rcmax_50_20_9_initial_machine1.png"),
        )
        plot_gantt_single_machine(
            initial_schedule,
            machine_2,
            initial_makespan,
            f"rcmax_50_20_9 — Initial plan (run1) — Machine 2 [maple-multiple]",
            os.path.join(OUTPUT_DIR, "rcmax_50_20_9_initial_machine2.png"),
        )
        plot_gantt_single_machine(
            initial_schedule,
            machine_20,
            initial_makespan,
            f"rcmax_50_20_9 — Initial plan (run1) — Machine 20 [maple-multiple]",
            os.path.join(OUTPUT_DIR, "rcmax_50_20_9_initial_machine20.png"),
        )
    else:
        print("No initial plan: rcmax_50_20_9_maple_run1.csv not found; skipping initial plan plots.")

    # ---- Final plan (best run) ----
    plot_gantt_overall(
        final_schedule,
        final_makespan,
        f"rcmax_50_20_9 — Final plan ({final_run_label}) — Overall [maple-multiple]",
        os.path.join(OUTPUT_DIR, "rcmax_50_20_9_final_overall.png"),
    )
    plot_gantt_single_machine(
        final_schedule,
        machine_1,
        final_makespan,
        f"rcmax_50_20_9 — Final plan ({final_run_label}) — Machine 1 [maple-multiple]",
        os.path.join(OUTPUT_DIR, "rcmax_50_20_9_final_machine1.png"),
    )
    plot_gantt_single_machine(
        final_schedule,
        machine_2,
        final_makespan,
        f"rcmax_50_20_9 — Final plan ({final_run_label}) — Machine 2 [maple-multiple]",
        os.path.join(OUTPUT_DIR, "rcmax_50_20_9_final_machine2.png"),
    )
    plot_gantt_single_machine(
        final_schedule,
        machine_20,
        final_makespan,
        f"rcmax_50_20_9 — Final plan ({final_run_label}) — Machine 20 [maple-multiple]",
        os.path.join(OUTPUT_DIR, "rcmax_50_20_9_final_machine20.png"),
    )

    print(f"\nAll Gantt charts saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
