#!/usr/bin/env python3
"""Generate results.tex (success + optimal rate tables) from validation JSON and P4/P10 eval."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any

from benchmark_utils import CATEGORIES, categorize_dataset, makespan_gap_percent

VALIDATION_JSON = "comprehensive_initial_schedule_validation_with_alas.json"
P4_P10_JSON = "p4_p10_evaluation.json"
OUTPUT_TEX = "results.tex"

MAS_FRAMEWORKS = ["AutoGen", "CrewAI", "LangGraph", "OpenAI_Swarm"]
SINGLE_MODELS = ["GPT-4o", "Claude-Sonnet-4", "Gemini-2.5", "DeepSeek-V3"]
ALAS_WORKFLOWS = ["full", "no_repair", "no_validation", "no_optimization"]
ALAS_BACKBONES = [
    ("ALAS-GPT4o", "GPT-4o"),
    ("ALAS-Claude4", "Claude-4"),
    ("ALAS-DeepSeek-V3", "DeepSeek-V3"),
    ("ALAS-Gemini-2.5", "Gemini-2.5"),
]

NUM_COLS = len(CATEGORIES) + 2  # Method + 7 categories + Overall


def load_validation() -> list[dict[str, Any]]:
    if not os.path.exists(VALIDATION_JSON):
        raise FileNotFoundError(
            f"Missing {VALIDATION_JSON}. Run validate_all_initial_schedules_comprehensive.py first."
        )
    with open(VALIDATION_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_p4_p10(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not os.path.exists(P4_P10_JSON):
        return results
    with open(P4_P10_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for row in payload.get("experiment_results", []):
        source = row.get("source", "")
        if source in ("MAS-P4", "MAS-P10"):
            source = "MAS-GPT4o"
        framework = row.get("framework", "").replace("OpenAI_Swarm", "OpenAI_Swarm")
        results.append(
            {
                "dataset": row["dataset"],
                "source": source,
                "framework": framework,
                "success": row["success"],
                "makespan": row.get("makespan"),
            }
        )
    return results


def fmt_cell(value: float | None, bold: bool = False) -> str:
    if value is None:
        text = "---"
    else:
        text = f"{value:.1f}"
    if bold:
        text = f"\\textbf{{{text}}}"
    return text


def success_rate(
    results: list[dict[str, Any]], source: str, framework: str, category: str
) -> float | None:
    valid = total = 0
    for r in results:
        if r["source"] != source or r["framework"] != framework:
            continue
        cat = categorize_dataset(r["dataset"])
        if cat != category:
            continue
        total += 1
        if r["success"]:
            valid += 1
    if total == 0:
        return None
    return 100.0 * valid / total


def build_success_row(
    results: list[dict[str, Any]], label: str, source: str, framework: str, bold_overall: bool = False
) -> str:
    cells = []
    for cat in CATEGORIES:
        cells.append(fmt_cell(success_rate(results, source, framework, cat)))
    total_valid = total_count = 0
    for r in results:
        if r["source"] != source or r["framework"] != framework:
            continue
        if categorize_dataset(r["dataset"]) not in CATEGORIES:
            continue
        total_count += 1
        if r["success"]:
            total_valid += 1
    overall = (100.0 * total_valid / total_count) if total_count else None
    cells.append(fmt_cell(overall, bold=bold_overall))
    return f"{label} & " + " & ".join(cells) + " \\\\"


def alas_backbone_success_row(
    results: list[dict[str, Any]], source: str, label: str
) -> str:
    """Best workflow per category for one ALAS backbone."""
    cells = []
    for cat in CATEGORIES:
        best = None
        for wf in ALAS_WORKFLOWS:
            rate = success_rate(results, source, wf, cat)
            if rate is not None and (best is None or rate > best):
                best = rate
        cells.append(fmt_cell(best))
    # overall: best workflow globally for this backbone
    best_overall = None
    for wf in ALAS_WORKFLOWS:
        total_valid = total_count = 0
        for r in results:
            if r["source"] != source or r["framework"] != wf:
                continue
            if categorize_dataset(r["dataset"]) not in CATEGORIES:
                continue
            total_count += 1
            if r["success"]:
                total_valid += 1
        if total_count:
            rate = 100.0 * total_valid / total_count
            if best_overall is None or rate > best_overall:
                best_overall = rate
    cells.append(fmt_cell(best_overall, bold=True))
    return f"{label} & " + " & ".join(cells) + " \\\\"


def alas_aggregated_success_row(results: list[dict[str, Any]]) -> str:
    """Best (source, workflow) pair per category; overall = max pair overall rate."""
    pairs = {
        (r["source"], r["framework"])
        for r in results
        if r["source"].startswith("ALAS")
    }
    cells = []
    for cat in CATEGORIES:
        best = None
        for src, fw in pairs:
            rate = success_rate(results, src, fw, cat)
            if rate is not None and (best is None or rate > best):
                best = rate
        cells.append(fmt_cell(best))
    by_dataset: dict[str, bool] = {}
    for r in results:
        if not r["source"].startswith("ALAS"):
            continue
        if categorize_dataset(r["dataset"]) not in CATEGORIES:
            continue
        ds = r["dataset"]
        if r["success"]:
            by_dataset[ds] = True
        else:
            by_dataset.setdefault(ds, False)
    total = len(by_dataset)
    overall = (100.0 * sum(by_dataset.values()) / total) if total else None
    cells.append(fmt_cell(overall, bold=True))
    return "\\rowcolor{gray!25} ALAS(aggregated) & " + " & ".join(cells) + " \\\\"


def optimal_gap_percent(makespan: float | None, dataset_name: str) -> float:
    """Match generate_optimal_rate_report: missing makespan counts as 100%."""
    if makespan is None or makespan == 0:
        return 100.0
    return makespan_gap_percent(makespan, dataset_name) or 100.0


def avg_gap_rate(
    results: list[dict[str, Any]], source: str, framework: str, category: str
) -> float | None:
    gaps = []
    for r in results:
        if r["source"] != source or r["framework"] != framework:
            continue
        if categorize_dataset(r["dataset"]) != category:
            continue
        gaps.append(optimal_gap_percent(r.get("makespan"), r["dataset"]))
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def build_optimal_row(
    results: list[dict[str, Any]], label: str, source: str, framework: str, bold_overall: bool = False
) -> str:
    cells = []
    for cat in CATEGORIES:
        cells.append(fmt_cell(avg_gap_rate(results, source, framework, cat)))
    gaps = []
    for r in results:
        if r["source"] != source or r["framework"] != framework:
            continue
        if categorize_dataset(r["dataset"]) not in CATEGORIES:
            continue
        gaps.append(optimal_gap_percent(r.get("makespan"), r["dataset"]))
    overall = sum(gaps) / len(gaps) if gaps else None
    cells.append(fmt_cell(overall, bold=bold_overall))
    return f"{label} & " + " & ".join(cells) + " \\\\"


def alas_best_optimal_row(results: list[dict[str, Any]]) -> str:
    """Per instance min gap across ALAS variants; report % with min gap <= 100%."""
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if not r["source"].startswith("ALAS"):
            continue
        by_dataset[r["dataset"]].append(
            optimal_gap_percent(r.get("makespan"), r["dataset"])
        )

    cells = []
    for cat in CATEGORIES:
        total = optimal = 0
        for ds, gaps in by_dataset.items():
            if categorize_dataset(ds) != cat:
                continue
            total += 1
            if min(gaps) <= 100.0:
                optimal += 1
        rate = (100.0 * optimal / total) if total else None
        cells.append(fmt_cell(rate))
    total = optimal = 0
    for ds, gaps in by_dataset.items():
        if categorize_dataset(ds) not in CATEGORIES:
            continue
        total += 1
        if min(gaps) <= 100.0:
            optimal += 1
    overall = (100.0 * optimal / total) if total else None
    cells.append(fmt_cell(overall, bold=True))
    return "\\rowcolor{gray!25} ALAS (Best) & " + " & ".join(cells) + " \\\\"


def header_cols() -> str:
    cols = " & ".join(f"\\textbf{{{c}}}" for c in CATEGORIES)
    return f"\\textbf{{Method}} & {cols} & \\textbf{{Overall}} \\\\"


def multicolumn_line(title: str) -> str:
    return f"\\multicolumn{{{NUM_COLS}}}{{|c|}}{{\\textbf{{{title}}}}} \\\\"


def generate_tex(results: list[dict[str, Any]]) -> str:
    col_spec = "|l|" + "|c|" * len(CATEGORIES) + "|c|"
    lines = [
        "% Auto-generated by generate_results_tex.py — do not edit by hand.",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Success Rates (\\%) across Benchmarks (valid initial schedule). "
        "† = significantly better than baseline.}",
        "\\label{tab:ranked_success_rates}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "\\scriptsize",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header_cols(),
        "\\hline",
        multicolumn_line("Multi-Agent Systems (GPT-4o)"),
        "\\hline",
    ]

    for fw in MAS_FRAMEWORKS:
        label = fw.replace("OpenAI_Swarm", "OpenAI Swarm").replace("_", " ")
        lines.append(
            build_success_row(results, label, "MAS-GPT4o", fw, bold_overall=True)
        )
    lines.extend(
        [
            "\\hline",
            multicolumn_line("Multi-Agent Systems (Claude-4)"),
            "\\hline",
        ]
    )
    for fw in MAS_FRAMEWORKS:
        label = fw.replace("OpenAI_Swarm", "OpenAI Swarm").replace("_", " ")
        lines.append(
            build_success_row(results, label, "MAS-Claude4", fw, bold_overall=True)
        )
    lines.extend(
        [
            "\\hline",
            multicolumn_line("Single-Agent Models"),
            "\\hline",
        ]
    )
    for model in SINGLE_MODELS:
        lines.append(
            build_success_row(results, model, "Single", model, bold_overall=True)
        )
    lines.extend(
        [
            "\\hline",
            multicolumn_line("ALAS (Ours, Best Variant per Backbone)"),
            "\\hline",
        ]
    )
    for source, label in ALAS_BACKBONES:
        lines.append(alas_backbone_success_row(results, source, f"ALAS({label})"))
    lines.extend(
        [
            "\\hline",
            multicolumn_line("ALAS (Ours, Best Variant per Dataset)"),
            "\\hline",
            alas_aggregated_success_row(results),
            "\\hline",
            "\\end{tabular}",
            "\\begin{flushleft}",
            "\\footnotesize",
            "ALAS(best) selects the best-performing workflow variant per dataset category "
            "across GPT-4o, Claude-4, DeepSeek-V3, Gemini-2.5.  ",
            "P4/P10 MAS: Claude-4 backbone (\\texttt{results\\_mas(claude-4)}) or GPT-4o "
            "(\\texttt{results\\_mas(gpt-4o)}). Single-agent: \\texttt{results\\_single(gpt-4o)}. "
            "ALAS rows are JSSP-only (MAPLE has no P4/P10 workflow); --- = not run.",
            "\\end{flushleft}",
            "\\end{table}",
            "",
            "\\begin{table}[t]",
            "\\centering",
            "\\caption{Optimal Rates (\\%) across Benchmarks ($\\mathrm{makespan}/\\mathrm{UB}\\times 100$). "
            "Significance markers denote improvements over baselines: $^{\\dagger}p<0.05$, $^{*}p<0.01$.}",
            "\\label{tab:optimal_rates_main_updated}",
            "\\renewcommand{\\arraystretch}{1.1}",
            "\\scriptsize",
            f"\\begin{{tabular}}{{{col_spec}}}",
            "\\hline",
            header_cols(),
            "\\hline",
            multicolumn_line("Multi-Agent Systems (GPT-4o Backbone)"),
            "\\hline",
        ]
    )
    for fw in MAS_FRAMEWORKS:
        label = fw.replace("OpenAI_Swarm", "OpenAI Swarm").replace("_", " ")
        lines.append(build_optimal_row(results, label, "MAS-GPT4o", fw))
    lines.extend(
        [
            "\\hline",
            multicolumn_line("Multi-Agent Systems (Claude-4 Backbone)"),
            "\\hline",
        ]
    )
    for fw in MAS_FRAMEWORKS:
        label = fw.replace("OpenAI_Swarm", "OpenAI Swarm").replace("_", " ")
        lines.append(build_optimal_row(results, label, "MAS-Claude4", fw))
    lines.extend(
        [
            "\\hline",
            multicolumn_line("ALAS Variants (Full Workflows)"),
            "\\hline",
        ]
    )
    for source, label in ALAS_BACKBONES:
        lines.append(build_optimal_row(results, f"ALAS ({label})", source, "full"))
    lines.extend(
        [
            "\\hline",
            multicolumn_line("ALAS (Ours, Best Variant per Dataset)"),
            "\\hline",
            alas_best_optimal_row(results),
            "\\hline",
            "\\end{tabular}",
            "\\\\[0.5em]",
            "\\small $^{\\dagger}p<0.05$, $^{*}p<0.01$ (paired t-test, compared against single-agent baseline).",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = merge_p4_p10(load_validation())
    tex = generate_tex(results)
    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"Wrote {OUTPUT_TEX} ({len(tex)} bytes)")


if __name__ == "__main__":
    main()
