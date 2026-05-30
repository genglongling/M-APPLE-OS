#!/usr/bin/env bash
# Use Python 3.11 (arm64) — required for autogen-agentchat. Loads .env from repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
cd "$ROOT"

case "${1:-}" in
  p4)
    shift
    exec arch -arm64 "$PY" applications/run_p4_framework_comparison.py "$@"
    ;;
  p10)
    shift
    exec arch -arm64 "$PY" applications/run_p10_framework_comparison.py "$@"
    ;;
  eval)
    exec "$PY" evaluate_p4_p10_benchmarks.py
    ;;
  tex)
    exec "$PY" generate_results_tex.py
    ;;
  full)
    shift
    exec arch -arm64 "$PY" applications/run_p4_p10_full_batch.py "$@"
    ;;
  full-claude)
    shift
    exec arch -arm64 env MAS_BACKBONE=claude-4 "$PY" applications/run_p4_p10_full_batch.py --mas-backbone claude-4 "$@"
    ;;
  full-gpt4o)
    shift
    exec arch -arm64 env MAS_BACKBONE=gpt-4o "$PY" applications/run_p4_p10_full_batch.py --mas-backbone gpt-4o "$@"
    ;;
  single)
    shift
    exec arch -arm64 "$PY" applications/run_p4_p10_single_agent_batch.py "$@"
    ;;
  monitor)
    shift
    exec "$PY" applications/monitor_p4_p10_batch.py "$@"
    ;;
  watch)
    exec "$PY" applications/monitor_p4_p10_batch.py --watch
    ;;
  *)
    echo "Usage: $0 {p4|p10|full|full-claude|full-gpt4o|single|monitor|watch|eval|tex} [args...]"
    echo "  $0 full-claude --skip-existing   # MAS + Claude-4 (default backbone)"
    echo "  $0 full-gpt4o --skip-existing    # MAS + GPT-4o table rows"
    echo "  $0 single --skip-existing        # 4 single-agent models × P4/P10"
    echo "  $0 watch                         # progress every 30s"
    exit 1
    ;;
esac
