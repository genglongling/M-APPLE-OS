# agent_frameworks_p10

Multi-agent frameworks for **P10: Global GPU Supply Chain Planning**.

Forked from `agent_frameworks_jssp`, with routers that dispatch on P10 supply-chain prompts.

## Agents

- **Procurement Planner** — supplier selection and component sourcing
- **Risk Analyst** — disruption scenarios and mitigation
- **Supply Chain Supervisor** — final plan, cost, and budget use

## Frameworks

Same four frameworks as P4/JSSP; see `agent_frameworks_p4/README.md` for entry points (replace `p4` with `p10`).

## Dataset & runner

- Instances: `applications/P10/custom/p10_instance_*.json`
- Prompt builder: `applications/p10_dataset.py`
- Batch runner: `applications/run_p10_framework_comparison.py`

```bash
python applications/run_p10_framework_comparison.py \
  --instance p10_instance_001 \
  --frameworks LangGraph \
  --output-dir results_mas(gpt-4o)
```

Outputs: `p10_results_{instance_id}_{Framework}.json`
