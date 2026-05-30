# agent_frameworks_p4

Multi-agent frameworks for **P4: Urban Ride-Sharing with Disruptions (URS)**.

Forked from `agent_frameworks_jssp` (same layout as `agent_frameworks`), with routers that dispatch on URS/P4 prompts.

## Agents

- **Ride Scheduler** — assign vehicles to ride requests
- **Disruption Coordinator** — handle traffic incidents and route changes
- **URS Supervisor** — final coordinated plan and total cost

## Frameworks

| Directory | Entry point |
|-----------|-------------|
| `autogen_multi_agent/router.py` | `run_autogen_agents(query)` |
| `crewai_multi_agent/router.py` | `run_crewai(query)` |
| `openai_swarm_agent/router.py` | `run_swarm_agents(query)` |
| `langgraph/router.py` | `run_agent(query)` |

## Dataset & runner

- Instances: `applications/P4/disruptions/p4_instance_*.json`
- Prompt builder: `applications/p4_dataset.py`
- Batch runner: `applications/run_p4_framework_comparison.py`

```bash
python applications/run_p4_framework_comparison.py \
  --instance p4_instance_001 \
  --frameworks AutoGen CrewAI \
  --output-dir results_mas(gpt-4o)
```

Outputs: `p4_results_{instance_id}_{Framework}.json`
