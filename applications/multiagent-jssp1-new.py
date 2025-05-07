import sys
import os

# Set up project root and src path
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# Define job steps and machine constraints for Experiment 1
jobs = [
    {'name': 'Job1', 'steps': [('MachineA', 3), ('MachineB', 2), ('MachineC', 2)]},
    {'name': 'Job2', 'steps': [('MachineA', 2), ('MachineC', 1), ('MachineB', 4)]},
    {'name': 'Job3', 'steps': [('MachineB', 4), ('MachineA', 1), ('MachineC', 3)]},
]

# Helper: create a placeholder schedule for demonstration
# In a real system, the agent would compute this based on constraints
step_labels = ['A', 'B', 'C', 'D', 'E']
def make_placeholder_schedule(job, offset=0):
    schedule = []
    t = offset
    for idx, (machine, duration) in enumerate(job['steps']):
        schedule.append({
            'job': job['name'],
            'step': idx+1,
            'machine': machine,
            'start': t,
            'end': t+duration,
            'precedence': f"After {job['name']} Step {idx}" if idx > 0 else None
        })
        t += duration
    return schedule

# Create agents for each job, with run() returning a standardized schedule
class JSSPAgent(Agent):
    def run(self):
        # For demo, use a placeholder schedule with a unique offset for each job
        job_idx = int(self.name.split('Job')[1][0]) - 1
        return {'schedule': make_placeholder_schedule(jobs[job_idx], offset=job_idx*2)}

agents = []
for job in jobs:
    agent = JSSPAgent(
        name=f"{job['name']} Agent",
        backstory=f"Agent for {job['name']} scheduling.",
        task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
        task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints."
    )
    agents.append(agent)

# Add a final Supervisor Agent
class SupervisorAgent(Agent):
    def run(self):
        # No-op: aggregation is handled outside
        return {}

supervisor_agent = SupervisorAgent(
    name="Supervisor Agent",
    backstory="Aggregates all job schedules and produces the overall JSSP schedule.",
    task_description="Combine all job agent schedules into a single overall JSSP schedule.",
    task_expected_output="Overall JSSP schedule as a table."
)
agents.append(supervisor_agent)

# Disruption event: MachineA unavailable from t=4 to t=6
# We'll encode this as part of the task_spec
task_spec = {
    'nodes': [
        {'agent': agents[0], 'dependencies': []},
        {'agent': agents[1], 'dependencies': []},
        {'agent': agents[2], 'dependencies': []},
        {'agent': supervisor_agent, 'dependencies': [agents[0].name, agents[1].name, agents[2].name]},
    ],
    'edges': [],
    'jobs': jobs,
    'disruptions': [
        {'machine': 'MachineA', 'unavailable': [(4, 6)]}
    ],
    'rules': [
        'Each job must perform its steps strictly in order.',
        'Each machine can only handle one operation at a time.',
        'No two operations use the same machine at the same time.'
    ]
}

# Initialize MAPLE (STATIC: dynamic_adaptation=False)
maple = MAPLE(task_spec, dynamic_adaptation=False)

# Run MAPLE (with dynamic adaptation enabled)
maple.run(with_rollback=True, validate=True)

# Aggregate and print overall schedule from all job agents
context = maple.executor.context
all_schedules = []
for agent in agents[:-1]:  # Exclude supervisor agent
    output = context.get(agent.name, {})
    if isinstance(output, dict) and 'schedule' in output:
        all_schedules.extend(output['schedule'])

if all_schedules:
    # Sort by start time for overall schedule
    all_schedules.sort(key=lambda x: (x.get('start', 0), x.get('machine', ''), x.get('job', '')))
    print("\n| Job  | Step | Machine  | Time Slot | Precedence Constraints      |")
    print("|------|------|----------|-----------|----------------------------|")
    for entry in all_schedules:
        job = entry.get('job', '?')
        step = entry.get('step', '?')
        machine = entry.get('machine', '?')
        start = entry.get('start', '?')
        end = entry.get('end', '?')
        prec = entry.get('precedence', 'None')
        print(f"| {job} | {step} | {machine} | {start}-{end} | {prec} |")
    # Store in supervisor agent's context
    context[supervisor_agent.name] = {'schedule': all_schedules}
else:
    print("\n(No detailed schedules found from job agents. Supervisor agent output unavailable.)")

# User prompt for Gantt chart or detailed schedule
def print_gantt_chart(context):
    print("\n=== JSSP Gantt Chart/Table (Textual) ===")
    found = False
    for agent_name, output in context.items():
        print(f"\n{agent_name}:")
        if isinstance(output, str):
            print(output)
        elif isinstance(output, dict) and 'schedule' in output:
            found = True
            print("Time | Machine | Job | Step")
            for entry in output['schedule']:
                print(f"{entry.get('start', '?')}-{entry.get('end', '?')} | {entry.get('machine', '?')} | {entry.get('job', '?')} | {entry.get('step', '?')}")
        else:
            print("(No detailed schedule available. Ensure agent outputs include schedule details.)")
    if not found:
        print("\n(To enable Gantt chart, agent outputs should include a 'schedule' key with step timing and machine info.)")

show = input("\nShow Gantt chart or detailed JSSP schedule? (y/n): ").strip().lower()
if show == 'y':
    print_gantt_chart(context) 