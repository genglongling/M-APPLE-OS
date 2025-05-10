import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
# from deepseek import Deepseek

# Set up project root and src path
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# Load environment variables
load_dotenv()

# Initialize different LLM clients
def get_llm_client(model_type="google"):
    if model_type == "openai":
        return OpenAI()
    elif model_type == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_type == "google":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai
    elif model_type == "deepseek":
        return Deepseek(api_key=os.getenv("DEEPSEEK_API_KEY"))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# DMU dataset loader
def load_dmu_dataset(filepath):
    jobs = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        n_jobs, n_machines = map(int, lines[0].split())
        for job_idx, line in enumerate(lines[1:]):
            tokens = list(map(int, line.split()))
            steps = [(f"Machine{machine}", duration) for machine, duration in zip(tokens[::2], tokens[1::2])]
            jobs.append({'name': f'Job{job_idx+1}', 'steps': steps})
    return jobs

# Load jobs from DMU dataset
DMU_FILE = os.path.join(project_root, 'applications', 'DMU', 'rcmax_50_20_9.txt')
jobs = load_dmu_dataset(DMU_FILE)

# After loading jobs
all_machine_indices = set()
for job in jobs:
    for machine, _ in job['steps']:
        idx = int(machine.replace('Machine', ''))
        all_machine_indices.add(idx)
machine_names = [f"Machine{idx}" for idx in sorted(all_machine_indices)]

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
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="google"):
        super().__init__(
            name=name,
            backstory=backstory,
            task_description=task_description,
            task_expected_output=task_expected_output
        )
        self.client = get_llm_client(model_type)
        self.model_type = model_type

    def run(self):
        job_idx = int(self.name.split('Job')[1][0]) - 1
        job = jobs[job_idx]
        
        # Initialize schedule
        schedule = []
        current_time = 0
        
        # Track machine availability
        machine_availability = {name: 0 for name in machine_names}
        
        for step_idx, (machine, duration) in enumerate(job['steps']):
            # Find the earliest possible start time considering:
            # 1. Machine availability
            # 2. Previous step completion
            # 3. No overlapping operations on the same machine
            start_time = max(
                machine_availability[machine],
                current_time
            )
            
            # Check for conflicts with other operations
            while True:
                conflict = False
                for existing_schedule in schedule:
                    if (existing_schedule['machine'] == machine and
                        not (start_time >= existing_schedule['end'] or 
                             start_time + duration <= existing_schedule['start'])):
                        conflict = True
                        start_time = existing_schedule['end']
                        break
                if not conflict:
                    break
            
            # Update machine availability
            machine_availability[machine] = start_time + duration
            
            # Update current time for next step
            current_time = start_time + duration
            
            schedule.append({
                'job': job['name'],
                'step': step_idx + 1,
                'machine': machine,
                'start': start_time,
                'end': start_time + duration,
                'precedence': f"After {job['name']} Step {step_idx}" if step_idx > 0 else None
            })
        
        # Store schedule in agent's context
        self.context = {'schedule': schedule}
        return {'schedule': schedule}

# Add a final Supervisor Agent
class SupervisorAgent(Agent):
    """
    SupervisorAgent: Aggregates all job agent schedules.
    Objective: Find the schedule with the minimum possible makespan.
    (Note: This baseline implementation does not perform optimization.)
    """
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="google"):
        super().__init__(
            name=name,
            backstory=backstory,
            task_description=task_description,
            task_expected_output=task_expected_output
        )
        self.client = get_llm_client(model_type)
        self.model_type = model_type

    def run(self):
        # Aggregate all job agent schedules from context
        all_schedules = []
        for agent in self.dependencies:
            if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
                all_schedules.extend(agent.context['schedule'])

        if not all_schedules:
            print("Warning: No schedules found from job agents")
            return {'schedule': []}

        # Sort by start time
        all_schedules.sort(key=lambda x: (x.get('start', 0), x.get('machine', ''), x.get('job', '')))

        # Calculate makespan
        makespan = max(entry.get('end', 0) for entry in all_schedules)

        # Calculate upper bound (UB) from job definitions (theoretical UB)
        job_sums = {}
        machine_sums = {name: 0 for name in machine_names}
        for job in jobs:
            job_sum = sum(duration for _, duration in job['steps'])
            job_sums[job['name']] = job_sum
            for machine, duration in job['steps']:
                if machine in machine_sums:
                    machine_sums[machine] += duration
        ub = max(max(job_sums.values()), max(machine_sums.values()))

        print(f"\n[SupervisorAgent] (Baseline) Objective: Find minimum makespan.")
        print(f"Current Makespan: {makespan}")
        print(f"Upper Bound (UB): {ub}")
        print(f"Gap to UB: {makespan - ub}")

        self.context = {'schedule': all_schedules}
        return {'schedule': all_schedules}

# Create agents for each job
agents = []
for job in jobs:
    agent = JSSPAgent(
        name=f"{job['name']} Agent",
        backstory=f"Agent for {job['name']} scheduling.",
        task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
        task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
        model_type="google"  # or your preferred model
    )
    agents.append(agent)

# Add supervisor agent
supervisor_agent = SupervisorAgent(
    name="Supervisor Agent",
    backstory="Aggregates all job schedules and produces the overall minimum makespan JSSP schedule.",
    task_description="Combine all job agent schedules into a single overall minimum makespan JSSP schedule.",
    task_expected_output="Overall minimum makespan JSSP schedule as a table.",
    model_type="google"
)
agents.append(supervisor_agent)

# Only job agents as initial nodes
nodes = [{'agent': agent, 'dependencies': []} for agent in agents if isinstance(agent, JSSPAgent)]

# Supervisor depends on all job agents
nodes.append({'agent': supervisor_agent, 'dependencies': [agent.name for agent in agents if isinstance(agent, JSSPAgent)]})

task_spec = {
    'nodes': nodes,
    'edges': [],
    'jobs': jobs,
    'disruptions': [
        # You may want to update this for dynamic machine names
        # {'machine': 'MachineA', 'unavailable': [(4, 6)]}
    ],
    'rules': [
        'Each job must perform its steps strictly in order.',
        'Each machine can only handle one operation at a time.',
        'No two operations use the same machine at the same time.'
    ]
}

# Initialize MAPLE (no dynamic_adaptation argument in original MAPLE.py)
maple = MAPLE(task_spec)

# Run MAPLE
maple.run(with_rollback=True, validate=True)

# After MAPLE run, set context and re-run supervisor agent
context = maple.executor.context
supervisor_agent.context = context
context[supervisor_agent.name] = supervisor_agent.run()

# Extract and print overall schedule from supervisor agent
context = maple.executor.context
supervisor_output = context.get(supervisor_agent.name, {})

if isinstance(supervisor_output, dict) and 'schedule' in supervisor_output:
    all_schedules = supervisor_output['schedule']
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
    # Calculate makespan
    makespan = max(entry.get('end', 0) for entry in all_schedules)
    print(f"\nBest static makespan: {makespan}")
else:
    print("\n(No detailed schedules found from supervisor agent. Supervisor agent output unavailable.)")

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