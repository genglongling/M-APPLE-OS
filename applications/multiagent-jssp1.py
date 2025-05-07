import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
from deepseek import Deepseek

# Set up project root and src path
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# Load environment variables
load_dotenv()

# Initialize different LLM clients
def get_llm_client(model_type="openai"):
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
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai"):
        super().__init__(name, backstory, task_description, task_expected_output)
        self.client = get_llm_client(model_type)
        self.model_type = model_type

    def run(self):
        job_idx = int(self.name.split('Job')[1][0]) - 1
        job = jobs[job_idx]
        
        # Initialize schedule
        schedule = []
        current_time = 0
        
        # Track machine availability
        machine_availability = {'MachineA': 0, 'MachineB': 0, 'MachineC': 0}
        
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
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai"):
        super().__init__(name, backstory, task_description, task_expected_output)
        self.client = get_llm_client(model_type)
        self.model_type = model_type

    def run(self):
        # Get schedules from all job agents
        all_schedules = []
        for agent in self.dependencies:
            if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
                all_schedules.extend(agent.context['schedule'])
        
        if not all_schedules:
            print("Warning: No schedules found from job agents")
            return {'schedule': []}

        # Algorithm 1: Reschedule all jobs together
        # Initialize machine availability times
        machine_availability = {'MachineA': 0, 'MachineB': 0, 'MachineC': 0}
        # Track completion time for each job
        job_completion = {job['name']: 0 for job in jobs}
        # Track which step each job is on
        job_step = {job['name']: 0 for job in jobs}
        # Track which jobs are completed
        completed_jobs = set()
        # New schedule
        new_schedule = []

        while len(completed_jobs) < len(jobs):
            # For each job that's not completed
            for job in jobs:
                job_name = job['name']
                if job_name in completed_jobs:
                    continue

                current_step = job_step[job_name]
                if current_step >= len(job['steps']):
                    completed_jobs.add(job_name)
                    continue

                machine, duration = job['steps'][current_step]
                
                # Calculate earliest possible start time
                # Must be after:
                # 1. Previous step completion
                # 2. Machine availability
                start_time = max(
                    job_completion[job_name],  # Previous step completion
                    machine_availability[machine]  # Machine availability
                )

                # Check for conflicts with existing operations
                while True:
                    conflict = False
                    for existing_op in new_schedule:
                        if (existing_op['machine'] == machine and
                            not (start_time >= existing_op['end'] or 
                                 start_time + duration <= existing_op['start'])):
                            conflict = True
                            start_time = existing_op['end']
                            break
                    if not conflict:
                        break

                # Schedule this operation
                end_time = start_time + duration
                new_schedule.append({
                    'job': job_name,
                    'step': current_step + 1,
                    'machine': machine,
                    'start': start_time,
                    'end': end_time,
                    'precedence': f"After {job_name} Step {current_step}" if current_step > 0 else None
                })

                # Update tracking variables
                machine_availability[machine] = end_time
                job_completion[job_name] = end_time
                job_step[job_name] += 1

        # Sort the new schedule by start time
        new_schedule.sort(key=lambda x: (x.get('start', 0), x.get('machine', ''), x.get('job', '')))
        
        # Calculate makespan
        makespan = max(entry.get('end', 0) for entry in new_schedule)
        
        # Calculate upper bound
        job_sums = {}
        machine_sums = {'MachineA': 0, 'MachineB': 0, 'MachineC': 0}
        
        for job in jobs:
            job_sum = sum(duration for _, duration in job['steps'])
            job_sums[job['name']] = job_sum
            
        for entry in new_schedule:
            machine = entry.get('machine')
            duration = entry.get('end', 0) - entry.get('start', 0)
            if machine in machine_sums:
                machine_sums[machine] += duration
        
        ub = max(max(job_sums.values()), max(machine_sums.values()))
        
        print(f"\nUpper Bound (UB): {ub}")
        print(f"Current Makespan: {makespan}")
        print(f"Gap to UB: {makespan - ub}")
        
        # Store in agent's context
        self.context = {'schedule': new_schedule}
        return {'schedule': new_schedule}

class JSSPValidationAgent(Agent):
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai"):
        super().__init__(name, backstory, task_description, task_expected_output)
        self.client = get_llm_client(model_type)
        self.model_type = model_type

    def run(self):
        # Get schedule from supervisor agent
        all_schedules = []
        for agent in self.dependencies:
            if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
                all_schedules.extend(agent.context['schedule'])
        
        if not all_schedules:
            return {
                'valid': False,
                'errors': ['No schedules found from supervisor agent'],
                'makespan': None
            }
        
        # Initialize validation results
        errors = []
        
        # 1. Check machine constraints (no overlapping operations)
        machine_schedules = {'MachineA': [], 'MachineB': [], 'MachineC': []}
        for entry in all_schedules:
            machine = entry.get('machine')
            if machine in machine_schedules:
                machine_schedules[machine].append(entry)
        
        for machine, schedule in machine_schedules.items():
            schedule.sort(key=lambda x: x.get('start', 0))
            for i in range(len(schedule)-1):
                if schedule[i].get('end', 0) > schedule[i+1].get('start', 0):
                    errors.append(f"Overlap detected on {machine}: {schedule[i].get('job')} Step {schedule[i].get('step')} and {schedule[i+1].get('job')} Step {schedule[i+1].get('step')}")
        
        # 2. Check job precedence constraints
        job_steps = {}
        for entry in all_schedules:
            job = entry.get('job')
            if job not in job_steps:
                job_steps[job] = []
            job_steps[job].append(entry)
        
        for job, steps in job_steps.items():
            steps.sort(key=lambda x: x.get('step', 0))
            for i in range(len(steps)-1):
                if steps[i].get('end', 0) > steps[i+1].get('start', 0):
                    errors.append(f"Precedence violation in {job}: Step {steps[i].get('step')} ends after Step {steps[i+1].get('step')} starts")
        
        # 3. Check job completion
        for job in jobs:
            job_name = job['name']
            if job_name not in job_steps or len(job_steps[job_name]) != len(job['steps']):
                errors.append(f"Incomplete schedule for {job_name}")
        
        # 4. Check makespan
        makespan = max(entry.get('end', 0) for entry in all_schedules)
        if makespan < 10:  # UB is 10
            errors.append(f"Makespan {makespan} is below theoretical UB of 10")
        
        # Print validation results
        print("\n=== Validation Results ===")
        if errors:
            print("❌ Validation failed with the following errors:")
            for error in errors:
                print(f"- {error}")
        else:
            print("✅ Schedule is valid!")
        print(f"Final Makespan: {makespan}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'makespan': makespan
        }

agents = []
for job in jobs:
    agent = JSSPAgent(
        name=f"{job['name']} Agent",
        backstory=f"Agent for {job['name']} scheduling.",
        task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
        task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
        model_type="openai"  # Can be changed to "anthropic", "google", or "deepseek"
    )
    agents.append(agent)

# Add validation agent
validation_agent = JSSPValidationAgent(
    name="JSSP Validation Agent",
    backstory="Validates JSSP schedules for constraint violations.",
    task_description="Check all schedules for machine constraints, precedence constraints, and makespan validity.",
    task_expected_output="Validation results with any detected violations.",
    model_type="openai"  # Can be changed to "anthropic", "google", or "deepseek"
)

# Add supervisor agent
supervisor_agent = SupervisorAgent(
    name="Supervisor Agent",
    backstory="Aggregates all job schedules and produces the overall JSSP schedule.",
    task_description="Combine all job agent schedules into a single overall JSSP schedule.",
    task_expected_output="Overall JSSP schedule as a table.",
    model_type="openai"  # Can be changed to "anthropic", "google", or "deepseek"
)

agents.extend([supervisor_agent, validation_agent])

# Disruption event: MachineA unavailable from t=4 to t=6
# We'll encode this as part of the task_spec
task_spec = {
    'nodes': [
        {'agent': agents[0], 'dependencies': []},
        {'agent': agents[1], 'dependencies': []},
        {'agent': agents[2], 'dependencies': []},
        {'agent': supervisor_agent, 'dependencies': [agents[0].name, agents[1].name, agents[2].name]},
        {'agent': validation_agent, 'dependencies': [supervisor_agent.name]},
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