import sys
import os
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
# import google.generativeai as genai  # Commented out - using OpenAI only
from anthropic import Anthropic
# from deepseek import Deepseek  # Commented out for now

# Set up project root and src path
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(dataset, log_dir="./results"):
    # Create results directory if it doesn't exist
    results_dir = log_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file = f"{log_dir}/{dataset}_maple.txt"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize different LLM clients
def get_llm_client(model_type="openai"):
    if model_type == "openai":
        return OpenAI()
    elif model_type == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_type == "google":
        # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # return genai
        raise ValueError("Google model not available - using OpenAI only")
    elif model_type == "deepseek":
        # return Deepseek(api_key=os.getenv("DEEPSEEK_API_KEY"))  # Commented out for now
        raise ValueError("Deepseek model not available in current setup")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# Universal dataset loader for DMU, TA, and ABZSWVYN formats
def load_dmu_dataset(filepath):
    jobs = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        
        # Check if it's a TA format (starts with "Nb of jobs")
        if lines[0].startswith('Nb of jobs'):
            # TA format - line 2 has: n_jobs n_machines time_seed machine_seed upper_bound lower_bound
            values = lines[1].split()
            n_jobs, n_machines = int(values[0]), int(values[1])
            
            # Find the "Times" section and get data after it
            data_start = 0
            for i, line in enumerate(lines):
                if line.strip() == 'Times':
                    data_start = i + 1
                    break
            
            for job_idx, line in enumerate(lines[data_start:data_start + n_jobs]):
                tokens = list(map(int, line.split()))
                steps = [(f"Machine{machine}", duration) for machine, duration in zip(tokens[::2], tokens[1::2])]
                jobs.append({'name': f'Job{job_idx+1}', 'steps': steps})
        else:
            # DMU/ABZSWVYN format
            n_jobs, n_machines = map(int, lines[0].split())
            for job_idx, line in enumerate(lines[1:]):
                tokens = list(map(int, line.split()))
                steps = [(f"Machine{machine}", duration) for machine, duration in zip(tokens[::2], tokens[1::2])]
                jobs.append({'name': f'Job{job_idx+1}', 'steps': steps})
    
    return jobs

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run MAPLE multi-agent JSSP with DMU dataset')
parser.add_argument('--dataset', type=str, default='rcmax_20_15_5', 
                    help='DMU dataset name (default: rcmax_20_15_5)')
parser.add_argument('--model', type=str, default='openai', 
                    choices=['google', 'openai', 'anthropic', 'deepseek'],
                    help='LLM model to use (default: openai)')
args = parser.parse_args()

# Load dataset (DMU, TA, or ABZSWVYN)
dataset_file = None
if args.dataset.startswith('rcmax_') or args.dataset.startswith('cscmax_'):
    dataset_file = os.path.join(project_root, 'applications', 'DMU', f'{args.dataset}.txt')
elif args.dataset.startswith('TA'):
    dataset_file = os.path.join(project_root, 'applications', 'TA', f'{args.dataset}.txt')
elif args.dataset.startswith('abz') or args.dataset.startswith('swv') or args.dataset.startswith('yn'):
    dataset_file = os.path.join(project_root, 'applications', 'abzswvyn', f'{args.dataset}.txt')
else:
    # Try DMU first, then TA, then ABZSWVYN
    for dataset_type, subdir in [('DMU', 'DMU'), ('TA', 'TA'), ('ABZSWVYN', 'abzswvyn')]:
        test_file = os.path.join(project_root, 'applications', subdir, f'{args.dataset}.txt')
        if os.path.exists(test_file):
            dataset_file = test_file
            break
    
    if dataset_file is None:
        raise FileNotFoundError(f"Dataset {args.dataset} not found in DMU, TA, or ABZSWVYN directories")

print(f"Loading dataset: {args.dataset}")
jobs = load_dmu_dataset(dataset_file)
print(f"Loaded {len(jobs)} jobs")

# Get machine names from the dataset
all_machine_indices = set()
for job in jobs:
    for machine, _ in job['steps']:
        idx = int(machine.replace('Machine', ''))
        all_machine_indices.add(idx)
machine_names = [f"Machine{idx}" for idx in sorted(all_machine_indices)]
print(f"Using {len(machine_names)} machines: {machine_names}")

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
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai", model_name=None):
        # Pass model_name to the base Agent class if provided
        if model_name:
            super().__init__(name, backstory, task_description, task_expected_output, model_type=model_type, llm=model_name)
        else:
            super().__init__(name, backstory, task_description, task_expected_output, model_type=model_type)
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
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai", model_name=None, logger=None):
        # Pass model_name to the base Agent class if provided
        if model_name:
            super().__init__(name, backstory, task_description, task_expected_output, model_type=model_type, llm=model_name)
        else:
            super().__init__(name, backstory, task_description, task_expected_output, model_type=model_type)
        self.client = get_llm_client(model_type)
        self.model_type = model_type
        self.logger = logger

    def run(self):
        # Get schedules from context (new workflow) or job agents (old workflow)
        all_schedules = []
        
        # Check if we have a pre-generated schedule in context (new workflow)
        if hasattr(self, 'context') and isinstance(self.context, dict):
            if 'query_result' in self.context and 'schedule' in self.context['query_result']:
                all_schedules = self.context['query_result']['schedule']
                pre_generated_makespan = self.context['query_result'].get('makespan', None)
                print(f"📊 Using pre-generated schedule with {len(all_schedules)} entries")
                if pre_generated_makespan:
                    print(f"📊 Using pre-generated makespan: {pre_generated_makespan}")
                    print(f"✅ Skipping optimization - using Claude-4 result directly")
                    return {'schedule': all_schedules, 'makespan': pre_generated_makespan}
            elif 'schedule' in self.context:
                all_schedules = self.context['schedule']
                print(f"📊 Using context schedule with {len(all_schedules)} entries")
        
        # Fallback to old workflow (job agents)
        if not all_schedules and hasattr(self, 'dependencies'):
            for agent in self.dependencies:
                if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
                    all_schedules.extend(agent.context['schedule'])
        
        if not all_schedules:
            print("Warning: No schedules found from job agents")
            return {'schedule': []}

        # IMPROVED ALGORITHM: Try multiple scheduling strategies and pick the best VALID one
        best_schedule = None
        best_makespan = float('inf')
        
        # Strategy 1: Original greedy approach
        schedule1 = self._greedy_schedule()
        if schedule1 and self._validate_schedule(schedule1):
            makespan1 = max(entry.get('end', 0) for entry in schedule1)
            if makespan1 < best_makespan:
                best_schedule = schedule1
                best_makespan = makespan1
                print(f"✅ Greedy strategy: Valid schedule with makespan {makespan1}")
        else:
            print("❌ Greedy strategy: Invalid schedule")
        
        # Strategy 2: Shortest Processing Time (SPT) first
        schedule2 = self._spt_schedule()
        if schedule2 and self._validate_schedule(schedule2):
            makespan2 = max(entry.get('end', 0) for entry in schedule2)
            if makespan2 < best_makespan:
                best_schedule = schedule2
                best_makespan = makespan2
                print(f"✅ SPT strategy: Valid schedule with makespan {makespan2}")
        else:
            print("❌ SPT strategy: Invalid schedule")
        
        # Strategy 3: Longest Processing Time (LPT) first
        schedule3 = self._lpt_schedule()
        if schedule3 and self._validate_schedule(schedule3):
            makespan3 = max(entry.get('end', 0) for entry in schedule3)
            if makespan3 < best_makespan:
                best_schedule = schedule3
                best_makespan = makespan3
                print(f"✅ LPT strategy: Valid schedule with makespan {makespan3}")
        else:
            print("❌ LPT strategy: Invalid schedule")
        
        # Strategy 4: Random order (try a few random permutations)
        for i in range(3):
            schedule4 = self._random_schedule()
            if schedule4 and self._validate_schedule(schedule4):
                makespan4 = max(entry.get('end', 0) for entry in schedule4)
                if makespan4 < best_makespan:
                    best_schedule = schedule4
                    best_makespan = makespan4
                    print(f"✅ Random strategy {i+1}: Valid schedule with makespan {makespan4}")
            else:
                print(f"❌ Random strategy {i+1}: Invalid schedule")
        
        new_schedule = best_schedule if best_schedule else []
        
        # Sort the new schedule by start time
        new_schedule.sort(key=lambda x: (x.get('start', 0), x.get('machine', ''), x.get('job', '')))
        
        # Calculate makespan
        makespan = max(entry.get('end', 0) for entry in new_schedule)
        
        # Calculate upper bound
        job_sums = {}
        machine_sums = {name: 0 for name in machine_names}
        
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
    
    def _greedy_schedule(self):
        """Original greedy scheduling algorithm"""
        machine_availability = {name: 0 for name in machine_names}
        job_completion = {job['name']: 0 for job in jobs}
        job_step = {job['name']: 0 for job in jobs}
        completed_jobs = set()
        new_schedule = []

        while len(completed_jobs) < len(jobs):
            for job in jobs:
                job_name = job['name']
                if job_name in completed_jobs:
                    continue

                current_step = job_step[job_name]
                if current_step >= len(job['steps']):
                    completed_jobs.add(job_name)
                    continue

                machine, duration = job['steps'][current_step]
                start_time = max(job_completion[job_name], machine_availability[machine])

                # Check for conflicts
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

                end_time = start_time + duration
                new_schedule.append({
                    'job': job_name,
                    'step': current_step + 1,
                    'machine': machine,
                    'start': start_time,
                    'end': end_time,
                    'precedence': f"After {job_name} Step {current_step}" if current_step > 0 else None
                })

                machine_availability[machine] = end_time
                job_completion[job_name] = end_time
                job_step[job_name] += 1

        return new_schedule
    
    def _spt_schedule(self):
        """Shortest Processing Time first"""
        import random
        # Sort jobs by total processing time (shortest first)
        sorted_jobs = sorted(jobs, key=lambda j: sum(duration for _, duration in j['steps']))
        return self._schedule_with_job_order(sorted_jobs)
    
    def _lpt_schedule(self):
        """Longest Processing Time first"""
        # Sort jobs by total processing time (longest first)
        sorted_jobs = sorted(jobs, key=lambda j: sum(duration for _, duration in j['steps']), reverse=True)
        return self._schedule_with_job_order(sorted_jobs)
    
    def _random_schedule(self):
        """Random job order"""
        import random
        sorted_jobs = jobs.copy()
        random.shuffle(sorted_jobs)
        return self._schedule_with_job_order(sorted_jobs)
    
    def _schedule_with_job_order(self, job_order):
        """Schedule with a specific job order"""
        machine_availability = {name: 0 for name in machine_names}
        job_completion = {job['name']: 0 for job in jobs}
        job_step = {job['name']: 0 for job in jobs}
        completed_jobs = set()
        new_schedule = []

        while len(completed_jobs) < len(jobs):
            for job in job_order:
                job_name = job['name']
                if job_name in completed_jobs:
                    continue

                current_step = job_step[job_name]
                if current_step >= len(job['steps']):
                    completed_jobs.add(job_name)
                    continue

                machine, duration = job['steps'][current_step]
                start_time = max(job_completion[job_name], machine_availability[machine])

                # Check for conflicts
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

                end_time = start_time + duration
                new_schedule.append({
                    'job': job_name,
                    'step': current_step + 1,
                    'machine': machine,
                    'start': start_time,
                    'end': end_time,
                    'precedence': f"After {job_name} Step {current_step}" if current_step > 0 else None
                })

                machine_availability[machine] = end_time
                job_completion[job_name] = end_time
                job_step[job_name] += 1

        return new_schedule
    
    def _validate_schedule(self, schedule):
        """Validate a schedule for JSSP constraints"""
        if not schedule:
            return False
        
        # 1. FIRST CHECK: Verify schedule matches the dataset
        # Check that all jobs in schedule exist in the dataset
        dataset_job_names = {job['name'] for job in jobs}
        schedule_job_names = {entry.get('job') for entry in schedule}
        
        # All jobs in schedule must exist in dataset
        if not schedule_job_names.issubset(dataset_job_names):
            print(f"❌ Validation failed: Schedule contains jobs not in dataset")
            return False
        
        # Check that all machines in schedule exist in the dataset
        dataset_machine_names = set(machine_names)
        schedule_machine_names = {entry.get('machine') for entry in schedule}
        
        # All machines in schedule must exist in dataset
        if not schedule_machine_names.issubset(dataset_machine_names):
            print(f"❌ Validation failed: Schedule contains machines not in dataset")
            return False
        
        # Check that operations match dataset requirements
        for entry in schedule:
            job_name = entry.get('job')
            step = entry.get('step', 0)
            machine = entry.get('machine')
            
            # Find the corresponding job in dataset
            dataset_job = None
            for job in jobs:
                if job['name'] == job_name:
                    dataset_job = job
                    break
            
            if not dataset_job:
                print(f"❌ Validation failed: Job {job_name} not found in dataset")
                return False
            
            # Check step number is valid (1-based indexing)
            if step < 1 or step > len(dataset_job['steps']):
                print(f"❌ Validation failed: Job {job_name} step {step} out of range (1-{len(dataset_job['steps'])})")
                return False
            
            # Check machine matches the required machine for this step
            required_machine, required_duration = dataset_job['steps'][step-1]
            if machine != required_machine:
                print(f"❌ Validation failed: Job {job_name} step {step} uses machine {machine}, but dataset requires {required_machine}")
                return False
            
            # Check duration matches
            actual_duration = entry.get('end', 0) - entry.get('start', 0)
            if actual_duration != required_duration:
                print(f"❌ Validation failed: Job {job_name} step {step} duration {actual_duration} doesn't match required {required_duration}")
                return False
        
        # 2. Check machine constraints (no overlapping operations)
        machine_schedules = {name: [] for name in machine_names}
        for entry in schedule:
            machine = entry.get('machine')
            if machine in machine_schedules:
                machine_schedules[machine].append(entry)
        
        for machine, machine_schedule in machine_schedules.items():
            machine_schedule.sort(key=lambda x: x.get('start', 0))
            for i in range(len(machine_schedule)-1):
                if machine_schedule[i].get('end', 0) > machine_schedule[i+1].get('start', 0):
                    print(f"❌ Validation failed: Machine {machine} has overlapping operations")
                    return False  # Overlap detected
        
        # 3. Check job precedence constraints
        job_steps = {}
        for entry in schedule:
            job = entry.get('job')
            if job not in job_steps:
                job_steps[job] = []
            job_steps[job].append(entry)
        
        for job, steps in job_steps.items():
            steps.sort(key=lambda x: x.get('step', 0))
            for i in range(len(steps)-1):
                if steps[i].get('end', 0) > steps[i+1].get('start', 0):
                    print(f"❌ Validation failed: Job {job} has precedence violation between steps {steps[i].get('step')} and {steps[i+1].get('step')}")
                    return False  # Precedence violation
        
        # 4. Check job completion (all jobs must have all their steps)
        for job in jobs:
            job_name = job['name']
            if job_name not in job_steps:
                print(f"❌ Validation failed: Job {job_name} not scheduled")
                return False  # Job not scheduled
            if len(job_steps[job_name]) != len(job['steps']):
                print(f"❌ Validation failed: Job {job_name} has {len(job_steps[job_name])} steps scheduled, but dataset requires {len(job['steps'])}")
                return False  # Job not fully scheduled
        
        print("✅ Schedule validation passed: All constraints satisfied")
        return True

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
        machine_schedules = {name: [] for name in machine_names}
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

if __name__ == "__main__":
    # Setup logging for this dataset
    logger = setup_logging(args.dataset)
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing dataset: {args.dataset}")
    logger.info(f"{'='*50}\n")
    logger.info(f"=== MAPLE Multi-Agent JSSP with {args.dataset} ===")
    logger.info(f"Using {args.model} model")
    logger.info(f"Dataset: {dataset_file}")
    logger.info(f"Jobs: {len(jobs)}, Machines: {len(machine_names)}")
    logger.info("=" * 50)

    # Create agents
    agents = []
    for job in jobs:
        agent = JSSPAgent(
            name=f"{job['name']} Agent",
            backstory=f"Agent for {job['name']} scheduling.",
            task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
            task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
            model_type=args.model
        )
        agents.append(agent)

    # Add validation agent
    validation_agent = JSSPValidationAgent(
        name="JSSP Validation Agent",
        backstory="Validates JSSP schedules for constraint violations.",
        task_description="Check all schedules for machine constraints, precedence constraints, and makespan validity.",
        task_expected_output="Validation results with any detected violations.",
        model_type=args.model
    )

    # Add supervisor agent
    supervisor_agent = SupervisorAgent(
        name="Supervisor Agent",
        backstory="Aggregates all job schedules and produces the overall JSSP schedule.",
        task_description="Combine all job agent schedules into a single overall JSSP schedule.",
        task_expected_output="Overall JSSP schedule as a table.",
        model_type=args.model,
        logger=logger
    )

    agents.extend([supervisor_agent, validation_agent])

    # Create task specification
    task_spec = {
        'nodes': [
            {'agent': agent, 'dependencies': []} for agent in agents[:-2]  # Job agents
        ] + [
            {'agent': supervisor_agent, 'dependencies': [agent.name for agent in agents[:-2]]},
            {'agent': validation_agent, 'dependencies': [supervisor_agent.name]},
        ],
        'edges': [],
        'jobs': jobs,
        'disruptions': [],  # No disruptions for now
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
        
        # Log results
        logger.info(f"\n=== Results for {args.dataset} ===")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info("| Job  | Step | Machine  | Time Slot | Precedence Constraints      |")
        logger.info("|------|------|----------|-----------|----------------------------|")
        for entry in all_schedules:
            job = entry.get('job', '?')
            step = entry.get('step', '?')
            machine = entry.get('machine', '?')
            start = entry.get('start', '?')
            end = entry.get('end', '?')
            prec = entry.get('precedence', 'None')
            logger.info(f"| {job} | {step} | {machine} | {start}-{end} | {prec} |")
        
        # Calculate makespan
        makespan = max(entry.get('end', 0) for entry in all_schedules)
        
        # Calculate upper bound and gap
        # Upper bound = max(total job time, max machine total time)
        job_sums = {}
        machine_sums = {name: 0 for name in machine_names}
        
        # Calculate job sums (total time for each job)
        for job in jobs:
            job_sum = sum(duration for _, duration in job['steps'])
            job_sums[job['name']] = job_sum
        
        # Calculate machine sums (total work required on each machine from original dataset)
        for job in jobs:
            for machine, duration in job['steps']:
                if machine in machine_sums:
                    machine_sums[machine] += duration
        
        # Upper bound is the maximum of:
        # 1. Longest job time (no job can finish before its total duration)
        # 2. Most loaded machine time (no schedule can finish before the most loaded machine)
        ub = max(max(job_sums.values()), max(machine_sums.values()))
        gap = makespan - ub
        
        logger.info(f"\nBest static makespan for {args.dataset}: {makespan}")
        logger.info(f"Upper Bound (UB): {ub}")
        logger.info(f"Gap to UB: {gap}")
        
        # Also print to console
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
        print(f"Upper Bound (UB): {ub}")
        print(f"Gap to UB: {gap}")
    else:
        logger.info(f"\nNo detailed schedules found for {args.dataset}")
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

    # show = input("\nShow Gantt chart or detailed JSSP schedule? (y/n): ").strip().lower()
    # if show == 'y':
    #     print_gantt_chart(context) 