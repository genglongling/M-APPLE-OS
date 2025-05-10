import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
# from deepseek import Deepseek
from datetime import datetime
import logging

# Set up project root and src path
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(dataset):
    # Create results directory if it doesn't exist
    results_dir = "./results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler
    log_file = f"./results/{dataset}_dmu.txt"
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

# After imports, add the list of datasets
DATASETS = [
    "rcmax_20_15_5", #Dmu03_
    "rcmax_20_15_8", #Dmu04_
    "rcmax_20_20_7", #Dmu08_
    "rcmax_20_20_8", #Dmu09_
    "rcmax_30_15_5", #Dmu13_
    "rcmax_30_15_4", #Dmu14_
    "rcmax_30_20_9", #Dmu18_
    "rcmax_30_20_8", #Dmu19_
    "rcmax_40_15_10",#Dmu23_
    "rcmax_40_15_8", #Dmu24_
    "rcmax_40_20_6", #Dmu28_
    "rcmax_40_20_2", #Dmu29_
    "rcmax_50_15_2", #Dmu33_
    "rcmax_50_15_4", #Dmu34_
    "rcmax_50_20_6", #Dmu38_
    "rcmax_50_20_9" #Dmu39_
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
# Algorithm 1: Reschedule all jobs together
# class SupervisorAgent(Agent):
#     def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai"):
#         super().__init__(name, backstory, task_description, task_expected_output)
#         self.client = get_llm_client(model_type)
#         self.model_type = model_type

#     def run(self):
#         # Get schedules from all job agents
#         all_schedules = []
#         for agent in self.dependencies:
#             if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
#                 all_schedules.extend(agent.context['schedule'])
        
#         if not all_schedules:
#             print("Warning: No schedules found from job agents")
#             return {'schedule': []}

#         # Initialize machine availability times
#         machine_availability = {name: 0 for name in machine_names}
#         # Track completion time for each job
#         job_completion = {job['name']: 0 for job in jobs}
#         # Track which step each job is on
#         job_step = {job['name']: 0 for job in jobs}
#         # Track which jobs are completed
#         completed_jobs = set()
#         # New schedule
#         new_schedule = []

#         while len(completed_jobs) < len(jobs):
#             # For each job that's not completed
#             for job in jobs:
#                 job_name = job['name']
#                 if job_name in completed_jobs:
#                     continue

#                 current_step = job_step[job_name]
#                 if current_step >= len(job['steps']):
#                     completed_jobs.add(job_name)
#                     continue

#                 machine, duration = job['steps'][current_step]
                
#                 # Calculate earliest possible start time
#                 # Must be after:
#                 # 1. Previous step completion
#                 # 2. Machine availability
#                 start_time = max(
#                     job_completion[job_name],  # Previous step completion
#                     machine_availability[machine]  # Machine availability
#                 )

#                 # Check for conflicts with existing operations
#                 while True:
#                     conflict = False
#                     for existing_op in new_schedule:
#                         if (existing_op['machine'] == machine and
#                             not (start_time >= existing_op['end'] or 
#                                  start_time + duration <= existing_op['start'])):
#                             conflict = True
#                             start_time = existing_op['end']
#                             break
#                     if not conflict:
#                         break

#                 # Schedule this operation
#                 end_time = start_time + duration
#                 new_schedule.append({
#                     'job': job_name,
#                     'step': current_step + 1,
#                     'machine': machine,
#                     'start': start_time,
#                     'end': end_time,
#                     'precedence': f"After {job_name} Step {current_step}" if current_step > 0 else None
#                 })

#                 # Update tracking variables
#                 machine_availability[machine] = end_time
#                 job_completion[job_name] = end_time
#                 job_step[job_name] += 1

#         # Sort the new schedule by start time
#         new_schedule.sort(key=lambda x: (x.get('start', 0), x.get('machine', ''), x.get('job', '')))
        
#         # Calculate makespan
#         makespan = max(entry.get('end', 0) for entry in new_schedule)
        
#         # Calculate upper bound
#         job_sums = {}
#         machine_sums = {name: 0 for name in machine_names}
        
#         for job in jobs:
#             job_sum = sum(duration for _, duration in job['steps'])
#             job_sums[job['name']] = job_sum
            
#         for entry in new_schedule:
#             machine = entry.get('machine')
#             duration = entry.get('end', 0) - entry.get('start', 0)
#             if machine in machine_sums:
#                 machine_sums[machine] += duration
        
#         ub = max(max(job_sums.values()), max(machine_sums.values()))
        
#         print(f"\nUpper Bound (UB): {ub}")
#         print(f"Current Makespan: {makespan}")
#         print(f"Gap to UB: {makespan - ub}")
        
#         # Store in agent's context
#         self.context = {'schedule': new_schedule}
#         return {'schedule': new_schedule}

# class JSSPValidationAgent(Agent):
#     def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai"):
#         super().__init__(name, backstory, task_description, task_expected_output)
#         self.client = get_llm_client(model_type)
#         self.model_type = model_type

#     def run(self):
#         # Get schedule from supervisor agent
#         all_schedules = []
#         for agent in self.dependencies:
#             if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
#                 all_schedules.extend(agent.context['schedule'])
        
#         if not all_schedules:
#             return {
#                 'valid': False,
#                 'errors': ['No schedules found from supervisor agent'],
#                 'makespan': None
#             }
        
#         # Initialize validation results
#         errors = []
        
#         # 1. Check machine constraints (no overlapping operations)
#         machine_schedules = {name: [] for name in machine_names}
#         for entry in all_schedules:
#             machine = entry.get('machine')
#             if machine in machine_schedules:
#                 machine_schedules[machine].append(entry)
        
#         for machine, schedule in machine_schedules.items():
#             schedule.sort(key=lambda x: x.get('start', 0))
#             for i in range(len(schedule)-1):
#                 if schedule[i].get('end', 0) > schedule[i+1].get('start', 0):
#                     errors.append(f"Overlap detected on {machine}: {schedule[i].get('job')} Step {schedule[i].get('step')} and {schedule[i+1].get('job')} Step {schedule[i+1].get('step')}")
        
#         # 2. Check job precedence constraints
#         job_steps = {}
#         for entry in all_schedules:
#             job = entry.get('job')
#             if job not in job_steps:
#                 job_steps[job] = []
#             job_steps[job].append(entry)
        
#         for job, steps in job_steps.items():
#             steps.sort(key=lambda x: x.get('step', 0))
#             for i in range(len(steps)-1):
#                 if steps[i].get('end', 0) > steps[i+1].get('start', 0):
#                     errors.append(f"Precedence violation in {job}: Step {steps[i].get('step')} ends after Step {steps[i+1].get('step')} starts")
        
#         # 3. Check job completion
#         for job in jobs:
#             job_name = job['name']
#             if job_name not in job_steps or len(job_steps[job_name]) != len(job['steps']):
#                 errors.append(f"Incomplete schedule for {job_name}")
        
#         # 4. Check makespan
#         makespan = max(entry.get('end', 0) for entry in all_schedules)
#         if makespan < 10:  # UB is 10
#             errors.append(f"Makespan {makespan} is below theoretical UB of 10")
        
#         # Print validation results
#         print("\n=== Validation Results ===")
#         if errors:
#             print("❌ Validation failed with the following errors:")
#             for error in errors:
#                 print(f"- {error}")
#         else:
#             print("✅ Schedule is valid!")
#         print(f"Final Makespan: {makespan}")
        
#         return {
#             'valid': len(errors) == 0,
#             'errors': errors,
#             'makespan': makespan
#         }

# agents = []
# for job in jobs:
#     agent = JSSPAgent(
#         name=f"{job['name']} Agent",
#         backstory=f"Agent for {job['name']} scheduling.",
#         task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
#         task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
#         model_type="openai"  # or other model type
#     )
#     agents.append(agent)

# algorithm 1: tabu search
class SupervisorAgent(Agent):
    """
    Supervisor agent that coordinates all job schedules to find the minimum makespan solution.
    Implements multiple scheduling algorithms and selects the best one.
    """
    def __init__(self, name, backstory, task_description, task_expected_output, model_type="openai"):
        super().__init__(name, backstory, task_description, task_expected_output)
        self.client = get_llm_client(model_type)
        self.model_type = model_type
        self.has_run = False
        self.dataset_name = os.path.splitext(os.path.basename(DMU_FILE))[0]

    def schedule_random(self, jobs, machine_names):
        """Random scheduling algorithm"""
        import random
        job_order = jobs.copy()
        random.shuffle(job_order)
        return self.schedule_from_order(job_order, machine_names)

    def schedule_lpt(self, jobs, machine_names):
        """Longest Processing Time first"""
        # Sort jobs by total processing time (descending)
        job_order = sorted(jobs, key=lambda x: sum(duration for _, duration in x['steps']), reverse=True)
        return self.schedule_from_order(job_order, machine_names)

    def schedule_spt(self, jobs, machine_names):
        from heapq import heappush, heappop

        machine_avail = {m: 0 for m in machine_names}
        job_next_step = {job['name']: 0 for job in jobs}
        job_avail = {job['name']: 0 for job in jobs}
        schedule = []
        completed_jobs = set()
        job_map = {job['name']: job for job in jobs}
        num_jobs = len(jobs)

        while len(completed_jobs) < num_jobs:
            candidates = []

            for jname in job_next_step:
                if jname in completed_jobs:
                    continue
                step_idx = job_next_step[jname]
                job = job_map[jname]
                if step_idx < len(job['steps']):
                    machine, duration = job['steps'][step_idx]
                    ready_time = max(machine_avail[machine], job_avail[jname])
                    heappush(candidates, (duration, ready_time, jname, step_idx, machine))

            if not candidates:
                raise RuntimeError("Deadlock: no operations available, check for data inconsistency.")

            duration, start_time, jname, step_idx, machine = heappop(candidates)
            end_time = start_time + duration

            schedule.append({
                'job': jname,
                'step': step_idx + 1,
                'machine': machine,
                'start': start_time,
                'end': end_time,
                'precedence': f"After {jname} Step {step_idx}" if step_idx > 0 else None
            })

            # Update states
            machine_avail[machine] = end_time
            job_avail[jname] = end_time
            job_next_step[jname] += 1
            if job_next_step[jname] >= len(job_map[jname]['steps']):
                completed_jobs.add(jname)

        return schedule

    
    def schedule_stpt(self, jobs, machine_names):
        """Shortest Total Processing Time first"""
        # Sort jobs by total processing time (ascending)
        job_order = sorted(jobs, key=lambda x: sum(duration for _, duration in x['steps']))
        return self.schedule_from_order(job_order, machine_names)

    def schedule_mpsr(self, jobs, machine_names):
        """Most Processing Steps Remaining"""
        # Sort jobs by number of remaining steps (descending)
        job_order = sorted(jobs, key=lambda x: len(x['steps']), reverse=True)
        return self.schedule_from_order(job_order, machine_names)

    def schedule_drl_liu(self, jobs, machine_names):
        """DRL-Liu algorithm (simplified version)"""
        # Sort jobs by machine load and processing time
        machine_loads = {name: 0 for name in machine_names}
        for job in jobs:
            for machine, duration in job['steps']:
                machine_loads[machine] += duration
        
        # include step-level machine imbalance
        def job_score(job):
            return sum(duration / (machine_loads[machine] + 1e-5) for machine, duration in job['steps'])

        
        job_order = sorted(jobs, key=job_score, reverse=True)
        return self.schedule_from_order(job_order, machine_names)

    def schedule_gp(self, jobs, machine_names):
        """Genetic Programming approach (simplified)"""
        # Sort jobs by processing time and machine utilization
        machine_utilization = {name: 0 for name in machine_names}
        for job in jobs:
            for machine, duration in job['steps']:
                machine_utilization[machine] += duration
        
        # include machine-specific cost weighting/bottleneck
        def job_score(job):
            bottleneck = max(machine_utilization[machine] for machine, _ in job['steps'])
            return sum(duration for _, duration in job['steps']) * bottleneck

        
        job_order = sorted(jobs, key=job_score)
        return self.schedule_from_order(job_order, machine_names)

    def schedule_gep(self, jobs, machine_names):
        """Gene Expression Programming approach"""
        # Calculate machine loads and job characteristics
        machine_loads = {name: 0 for name in machine_names}
        job_characteristics = {}
        
        # First pass: calculate machine loads and job characteristics
        for job in jobs:
            total_time = 0
            machine_usage = {name: 0 for name in machine_names}
            step_times = []
            
            for machine, duration in job['steps']:
                total_time += duration
                machine_usage[machine] += duration
                step_times.append(duration)
                machine_loads[machine] += duration
            
            # Calculate job characteristics
            job_characteristics[job['name']] = {
                'total_time': total_time,
                'machine_usage': machine_usage,
                'step_times': step_times,
                'avg_step_time': total_time / len(step_times),
                'max_step_time': max(step_times),
                'min_step_time': min(step_times)
            }
        
        # Calculate machine utilization ratios
        total_machine_time = sum(machine_loads.values())
        machine_utilization = {
            name: load / total_machine_time 
            for name, load in machine_loads.items()
        }
        
        def job_score(job):
            """Complex scoring function using multiple factors"""
            char = job_characteristics[job['name']]
            
            # Factor 1: Machine load balance
            load_balance = sum(
                machine_utilization[machine] * usage 
                for machine, usage in char['machine_usage'].items()
            )
            
            # Factor 2: Step time variation
            time_variation = (char['max_step_time'] - char['min_step_time']) / char['avg_step_time']
            
            # Factor 3: Machine utilization impact
            utilization_impact = sum(
                usage / machine_loads[machine]
                for machine, usage in char['machine_usage'].items()
                if machine_loads[machine] > 0
            )
            
            # Factor 4: Processing time efficiency
            time_efficiency = char['total_time'] / (len(char['step_times']) * char['avg_step_time'])
            
            # Combine factors with weights
            score = (
                0.3 * load_balance +      # Machine load balance
                0.2 * time_variation +    # Step time variation
                0.3 * utilization_impact + # Machine utilization
                0.2 * time_efficiency     # Processing efficiency
            )
            
            return score
        
        # Sort jobs by the complex scoring function
        job_order = sorted(jobs, key=job_score)
        
        # Apply local search to improve the initial solution
        best_schedule = self.schedule_from_order(job_order, machine_names)
        best_makespan = max(entry.get('end', 0) for entry in best_schedule)
        
        # Try swapping adjacent jobs to improve the solution
        for i in range(len(job_order) - 1):
            # Create a new order by swapping adjacent jobs
            new_order = job_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            
            # Evaluate the new order
            new_schedule = self.schedule_from_order(new_order, machine_names)
            new_makespan = max(entry.get('end', 0) for entry in new_schedule)
            
            # Keep the better solution
            if new_makespan < best_makespan:
                best_schedule = new_schedule
                best_makespan = new_makespan
                job_order = new_order
        
        return best_schedule

    def schedule_tabu(self, jobs, machine_names):
        """Tabu Search algorithm"""
        from collections import deque
        
        def schedule_from_order(job_order):
            return self.schedule_from_order(job_order, machine_names)
        
        # Initial solution: greedy order
        current_order = jobs[:]
        current_schedule = schedule_from_order(current_order)
        current_makespan = max(entry.get('end', 0) for entry in current_schedule)
        best_order = list(current_order)
        best_schedule = list(current_schedule)
        best_makespan = current_makespan
        
        tabu_list = deque(maxlen=10)  # Tabu tenure
        tabu_list.append(tuple(job['name'] for job in current_order))
        
        # Track previous makespan for convergence checking
        previous_makespan = float('inf')
        no_improvement_count = 0
        max_no_improvement = 5
        
        for it in range(50):  # Max iterations
            neighbors = []
            # Generate neighbors by swapping every pair of jobs
            for i in range(len(current_order)):
                for j in range(i+1, len(current_order)):
                    neighbor_order = list(current_order)
                    neighbor_order[i], neighbor_order[j] = neighbor_order[j], neighbor_order[i]
                    order_tuple = tuple(job['name'] for job in neighbor_order)
                    if order_tuple in tabu_list:
                        continue
                    neighbor_schedule = schedule_from_order(neighbor_order)
                    neighbor_makespan = max(entry.get('end', 0) for entry in neighbor_schedule)
                    neighbors.append((neighbor_makespan, neighbor_order, neighbor_schedule, order_tuple))
            
            if not neighbors:
                break
            
            # Choose the best neighbor
            neighbors.sort(key=lambda x: x[0])
            best_neighbor = neighbors[0]
            current_makespan, current_order, current_schedule, order_tuple = best_neighbor
            tabu_list.append(order_tuple)
            
            # Check for convergence
            if current_makespan == previous_makespan:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    break
            else:
                no_improvement_count = 0
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
                best_schedule = list(current_schedule)
            
            previous_makespan = current_makespan
        
        return best_schedule

    def schedule_from_order(self, job_order, machine_names):
        """Helper function to create a schedule from a job order"""
        machine_availability = {name: 0 for name in machine_names}
        job_completion = {job['name']: 0 for job in job_order}
        job_step = {job['name']: 0 for job in job_order}
        completed_jobs = set()
        schedule = []
        
        while len(completed_jobs) < len(job_order):
            for job in job_order:
                job_name = job['name']
                if job_name in completed_jobs:
                    continue
                
                current_step = job_step[job_name]
                if current_step >= len(job['steps']):
                    completed_jobs.add(job_name)
                    continue
                
                machine, duration = job['steps'][current_step]
                start_time = max(
                    job_completion[job_name],
                    machine_availability[machine]
                )
                
                # Check for conflicts
                while True:
                    conflict = False
                    for existing_op in schedule:
                        if (existing_op['machine'] == machine and
                            not (start_time >= existing_op['end'] or 
                                 start_time + duration <= existing_op['start'])):
                            conflict = True
                            start_time = existing_op['end']
                            break
                    if not conflict:
                        break
                
                end_time = start_time + duration
                schedule.append({
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
                
                if job_step[job_name] >= len(job['steps']):
                    completed_jobs.add(job_name)
        
        return schedule

    def run(self):
        if self.has_run and hasattr(self, 'context') and 'schedule' in self.context:
            print("[SupervisorAgent] Using previously computed schedule")
            return self.context

        # Create results directory if it doesn't exist
        results_dir = "./results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created results directory: {results_dir}")

        # Create output file
        output_file = f"./results/{self.dataset_name}_dmu.txt"
        with open(output_file, 'w') as f:
            f.write(f"=== JSSP Schedule Results for {self.dataset_name} ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            print(f"=== JSSP Schedule Results for {self.dataset_name} ===")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Get all job schedules from dependencies
        all_schedules = []
        for agent in self.dependencies:
            if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'schedule' in agent.context:
                all_schedules.extend(agent.context['schedule'])

        if not all_schedules:
            print("Warning: No schedules found from job agents")
            return {'schedule': []}

        # Run all algorithms and track their results
        algorithms = {
            'Random': self.schedule_random,
            'LPT': self.schedule_lpt,
            'SPT': self.schedule_spt,
            'STPT': self.schedule_stpt,
            'MPSR': self.schedule_mpsr,
            'DRL-Liu': self.schedule_drl_liu,
            'GP': self.schedule_gp,
            'GEP': self.schedule_gep,
            'Tabu': self.schedule_tabu
        }

        results = {}
        for name, algorithm in algorithms.items():
            print(f"\nRunning {name} algorithm...")
            schedule = algorithm(jobs, machine_names)
            makespan = max(entry.get('end', 0) for entry in schedule)
            results[name] = {
                'schedule': schedule,
                'makespan': makespan
            }
            print(f"{name} makespan: {makespan}")

        # Find the best algorithm
        best_algorithm = min(results.items(), key=lambda x: x[1]['makespan'])
        best_name, best_result = best_algorithm
        
        print(f"\nBest algorithm: {best_name} with makespan {best_result['makespan']}")
        
        # Calculate upper bound
        job_sums = {}
        machine_sums = {name: 0 for name in machine_names}
        for job in jobs:
            job_sum = sum(duration for _, duration in job['steps'])
            job_sums[job['name']] = job_sum
        for entry in best_result['schedule']:
            machine = entry.get('machine')
            duration = entry.get('end', 0) - entry.get('start', 0)
            if machine in machine_sums:
                machine_sums[machine] += duration
        ub = max(max(job_sums.values()), max(machine_sums.values()))
        
        # Write results to file and print to console
        results_text = [
            "\n=== Final Results ===",
            f"Best Algorithm: {best_name}",
            f"Minimum Makespan found: {best_result['makespan']}",
            f"Upper Bound (UB): {ub}",
            f"Gap to UB: {best_result['makespan'] - ub}\n",
            "=== Algorithm Comparison ==="
        ]
        
        for name, result in results.items():
            results_text.append(f"{name}: {result['makespan']}")
        
        results_text.append("\n=== Detailed Schedule ===")

        with open(output_file, 'a') as f:
            for line in results_text:
                print(line)
                f.write(line + "\n")
            
            # Write detailed schedule
            for entry in best_result['schedule']:
                schedule_line = f"Job: {entry['job']}, Step: {entry['step']}, Machine: {entry['machine']}, Start: {entry['start']}, End: {entry['end']}"
                print(schedule_line)
                f.write(schedule_line + "\n")

        self.has_run = True
        self.context = {'schedule': best_result['schedule']}
        return {'schedule': best_result['schedule']}

# Modify the main execution section at the bottom of the file
if __name__ == "__main__":
    # Process each dataset
    for dataset in DATASETS:
        # Setup logging for this dataset
        logger = setup_logging(dataset)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*50}\n")

        # Update DMU file path for current dataset
        DMU_FILE = os.path.join(project_root, 'applications', 'DMU', f'{dataset}.txt')
        
        # Load jobs for current dataset
        jobs = load_dmu_dataset(DMU_FILE)
        
        # After loading jobs
        all_machine_indices = set()
        for job in jobs:
            for machine, _ in job['steps']:
                idx = int(machine.replace('Machine', ''))
                all_machine_indices.add(idx)
        machine_names = [f"Machine{idx}" for idx in sorted(all_machine_indices)]

        # Create agents for each job
        agents = []
        for job in jobs:
            agent = JSSPAgent(
                name=f"{job['name']} Agent",
                backstory=f"Agent for {job['name']} scheduling.",
                task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
                task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
                model_type="openai"
            )
            agents.append(agent)
        
        # Add validation agent
        # validation_agent = JSSPValidationAgent(
        #     name="JSSP Validation Agent",
        #     backstory="Validates JSSP schedules for constraint violations.",
        #     task_description="Check all schedules for machine constraints, precedence constraints, and makespan validity.",
        #     task_expected_output="Validation results with any detected violations.",
        #     model_type="openai"
        # )

        # Add supervisor agent
        supervisor_agent = SupervisorAgent(
            name="Supervisor Agent",
            backstory="Supervisor agent that coordinates all job schedules to find the minimum makespan solution.",
            task_description="""Find the minimum makespan schedule for all jobs while strictly following these rules:
1. Each job's steps must be completed in strict order (e.g., Job1's step 2 can only start after step 1 is completed).
2. Each machine can only process one job step at a time (e.g., if MachineA is processing Job1's step 1 from time 0-3, it cannot process any other job steps during that time).

The goal is to minimize the total completion time (makespan) while ensuring all jobs are completed and all constraints are satisfied.""",
            task_expected_output="A complete schedule with minimum makespan that satisfies all constraints.",
            model_type="openai"
        )

        agents.extend([supervisor_agent])

        # Only job agents as initial nodes
        nodes = [{'agent': agent, 'dependencies': []} for agent in agents if isinstance(agent, JSSPAgent)]

        # Supervisor depends on all job agents
        nodes.append({'agent': supervisor_agent, 'dependencies': [agent.name for agent in agents if isinstance(agent, JSSPAgent)]})

        # Validation agent depends on supervisor
        # nodes.append({'agent': validation_agent, 'dependencies': [supervisor_agent.name]})

        task_spec = {
            'nodes': nodes,
            'edges': [],
            'jobs': jobs,
            'disruptions': [],
            'rules': [
                'Each job must perform its steps strictly in order.',
                'Each machine can only handle one operation at a time.',
                'No two operations use the same machine at the same time.'
            ]
        }

        # Initialize MAPLE
        maple = MAPLE(task_spec)

        # Run MAPLE
        maple.run(with_rollback=True, validate=True)

        # Extract and print overall schedule from supervisor agent
        context = maple.executor.context
        supervisor_output = context.get(supervisor_agent.name, {})

        logger.info(f"=== Results for {dataset} ===")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if isinstance(supervisor_output, dict) and 'schedule' in supervisor_output:
            all_schedules = supervisor_output['schedule']
            # Sort by start time for overall schedule
            all_schedules.sort(key=lambda x: (x.get('start', 0), x.get('machine', ''), x.get('job', '')))
            
            # Write detailed schedule
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
            
            # Calculate and write makespan
            makespan = max(entry.get('end', 0) for entry in all_schedules)
            logger.info(f"\nBest static makespan for {dataset}: {makespan}")

            # Write Gantt chart
            logger.info("\n=== JSSP Gantt Chart/Table (Textual) ===")
            logger.info("Time | Machine | Job | Step")
            for entry in all_schedules:
                logger.info(f"{entry.get('start', '?')}-{entry.get('end', '?')} | {entry.get('machine', '?')} | {entry.get('job', '?')} | {entry.get('step', '?')}")
        else:
            logger.info(f"\nNo detailed schedules found for {dataset}")

        logger.info(f"\nCompleted processing {dataset}")
        logger.info(f"{'='*50}\n")

        # Remove handlers to avoid duplicate logging in next iteration
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler) 