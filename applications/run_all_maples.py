#!/usr/bin/env python3
"""
Comprehensive MAPLE Multi-Agent System Runner
Runs MAPLE on all datasets from DMU, TA, and ABZSWVYN categories
Processes datasets sequentially within the same Python process
"""

import os
import time
from datetime import datetime
import sys

# Set API keys
os.environ["OPENAI_API_KEY"] = "API_KEY_TO_REPLACE"

# Import the necessary functions from multiagent-jssp1-dmu.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import importlib.util
spec = importlib.util.spec_from_file_location("multiagent_jssp1_dmu", "applications/multiagent-jssp1-dmu.py")
multiagent_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiagent_module)

# Import the functions from the module
setup_logging = multiagent_module.setup_logging
get_llm_client = multiagent_module.get_llm_client
load_dmu_dataset = multiagent_module.load_dmu_dataset
JSSPAgent = multiagent_module.JSSPAgent
SupervisorAgent = multiagent_module.SupervisorAgent
JSSPValidationAgent = multiagent_module.JSSPValidationAgent
MAPLE = multiagent_module.MAPLE

# DMU datasets (16 selected datasets from generate_files.py)
dmu_datasets = [
    "rcmax_20_15_5", "rcmax_20_15_8", "rcmax_20_20_7", "rcmax_20_20_8",
    "rcmax_30_15_5", "rcmax_30_15_4", "rcmax_30_20_9", "rcmax_30_20_8",
    "rcmax_40_15_10", "rcmax_40_15_8", "rcmax_40_20_6", "rcmax_40_20_2",
    "rcmax_50_15_2", "rcmax_50_15_4", "rcmax_50_20_6", "rcmax_50_20_9"
]

#     # rcmax datasets
    # "rcmax_20_15_1", "rcmax_20_15_4", "rcmax_20_15_5", "rcmax_20_15_8", "rcmax_20_15_10",
    # "rcmax_20_20_4", "rcmax_20_20_5", "rcmax_20_20_6", "rcmax_20_20_7", "rcmax_20_20_8",
    # "rcmax_30_15_1", "rcmax_30_15_4", "rcmax_30_15_5", "rcmax_30_15_9", "rcmax_30_15_10",
    # "rcmax_30_20_2", "rcmax_30_20_7", "rcmax_30_20_8", "rcmax_30_20_9", "rcmax_30_20_10",
    # "rcmax_40_15_1", "rcmax_40_15_4", "rcmax_40_15_8", "rcmax_40_15_10", "rcmax_40_15_12",
    # "rcmax_40_20_2", "rcmax_40_20_6", "rcmax_40_20_8", "rcmax_40_20_10", "rcmax_40_20_12",
    # "rcmax_50_15_2", "rcmax_50_15_4", "rcmax_50_15_6", "rcmax_50_15_8", "rcmax_50_15_10",
    # "rcmax_50_20_2", "rcmax_50_20_4", "rcmax_50_20_6", "rcmax_50_20_8", "rcmax_50_20_10",
    # # cscmax datasets
    # "cscmax_20_15_1", "cscmax_20_15_4", "cscmax_20_15_5", "cscmax_20_15_8", "cscmax_20_15_10",
    # "cscmax_20_20_4", "cscmax_20_20_5", "cscmax_20_20_6", "cscmax_20_20_7", "cscmax_20_20_8",
    # "cscmax_30_15_1", "cscmax_30_15_4", "cscmax_30_15_5", "cscmax_30_15_9", "cscmax_30_15_10",
    # "cscmax_30_20_2", "cscmax_30_20_7", "cscmax_30_20_8", "cscmax_30_20_9", "cscmax_30_20_10",
    # "cscmax_40_15_1", "cscmax_40_15_4", "cscmax_40_15_8", "cscmax_40_15_10", "cscmax_40_15_12",
    # "cscmax_40_20_2", "cscmax_40_20_6", "cscmax_40_20_8", "cscmax_40_20_10", "cscmax_40_20_12",
    # "cscmax_50_15_2", "cscmax_50_15_4", "cscmax_50_15_6", "cscmax_50_15_8", "cscmax_50_15_10",
    # "cscmax_50_20_2", "cscmax_50_20_4", "cscmax_50_20_6", "cscmax_50_20_8", "cscmax_50_20_10"
# TA datasets (only available ones)
ta_datasets = [
    "TA01", "TA02", "TA51", "TA52", "TA61", "TA71", "TA72"
]

# ABZSWVYN datasets (only available ones)
abzswvyn_datasets = [
    # ABZ datasets
    "abz07", "abz08", "abz09",
    # SWV datasets
    "swv01", "swv02", "swv03", "swv04", "swv05", "swv06", "swv07", "swv08", "swv09", "swv10",
    "swv11", "swv12", "swv13", "swv14", "swv15",
    # YN datasets
    "yn01", "yn02", "yn03", "yn04"
]

# Combine all datasets
all_datasets = dmu_datasets + ta_datasets + abzswvyn_datasets

def run_dataset(dataset_name, category):
    """Run MAPLE on a single dataset"""
    print(f"\n{'='*60}")
    print(f"Running MAPLE on {dataset_name} ({category})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # Setup logging for this dataset
        logger = setup_logging(dataset_name)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*50}\n")

        # Determine dataset file path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_file = None
        
        if dataset_name.startswith('rcmax_') or dataset_name.startswith('cscmax_'):
            dataset_file = os.path.join(project_root, 'applications', 'DMU', f'{dataset_name}.txt')
        elif dataset_name.startswith('TA'):
            dataset_file = os.path.join(project_root, 'applications', 'TA', f'{dataset_name}.txt')
        elif dataset_name.startswith('abz') or dataset_name.startswith('swv') or dataset_name.startswith('yn'):
            dataset_file = os.path.join(project_root, 'applications', 'abzswvyn', f'{dataset_name}.txt')
        else:
            # Try DMU first, then TA, then ABZSWVYN
            for dataset_type, subdir in [('DMU', 'DMU'), ('TA', 'TA'), ('ABZSWVYN', 'abzswvyn')]:
                potential_path = os.path.join(project_root, 'applications', subdir, f'{dataset_name}.txt')
                if os.path.exists(potential_path):
                    dataset_file = potential_path
                    break
        
        if not dataset_file or not os.path.exists(dataset_file):
            logger.info(f"Dataset file not found for {dataset_name}")
            return False
        
        print(f"Loading dataset: {dataset_name} from {dataset_file}")
        jobs = load_dmu_dataset(dataset_file)
        print(f"Loaded {len(jobs)} jobs")
        
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

        logger.info(f"=== Results for {dataset_name} ===")
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
            logger.info(f"\nBest static makespan for {dataset_name}: {makespan}")

            # Calculate Upper Bound (UB) and Gap
            total_job_times = {job['name']: sum(duration for _, duration in job['steps']) for job in jobs}
            total_machine_times = {machine: 0 for machine in machine_names}
            for job in jobs:
                for machine, duration in job['steps']:
                    total_machine_times[machine] += duration

            ub = max(max(total_job_times.values()), max(total_machine_times.values()))
            gap = makespan - ub

            logger.info(f"Upper Bound (UB): {ub}")
            logger.info(f"Gap to UB: {gap}")

            # Write Gantt chart
            logger.info("\n=== JSSP Gantt Chart/Table (Textual) ===")
            logger.info("Time | Machine | Job | Step")
            for entry in all_schedules:
                logger.info(f"{entry.get('start', '?')}-{entry.get('end', '?')} | {entry.get('machine', '?')} | {entry.get('job', '?')} | {entry.get('step', '?')}")
        else:
            logger.info(f"\nNo detailed schedules found for {dataset_name}")

        logger.info(f"\nCompleted processing {dataset_name}")
        logger.info(f"{'='*50}\n")

        # Remove handlers to avoid duplicate logging in next iteration
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        print(f"Successfully processed {dataset_name}")
        return True
        
    except Exception as e:
        print(f"An unexpected error occurred for {dataset_name}: {e}")
        return False

if __name__ == "__main__":
    print("Starting comprehensive MAPLE multi-agent system test...")
    print(f"Total datasets to process: {len(all_datasets)}")
    
    successful_runs = 0
    failed_runs = 0
    
    for i, dataset in enumerate(all_datasets):
        print(f"\n--- Processing dataset {i+1}/{len(all_datasets)}: {dataset} ---")
        
        # Determine category
        if dataset.startswith('rcmax_') or dataset.startswith('cscmax_'):
            category = "DMU"
        elif dataset.startswith('TA'):
            category = "TA"
        elif dataset.startswith('abz') or dataset.startswith('swv') or dataset.startswith('yn'):
            category = "ABZSWVYN"
        else:
            category = "Unknown"
        
        success = run_dataset(dataset, category)
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
        
        time.sleep(1)  # Small delay between runs

    print(f"\nComprehensive MAPLE multi-agent system test completed.")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total datasets: {len(all_datasets)}")