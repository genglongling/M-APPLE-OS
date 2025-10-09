#!/usr/bin/env python3
"""
MAPLE Workflow Comparison Script
Tests 4 different workflow configurations with Claude-4
"""

import os
import time
import json
import sys
import logging
from datetime import datetime
from io import StringIO

# Set API keys for GPT-4o
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

# Full dataset suite from run_all_frameworks.sh
test_datasets = [
    # DMU datasets (16 datasets)
    "rcmax_20_15_5", "rcmax_20_15_8", "rcmax_20_20_7", "rcmax_20_20_8",
    "rcmax_30_15_5", "rcmax_30_15_4", "rcmax_30_20_9", "rcmax_30_20_8",
    "rcmax_40_15_10", "rcmax_40_15_8", "rcmax_40_20_6", "rcmax_40_20_2",
    "rcmax_50_15_2", "rcmax_50_15_4", "rcmax_50_20_6", "rcmax_50_20_9",
    # TA datasets (7 datasets)
    "TA01", "TA02", "TA51", "TA52", "TA61", "TA71", "TA72",
    # ABZSWVYN datasets (18 datasets)
    "abz07", "abz08", "abz09",
    "swv01", "swv02", "swv03", "swv04", "swv05", "swv06", "swv07", "swv08", "swv09", "swv10",
    "swv11", "swv12", "swv13", "swv14", "swv15",
    "yn01", "yn02", "yn03", "yn04"
]

# Workflow configurations
WORKFLOW_CONFIGS = {
    "full": {
        "name": "Full Workflow",
        "description": "Complete workflow with all tools",
        "use_validation": True,
        "use_repair": True,
        "use_optimization": True,
        "max_attempts": 3
    },
    "no_repair": {
        "name": "No Repair Tools",
        "description": "Full workflow without repair tools",
        "use_validation": True,
        "use_repair": False,
        "use_optimization": True,
        "max_attempts": 3
    },
    "no_validation": {
        "name": "No Validation Tools",
        "description": "Full workflow without validation tools",
        "use_validation": False,
        "use_repair": True,
        "use_optimization": True,
        "max_attempts": 3
    },
    "no_optimization": {
        "name": "No Optimization Tools",
        "description": "Full workflow without optimization tools",
        "use_validation": True,
        "use_repair": True,
        "use_optimization": False,
        "max_attempts": 3
    }
}

def run_dataset(dataset_name, category):
    """Run all 4 workflow configurations on a single dataset"""
    print(f"\n{'='*80}")
    print(f"Running MAPLE Workflow Comparison on {dataset_name} ({category})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Setup terminal output capture
    terminal_output = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Setup logging for this dataset
        os.makedirs("results_optimized(gpt-4o)", exist_ok=True)
        logger = setup_logging(f"{dataset_name}_workflow_comparison", log_dir="results_optimized(gpt-4o)")
        
        # Redirect stdout and stderr to capture terminal output
        sys.stdout = terminal_output
        sys.stderr = terminal_output
        
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
        
        # Extract machine names
        all_machine_indices = set()
        for job in jobs:
            for machine, _ in job['steps']:
                idx = int(machine.replace('Machine', ''))
                all_machine_indices.add(idx)
        machine_names = [f"Machine{idx}" for idx in sorted(all_machine_indices)]
        
        # Run all 4 workflow configurations
        results = {}
        
        for config_name, config in WORKFLOW_CONFIGS.items():
            print(f"\n--- Running {config['name']} ---")
            result = run_maple_workflow(dataset_name, jobs, machine_names, config_name, config)
            results[config_name] = result
            
            # Log results
            logger.info(f"\n=== {config['name']} Results ===")
            logger.info(f"Success: {result['success']}")
            logger.info(f"Makespan: {result['makespan']}")
            logger.info(f"Execution Time: {result['execution_time']:.2f}s")
            if 'error' in result:
                logger.info(f"Error: {result['error']}")
        
        # Save comprehensive results to JSON
        results_file = os.path.join("results_optimized(gpt-4o)", f"{dataset_name}_workflow_comparison.json")
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'category': category,
                'timestamp': datetime.now().isoformat(),
                'workflow_results': results
            }, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"WORKFLOW COMPARISON SUMMARY for {dataset_name}")
        print(f"{'='*60}")
        for config_name, result in results.items():
            status = "✅" if result['success'] else "❌"
            makespan = result['makespan'] if result['makespan'] else "N/A"
            time_str = f"{result['execution_time']:.2f}s"
            repair_info = ""
            if 'repair_iterations' in result and result['repair_iterations']:
                repair_count = len(result['repair_iterations'])
                repair_info = f" | Repairs: {repair_count}"
            print(f"{status} {WORKFLOW_CONFIGS[config_name]['name']:20} | Makespan: {makespan:>8} | Time: {time_str}{repair_info}")
        
        logger.info(f"\nCompleted workflow comparison for {dataset_name}")
        logger.info(f"{'='*50}\n")
        
        # Remove handlers to avoid duplicate logging
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Save terminal output to file
        terminal_output_file = os.path.join("results_optimized(gpt-4o)", f"{dataset_name}_terminal_output.log")
        with open(terminal_output_file, 'w') as f:
            f.write(terminal_output.getvalue())
        
        print(f"Successfully processed {dataset_name} with all workflow configurations")
        return True
        
    except Exception as e:
        # Restore original stdout/stderr even on error
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"An unexpected error occurred for {dataset_name}: {e}")
        return False

def run_maple_workflow(dataset_name, jobs, machine_names, config_name, config):
    """Run MAPLE with specific workflow configuration"""
    print(f"\n{'='*60}")
    print(f"Running MAPLE {config['name']} on {dataset_name}")
    print(f"Description: {config['description']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # Create agents for each job with GPT-4o
        agents = []
        for job in jobs:
            agent = JSSPAgent(
                name=f"{job['name']} Agent",
                backstory=f"Agent for {job['name']} scheduling using GPT-4o.",
                task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
                task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
                model_type="openai",
                model_name="gpt-4o"
            )
            agents.append(agent)
        

        # Create validation agent with GPT-4o
        validation_agent = JSSPValidationAgent(
            name="Validation Agent",
            backstory="Validation agent that checks schedule constraints using GPT-4o.",
            task_description="Validate schedules for job precedence, machine capacity, and makespan accuracy.",
            task_expected_output="Validation report with constraint satisfaction status.",
            model_type="openai"
        )

        # Create task specification with proper node structure
        task_spec = {
            'dataset_name': dataset_name,  # Add dataset name for JSON file lookup
            'jobs': jobs,
            'machine_names': machine_names,
            'workflow_config': config,
            'nodes': [
                {'agent': None, 'type': 'JSSPAgent'},  # No live agent, will load from JSON
                {'agent': None, 'type': 'JSSPValidationAgent'}  # No live agent, will use validation tools
            ],
            'edges': [
                {'from': 0, 'to': 1}   # Query agent to validation
            ]
        }

        # Initialize OptimizedMAPLE with workflow configuration
        from src.multi_agent.MAPLE_optimized import OptimizedMAPLE
        maple = OptimizedMAPLE(task_spec)
        
        # Run the workflow based on configuration
        start_time = time.time()
        
        # Run the OptimizedMAPLE with the specified configuration
        success = maple.run(with_rollback=True, validate=True)
        
        # Extract results from the executor context
        if success and maple.executor.best_schedule:
            result = {
                'success': True,
                'makespan': maple.executor.best_makespan,
                'schedule': maple.executor.best_schedule
            }
        else:
            result = {
                'success': False,
                'makespan': None,
                'schedule': []
            }
        
        # Extract repair iteration data if available
        repair_iterations = []
        if hasattr(maple.executor, 'context') and 'query_result' in maple.executor.context:
            query_result = maple.executor.context['query_result']
            if 'repair_iterations' in query_result:
                repair_iterations = query_result['repair_iterations']
                print(f"📊 Found {len(repair_iterations)} repair iterations")
                for i, iteration in enumerate(repair_iterations):
                    print(f"  Iteration {iteration['iteration']}: makespan={iteration['makespan']}, entries={iteration['schedule_entries_count']}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        return {
            'config_name': config_name,
            'config': config,
            'success': result['success'],
            'makespan': result['makespan'],
            'schedule': result['schedule'],
            'repair_iterations': repair_iterations,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in {config_name} workflow for {dataset_name}: {e}")
        return {
            'config_name': config_name,
            'config': config,
            'success': False,
            'makespan': None,
            'schedule': [],
            'repair_iterations': [],
            'execution_time': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    print("Starting MAPLE Workflow Comparison Test...")
    print(f"Total datasets to process: {len(test_datasets)}")
    print("🔄 Testing 4 workflow configurations:")
    for config_name, config in WORKFLOW_CONFIGS.items():
        print(f"  - {config['name']}: {config['description']}")
    
    successful_runs = 0
    failed_runs = 0
    
    for i, dataset in enumerate(test_datasets):
        print(f"\n--- Processing dataset {i+1}/{len(test_datasets)}: {dataset} ---")
        
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
        
        time.sleep(2)  # Delay between runs

    print(f"\nMAPLE Workflow Comparison Test Completed.")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total datasets: {len(test_datasets)}")
    print(f"Results saved in: results_optimized(gpt-4o)/")
