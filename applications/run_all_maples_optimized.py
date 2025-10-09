#!/usr/bin/env python3
"""
Optimized MAPLE Multi-Agent System Runner with 4 Workflow Configurations
Runs MAPLE with different tool combinations for comparison
"""

import os
import time
import json
from datetime import datetime
import sys

# Set API keys for Claude-4
os.environ["ANTHROPIC_API_KEY"] = "API_KEY_TO_REPLACE"
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

# Import MAPLE components
from src.multi_agent.MAPLE import MAPLE
from src.tools.validation_tools import ValidationTools
from src.tools.local_repair_tools import LocalRepairTools
from src.tools.optimization_tools import OptimizationTools

# Test with a few datasets first
test_datasets = [
    "rcmax_20_15_5", "swv01", "abz07", "TA01"
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

def run_maple_workflow(dataset_name, jobs, machine_names, config_name, config):
    """Run MAPLE with specific workflow configuration"""
    print(f"\n{'='*60}")
    print(f"Running MAPLE {config['name']} on {dataset_name}")
    print(f"Description: {config['description']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # Create agents for each job with Claude-4
        agents = []
        for job in jobs:
            agent = JSSPAgent(
                name=f"{job['name']} Agent",
                backstory=f"Agent for {job['name']} scheduling using Claude-4.",
                task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
                task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
                model_type="anthropic",
                model_name="claude-sonnet-4-20250514"
            )
            agents.append(agent)
        
        # Create supervisor agent with Claude-4
        supervisor_agent = SupervisorAgent(
            name="Supervisor Agent",
            backstory="Supervisor agent that coordinates all job schedules to find the minimum makespan solution using Claude-4.",
            task_description="""Find the minimum makespan schedule for all jobs while strictly following these rules:
1. Each job's steps must be completed in strict order (e.g., Job1's step 2 can only start after step 1 is completed).
2. Each machine can only process one job step at a time (e.g., if MachineA is processing Job1's step 1 from time 0-3, it cannot process any other job steps during that time).

The goal is to minimize the total completion time (makespan) while ensuring all jobs are completed and all constraints are satisfied.""",
            task_expected_output="A complete schedule with minimum makespan that satisfies all constraints.",
            model_type="anthropic",
            model_name="claude-sonnet-4-20250514"
        )

        # Create validation agent with Claude-4
        validation_agent = JSSPValidationAgent(
            name="Validation Agent",
            backstory="Validation agent that checks schedule constraints using Claude-4.",
            task_description="Validate schedules for job precedence, machine capacity, and makespan accuracy.",
            task_expected_output="Validation report with constraint satisfaction status.",
            model_type="anthropic",
            model_name="claude-sonnet-4-20250514"
        )

        # Create task specification
        task_spec = {
            'jobs': jobs,
            'machine_names': machine_names,
            'workflow_config': config,
            'agents': agents,
            'supervisor_agent': supervisor_agent,
            'validation_agent': validation_agent
        }

        # Initialize MAPLE with workflow configuration
        maple = MAPLE(task_spec)
        
        # Run the workflow based on configuration
        start_time = time.time()
        
        if config_name == "full":
            result = run_full_workflow(maple, config)
        elif config_name == "no_repair":
            result = run_no_repair_workflow(maple, config)
        elif config_name == "no_validation":
            result = run_no_validation_workflow(maple, config)
        elif config_name == "no_optimization":
            result = run_no_optimization_workflow(maple, config)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extract results
        makespan = result.get('makespan', None)
        schedule = result.get('schedule', [])
        success = result.get('success', False)
        
        return {
            'config_name': config_name,
            'config': config,
            'success': success,
            'makespan': makespan,
            'schedule': schedule,
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
            'execution_time': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def run_full_workflow(maple, config):
    """Run full workflow: Query -> Validation -> Repair -> Re-validation -> Optimization -> Final Check -> Supervisor"""
    print("🔄 Running Full Workflow...")
    
    # Step 1: LLM Query Agent generate_schedule()
    print("1️⃣ LLM Query Agent generating schedule...")
    query_result = maple.query_agent.generate_schedule()
    
    # Step 2: ValidationTools validate_schedule()
    if config['use_validation']:
        print("2️⃣ ValidationTools validating schedule...")
        validation_result = ValidationTools.validate_schedule(query_result)
        if not validation_result['valid']:
            print(f"❌ Initial validation failed: {validation_result['errors']}")
            
            # Step 3: RepairTools repair_schedule()
            if config['use_repair']:
                print("3️⃣ RepairTools repairing schedule...")
                repair_result = LocalRepairTools.fix_schedule(query_result, validation_result['errors'])
                
                # Step 4: ValidationTools revalidate_schedule()
                print("4️⃣ ValidationTools revalidating schedule...")
                revalidation_result = ValidationTools.validate_schedule(repair_result)
                if not revalidation_result['valid']:
                    print(f"❌ Revalidation failed: {revalidation_result['errors']}")
                    return {'success': False, 'makespan': None, 'schedule': []}
                query_result = repair_result
        else:
            print("✅ Initial validation passed")
    
    # Step 5: OptimizationTools optimize_schedule()
    if config['use_optimization']:
        print("5️⃣ OptimizationTools optimizing schedule...")
        optimization_result = OptimizationTools.run_optimization_schedule(query_result)
        query_result = optimization_result
    
    # Step 6: ValidationTools final_check()
    if config['use_validation']:
        print("6️⃣ ValidationTools final check...")
        final_validation = ValidationTools.validate_schedule(query_result)
        if not final_validation['valid']:
            print(f"❌ Final validation failed: {final_validation['errors']}")
            return {'success': False, 'makespan': None, 'schedule': []}
    
    # Step 7: Supervisor Agent select_best_schedule
    print("7️⃣ Supervisor Agent selecting best schedule...")
    supervisor_result = maple.supervisor_agent.select_best_schedule(query_result)
    
    return {
        'success': True,
        'makespan': supervisor_result.get('makespan'),
        'schedule': supervisor_result.get('schedule', [])
    }

def run_no_repair_workflow(maple, config):
    """Run workflow without repair tools: Query -> Validation -> Optimization -> Final Check -> Supervisor"""
    print("🔄 Running No Repair Workflow...")
    
    # Step 1: LLM Query Agent generate_schedule()
    print("1️⃣ LLM Query Agent generating schedule...")
    query_result = maple.query_agent.generate_schedule()
    
    # Step 2: ValidationTools validate_schedule()
    if config['use_validation']:
        print("2️⃣ ValidationTools validating schedule...")
        validation_result = ValidationTools.validate_schedule(query_result)
        if not validation_result['valid']:
            print(f"❌ Validation failed: {validation_result['errors']}")
            return {'success': False, 'makespan': None, 'schedule': []}
        print("✅ Validation passed")
    
    # Step 3: OptimizationTools optimize_schedule()
    if config['use_optimization']:
        print("3️⃣ OptimizationTools optimizing schedule...")
        optimization_result = OptimizationTools.run_optimization_schedule(query_result)
        query_result = optimization_result
    
    # Step 4: ValidationTools final_check()
    if config['use_validation']:
        print("4️⃣ ValidationTools final check...")
        final_validation = ValidationTools.validate_schedule(query_result)
        if not final_validation['valid']:
            print(f"❌ Final validation failed: {final_validation['errors']}")
            return {'success': False, 'makespan': None, 'schedule': []}
    
    # Step 5: Supervisor Agent select_best_schedule
    print("5️⃣ Supervisor Agent selecting best schedule...")
    supervisor_result = maple.supervisor_agent.select_best_schedule(query_result)
    
    return {
        'success': True,
        'makespan': supervisor_result.get('makespan'),
        'schedule': supervisor_result.get('schedule', [])
    }

def run_no_validation_workflow(maple, config):
    """Run workflow without validation tools: Query -> Repair -> Optimization -> Supervisor"""
    print("🔄 Running No Validation Workflow...")
    
    # Step 1: LLM Query Agent generate_schedule()
    print("1️⃣ LLM Query Agent generating schedule...")
    query_result = maple.query_agent.generate_schedule()
    
    # Step 2: RepairTools repair_schedule() (without validation)
    if config['use_repair']:
        print("2️⃣ RepairTools repairing schedule...")
        repair_result = LocalRepairTools.fix_schedule(query_result, [])
        query_result = repair_result
    
    # Step 3: OptimizationTools optimize_schedule()
    if config['use_optimization']:
        print("3️⃣ OptimizationTools optimizing schedule...")
        optimization_result = OptimizationTools.run_optimization_schedule(query_result)
        query_result = optimization_result
    
    # Step 4: Supervisor Agent select_best_schedule
    print("4️⃣ Supervisor Agent selecting best schedule...")
    supervisor_result = maple.supervisor_agent.select_best_schedule(query_result)
    
    return {
        'success': True,
        'makespan': supervisor_result.get('makespan'),
        'schedule': supervisor_result.get('schedule', [])
    }

def run_no_optimization_workflow(maple, config):
    """Run workflow without optimization tools: Query -> Validation -> Repair -> Re-validation -> Supervisor"""
    print("🔄 Running No Optimization Workflow...")
    
    # Step 1: LLM Query Agent generate_schedule()
    print("1️⃣ LLM Query Agent generating schedule...")
    query_result = maple.query_agent.generate_schedule()
    
    # Step 2: ValidationTools validate_schedule()
    if config['use_validation']:
        print("2️⃣ ValidationTools validating schedule...")
        validation_result = ValidationTools.validate_schedule(query_result)
        if not validation_result['valid']:
            print(f"❌ Initial validation failed: {validation_result['errors']}")
            
            # Step 3: RepairTools repair_schedule()
            if config['use_repair']:
                print("3️⃣ RepairTools repairing schedule...")
                repair_result = LocalRepairTools.fix_schedule(query_result, validation_result['errors'])
                
                # Step 4: ValidationTools revalidate_schedule()
                print("4️⃣ ValidationTools revalidating schedule...")
                revalidation_result = ValidationTools.validate_schedule(repair_result)
                if not revalidation_result['valid']:
                    print(f"❌ Revalidation failed: {revalidation_result['errors']}")
                    return {'success': False, 'makespan': None, 'schedule': []}
                query_result = repair_result
        else:
            print("✅ Initial validation passed")
    
    # Step 5: Supervisor Agent select_best_schedule
    print("5️⃣ Supervisor Agent selecting best schedule...")
    supervisor_result = maple.supervisor_agent.select_best_schedule(query_result)
    
    return {
        'success': True,
        'makespan': supervisor_result.get('makespan'),
        'schedule': supervisor_result.get('schedule', [])
    }
