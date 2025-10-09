#!/usr/bin/env python3
"""
JSSP Framework Comparison System
Compares MAPLE, AutoGen, CrewAI, OpenAI Swarm, and LangGraph on Job Shop Scheduling Problems
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import MAPLE with proper path handling
import sys
sys.path.append(os.path.join(project_root, 'src'))

# Dynamic import for MAPLE
import importlib.util
maple_spec = importlib.util.spec_from_file_location("MAPLE_module", os.path.join(project_root, "src", "multi_agent", "MAPLE.py"))
maple_module = importlib.util.module_from_spec(maple_spec)
maple_spec.loader.exec_module(maple_module)
MAPLE = maple_module.MAPLE

# Dynamic import for multiagent-jssp1-dmu due to hyphen in filename
import importlib.util
jssp_dmu_spec = importlib.util.spec_from_file_location("multiagent_jssp1_dmu", "applications/multiagent-jssp1-dmu.py")
jssp_dmu_module = importlib.util.module_from_spec(jssp_dmu_spec)

# Temporarily replace sys.argv to prevent argument parsing in imported module
original_argv = sys.argv
sys.argv = ['multiagent-jssp1-dmu.py']  # Minimal argv to prevent argument parsing

try:
    jssp_dmu_spec.loader.exec_module(jssp_dmu_module)
finally:
    sys.argv = original_argv  # Restore original argv

load_dmu_dataset = jssp_dmu_module.load_dmu_dataset
JSSPAgent = jssp_dmu_module.JSSPAgent
SupervisorAgent = jssp_dmu_module.SupervisorAgent
get_llm_client = jssp_dmu_module.get_llm_client
setup_logging = jssp_dmu_module.setup_logging

# Make get_llm_client available globally for ReactAgent
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.llm_client import get_llm_client as utils_get_llm_client

# Override the utils.llm_client.get_llm_client with our version
import utils.llm_client
utils.llm_client.get_llm_client = get_llm_client

# Import framework implementations from agent_frameworks_jssp (using real implementations)
try:
    from agent_frameworks_jssp.autogen_multi_agent.router import run_autogen_agents
except ImportError:
    run_autogen_agents = None

try:
    from agent_frameworks_jssp.crewai_multi_agent.router import run_crewai
except ImportError:
    run_crewai = None

try:
    from agent_frameworks_jssp.openai_swarm_agent.router import run_swarm_agents
except ImportError:
    run_swarm_agents = None

try:
    from agent_frameworks_jssp.langgraph.router import run_agent
except ImportError:
    run_agent = None

# Commented out simplified implementations (fallback if real ones fail)
# try:
#     from agent_frameworks_jssp.autogen_multi_agent.simple_router import run_autogen_agents
# except ImportError:
#     run_autogen_agents = None
# 
# try:
#     from agent_frameworks_jssp.crewai_multi_agent.simple_router import run_crewai
# except ImportError:
#     run_crewai = None
# 
# try:
#     from agent_frameworks_jssp.openai_swarm_agent.simple_router import SwarmRouter
# except ImportError:
#     SwarmRouter = None
# 
# try:
#     from agent_frameworks_jssp.langgraph.simple_router import run_agent
# except ImportError:
#     run_agent = None

class JSSPComparisonSystem:
    """
    Comprehensive comparison system for JSSP across multiple multi-agent frameworks
    """
    
    def __init__(self, api_key: str):
        """Initialize the comparison system"""
        self.api_key = api_key
        # Set both OpenAI and Anthropic API keys for compatibility
        # Use the provided API key for OpenAI (other frameworks need OpenAI)
        os.environ["OPENAI_API_KEY"] = "API_KEY_TO_REPLACE"
        # Use new Anthropic API key for MAS testing
        os.environ["ANTHROPIC_API_KEY"] = "API_KEY_TO_REPLACE"
        
        # Framework configurations - All frameworks enabled
        self.frameworks = {
            "MAPLE": self._run_maple,
            "AutoGen": self._run_autogen,
            "CrewAI": self._run_crewai,
            "OpenAI_Swarm": self._run_swarm,
            "LangGraph": self._run_langgraph
        }
        
        # Results storage
        self.results = {}
        
    def _run_maple(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run MAPLE framework on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running MAPLE on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Create agents for each job
            agents = []
            for job in jobs:
                agent = JSSPAgent(
                    name=f"{job['name']} Agent",
                    backstory=f"Agent for {job['name']} scheduling.",
                    task_description=f"Schedule steps for {job['name']} on required machines with precedence.",
                    task_expected_output=f"Step schedule for {job['name']} respecting machine and precedence constraints.",
                    model_type="anthropic",  # Use Claude-4 instead of OpenAI
                    model_name="claude-sonnet-4-20250514"  # Specify Claude-4 model
                )
                # Set the client explicitly
                agent.client = get_llm_client("anthropic")
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
                model_type="anthropic",  # Use Claude-4 instead of OpenAI
                model_name="claude-sonnet-4-20250514"  # Specify Claude-4 model
            )
            # Set the client explicitly
            supervisor_agent.client = get_llm_client("anthropic")

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

            # Extract results and agent outputs
            context = maple.executor.context
            supervisor_output = context.get(supervisor_agent.name, {})
            
            # Capture agent prompts and outputs
            agent_prompts = {}
            agent_outputs = {}
            
            for agent in agents:
                agent_context = context.get(agent.name, {})
                if isinstance(agent_context, dict):
                    agent_prompts[agent.name] = {
                        'backstory': agent.backstory,
                        'task_description': agent.task_description,
                        'task_expected_output': agent.task_expected_output
                    }
                    # Check for different possible output formats in MAPLE
                    if 'schedule' in agent_context:
                        agent_outputs[agent.name] = f"Generated schedule: {agent_context['schedule']}"
                    elif 'output' in agent_context:
                        agent_outputs[agent.name] = agent_context['output']
                    elif 'context' in agent_context and isinstance(agent_context['context'], dict):
                        if 'schedule' in agent_context['context']:
                            agent_outputs[agent.name] = f"Generated schedule: {agent_context['context']['schedule']}"
                        else:
                            agent_outputs[agent.name] = str(agent_context['context'])
                    else:
                        agent_outputs[agent.name] = f"Agent context: {agent_context}"
                else:
                    agent_prompts[agent.name] = {
                        'backstory': agent.backstory,
                        'task_description': agent.task_description,
                        'task_expected_output': agent.task_expected_output
                    }
                    agent_outputs[agent.name] = str(agent_context)
            
            if isinstance(supervisor_output, dict) and 'schedule' in supervisor_output:
                all_schedules = supervisor_output['schedule']
                makespan = max(entry.get('end', 0) for entry in all_schedules)
                
                # Calculate UB and gap
                machine_names = list({entry.get('machine') for entry in all_schedules if 'machine' in entry})
                job_sums = {job['name']: sum(duration for _, duration in job['steps']) for job in jobs}
                machine_sums = {name: 0 for name in machine_names}
                for job in jobs:
                    for machine, duration in job['steps']:
                        machine_sums[machine] += duration
                
                ub = max(max(job_sums.values()), max(machine_sums.values()))
                gap = makespan - ub
                
                result = {
                    'success': True,
                    'makespan': makespan,
                    'upper_bound': ub,
                    'gap': gap,
                    'schedule': all_schedules,
                    'agent_prompts': agent_prompts,
                    'agent_outputs': agent_outputs,
                    'execution_time': time.time() - start_time
                }
            else:
                result = {
                    'success': False,
                    'error': 'No valid schedule generated',
                    'agent_prompts': agent_prompts,
                    'agent_outputs': agent_outputs,
                    'execution_time': time.time() - start_time
                }
                
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
        return result
    
    def _run_autogen(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run AutoGen framework on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running AutoGen on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if run_autogen_agents is None:
                raise ImportError("AutoGen framework not available")
            
            # Create JSSP query for AutoGen
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Call real AutoGen framework
            response = run_autogen_agents(jssp_query)
            
            # Extract makespan and structured schedule from response
            makespan = self._extract_makespan_from_response(response)
            structured_schedule = self._extract_structured_schedule(response)
            
            # Parse agent outputs from AutoGen response
            agent_outputs = self._extract_autogen_agent_outputs(response)
            
            # Create agent prompts structure for 3-agent system
            agent_prompts = {
                "Job Scheduler Agent": {
                    'system_message': 'You are a Job Scheduler Agent responsible for analyzing job requirements and creating initial schedules.',
                    'query': jssp_query
                },
                "Machine Coordinator Agent": {
                    'system_message': 'You are a Machine Coordinator Agent responsible for coordinating machine usage and resolving conflicts.',
                    'query': jssp_query
                },
                "Supervisor Agent": {
                    'system_message': 'You are a Supervisor Agent responsible for final coordination and optimization.',
                    'query': jssp_query
                }
            }
            
            result = {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response,
                'agent_prompts': agent_prompts,
                'agent_outputs': agent_outputs,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
        return result
    
    def _run_crewai(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run CrewAI framework on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running CrewAI on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if run_crewai is None:
                raise ImportError("CrewAI framework not available")
            
            # Create JSSP query for CrewAI
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Capture complete terminal output during execution
            import io
            import sys
            from contextlib import redirect_stdout, redirect_stderr
            
            # Create string buffer to capture output
            terminal_output = io.StringIO()
            
            # Redirect stdout and stderr to capture all output
            with redirect_stdout(terminal_output), redirect_stderr(terminal_output):
                # Call real CrewAI framework
                response = run_crewai(jssp_query)
            
            # Get the captured terminal output
            captured_output = terminal_output.getvalue()
            
            # Extract makespan and structured schedule from response
            makespan = self._extract_makespan_from_response(response)
            structured_schedule = self._extract_structured_schedule(response)
            
            # Parse agent outputs from CrewAI response
            agent_outputs = self._extract_crewai_agent_outputs(response, jobs)
            
            # Create agent prompts structure for 3-agent system
            agent_prompts = {
                "Job Scheduler Agent": {
                    'role': 'Job Scheduler',
                    'goal': 'Analyze job requirements and create initial schedules for all jobs.',
                    'backstory': 'You are an expert job scheduler responsible for analyzing job requirements and creating initial schedules.',
                    'query': jssp_query
                },
                "Machine Coordinator Agent": {
                    'role': 'Machine Coordinator',
                    'goal': 'Coordinate machine usage and resolve conflicts between jobs.',
                    'backstory': 'You are an expert machine coordinator responsible for optimizing machine usage and resolving conflicts.',
                    'query': jssp_query
                },
                "Supervisor Agent": {
                    'role': 'JSSP Supervisor',
                    'goal': 'Aggregate all job schedules and produce the overall JSSP schedule. Coordinate between all job agents to find the optimal solution.',
                    'backstory': 'You are a supervisor with expertise in Job Shop Scheduling Problems and can coordinate multiple agents to find optimal solutions.',
                    'query': jssp_query
                }
            }
            
            result = {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response,
                'terminal_output': captured_output,
                'agent_prompts': agent_prompts,
                'agent_outputs': agent_outputs,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
        return result
    
    def _run_swarm(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run OpenAI Swarm framework on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running OpenAI Swarm on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if run_swarm_agents is None:
                raise ImportError("OpenAI Swarm framework not available")
            
            # Create JSSP query for OpenAI Swarm
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Call real OpenAI Swarm framework
            response = run_swarm_agents(jssp_query)
            
            # Extract makespan and structured schedule from response
            makespan = self._extract_makespan_from_response(response)
            structured_schedule = self._extract_structured_schedule(response)
            
            # Parse agent outputs from Swarm response
            agent_outputs = self._extract_swarm_agent_outputs(response)
            
            # Create agent prompts structure for 3 agents only with dataset information
            agent_prompts = {}
            
            # Create detailed job specifications for the prompts
            job_specs = ""
            for job in jobs:
                job_specs += f"\n{job['name']}:"
                for i, (machine, duration) in enumerate(job['steps']):
                    job_specs += f"\n  Step {i+1}: Machine {machine}, Duration {duration}"
            
            # Job Scheduler Agent
            agent_prompts["Job Scheduler Agent"] = {
                'role': 'Job Scheduler',
                'goal': 'Create initial schedules for all jobs considering precedence constraints.',
                'backstory': 'You are a job scheduling agent responsible for creating initial schedules for all jobs in Job Shop Scheduling Problems.',
                'query': f"""Create initial schedules for {len(jobs)} jobs with their respective steps.

Dataset: {dataset_name}
Number of jobs: {len(jobs)}
Number of machines: {len(set(machine for job in jobs for machine, _ in job['steps']))}

Job Specifications:
{job_specs}

You must analyze each job's operations and create a detailed schedule with start and end times for each operation."""
            }
            
            # Machine Coordinator Agent
            agent_prompts["Machine Coordinator Agent"] = {
                'role': 'Machine Coordinator',
                'goal': 'Resolve machine conflicts and optimize machine utilization.',
                'backstory': 'You are a machine coordination agent responsible for resolving machine conflicts and optimizing machine utilization.',
                'query': f"""Resolve machine conflicts and optimize machine utilization across all machines.

Dataset: {dataset_name}
Number of jobs: {len(jobs)}
Number of machines: {len(set(machine for job in jobs for machine, _ in job['steps']))}

Job Specifications:
{job_specs}

You must identify machine conflicts and provide specific solutions with start/end times for each operation."""
            }
            
            # Supervisor Agent
            agent_prompts["Supervisor Agent"] = {
                'role': 'JSSP Supervisor',
                'goal': 'Coordinate all agents and provide final JSSP solution with minimum makespan.',
                'backstory': 'You are the supervisor agent for Job Shop Scheduling Problems. Coordinate with all agents to find the optimal solution.',
                'query': f"""Aggregate schedules from all agents and optimize for minimum makespan.

Dataset: {dataset_name}
Number of jobs: {len(jobs)}
Number of machines: {len(set(machine for job in jobs for machine, _ in job['steps']))}

Job Specifications:
{job_specs}

You must provide a final schedule with specific start/end times and calculate the actual makespan."""
            }
            
            result = {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response,
                'agent_prompts': agent_prompts,
                'agent_outputs': agent_outputs,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
        return result
    
    def _run_langgraph(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run LangGraph framework on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running LangGraph on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if run_agent is None:
                raise ImportError("LangGraph framework not available")
            
            # Create JSSP query for LangGraph
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Call real LangGraph framework
            response = run_agent(jssp_query)
            
            # Extract makespan and structured schedule from response
            makespan = self._extract_makespan_from_response(response)
            structured_schedule = self._extract_structured_schedule(response)
            
            # Parse agent outputs from LangGraph response
            agent_outputs = self._extract_langgraph_agent_outputs(response)
            
            # Create agent prompts structure for 3-agent system
            agent_prompts = {
                "Job Scheduler Agent": {
                    'system_prompt': 'You are a Job Scheduler Agent using LangGraph workflow. Analyze job requirements and create initial schedules for all jobs.',
                    'query': jssp_query
                },
                "Machine Coordinator Agent": {
                    'system_prompt': 'You are a Machine Coordinator Agent using LangGraph workflow. Coordinate machine usage and resolve conflicts between jobs.',
                    'query': jssp_query
                },
                "Supervisor Agent": {
                    'system_prompt': 'You are a Supervisor Agent using LangGraph workflow. Aggregate all job schedules and produce the overall JSSP schedule. Coordinate between all job agents to find the optimal solution.',
                    'query': jssp_query
                }
            }
            
            result = {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response,
                'agent_prompts': agent_prompts,
                'agent_outputs': agent_outputs,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            result = {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
            
        return result
    
    def _create_jssp_query(self, dataset_name: str, jobs: List[Dict]) -> str:
        """Create a JSSP query for non-MAPLE frameworks"""
        query = f"""
        Job Shop Scheduling Problem (JSSP) - Dataset: {dataset_name}
        
        Problem Description:
        - Number of jobs: {len(jobs)}
        - Each job has multiple operations that must be performed in sequence
        - Each operation requires a specific machine and has a duration
        - Each machine can only process one operation at a time
        - Goal: Find the minimum makespan (total completion time)
        
        Job Specifications:
        """
        
        for job in jobs:
            query += f"\n{job['name']}:"
            for i, (machine, duration) in enumerate(job['steps']):
                query += f"\n  Step {i+1}: Machine {machine}, Duration {duration}"
        
        query += """
        
        REQUIRED OUTPUT FORMAT:
        You must provide your solution in the following exact format:
        
        1. FINAL MAKESPAN: [integer value]
        2. STRUCTURED SCHEDULE:
           For each operation, provide:
           - Job: [job_name]
           - Step: [step_number]
           - Machine: [machine_name]
           - Start Time: [start_time]
           - End Time: [end_time]
           - Duration: [duration]
        
        Example format:
        FINAL MAKESPAN: 25
        STRUCTURED SCHEDULE:
        - Job: Job1, Step: 1, Machine: Machine0, Start Time: 0, End Time: 3, Duration: 3
        - Job: Job1, Step: 2, Machine: Machine1, Start Time: 3, End Time: 7, Duration: 4
        - Job: Job2, Step: 1, Machine: Machine1, Start Time: 7, End Time: 10, Duration: 3
        
        Please solve this Job Shop Scheduling Problem and provide:
        1. A valid schedule with start and end times for each operation
        2. The minimum makespan (total completion time)
        3. Ensure all constraints are satisfied:
           - Job precedence: operations within a job must be sequential
           - Machine constraints: no overlapping operations on the same machine
        """
        
        return query
    
    def _extract_makespan_from_response(self, response: str) -> Optional[int]:
        """Extract makespan from framework response using the required format"""
        import re
        
        # Look for the specific format we requested: "FINAL MAKESPAN: [integer]"
        patterns = [
            r'FINAL MAKESPAN:\s*(\d+)',
            r'FINAL MAKESPAN\s*:\s*(\d+)',
            r'FINAL MAKESPAN\s*=\s*(\d+)',
            # Fallback patterns for other formats
            r'makespan[:\s]*(\d+)',
            r'total time[:\s]*(\d+)',
            r'completion time[:\s]*(\d+)',
            r'finish time[:\s]*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_structured_schedule(self, response: str) -> List[Dict]:
        """Extract structured schedule from framework response"""
        import re
        
        schedule = []
        
        # Look for the structured schedule format we requested
        # Pattern: "Job: [name], Step: [num], Machine: [name], Start Time: [num], End Time: [num], Duration: [num]"
        pattern = r'Job:\s*([^,]+),\s*Step:\s*(\d+),\s*Machine:\s*([^,]+),\s*Start Time:\s*(\d+),\s*End Time:\s*(\d+),\s*Duration:\s*(\d+)'
        
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for match in matches:
            job_name, step_num, machine_name, start_time, end_time, duration = match
            schedule.append({
                'job': job_name.strip(),
                'step': int(step_num),
                'machine': machine_name.strip(),
                'start': int(start_time),
                'end': int(end_time),
                'duration': int(duration)
            })
        
        return schedule
    
    def _simulate_job_schedule(self, job: Dict) -> List[Dict]:
        """Simulate a job agent scheduling its steps"""
        schedule = []
        current_time = 0
        
        for step_idx, (machine, duration) in enumerate(job['steps']):
            start_time = current_time
            end_time = start_time + duration
            
            schedule.append({
                'job': job['name'],
                'step': step_idx + 1,
                'machine': machine,
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
            
            current_time = end_time
        
        return schedule
    
    def _simulate_supervisor_coordination(self, job_schedules: List[List[Dict]], jobs: List[Dict]) -> List[Dict]:
        """Simulate supervisor agent coordinating all job schedules"""
        # Combine all job schedules
        all_operations = []
        for job_schedule in job_schedules:
            all_operations.extend(job_schedule)
        
        # Simple coordination: resolve machine conflicts by delaying operations
        machine_availability = {}
        final_schedule = []
        
        for operation in all_operations:
            machine = operation['machine']
            duration = operation['duration']
            
            # Find earliest available time for this machine
            start_time = max(
                machine_availability.get(machine, 0),
                operation['start']  # Respect job precedence
            )
            
            end_time = start_time + duration
            machine_availability[machine] = end_time
            
            final_schedule.append({
                'job': operation['job'],
                'step': operation['step'],
                'machine': machine,
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
        
        return final_schedule
    
    def _extract_autogen_agent_outputs(self, response: str) -> Dict[str, str]:
        """Extract individual agent outputs from AutoGen combined response"""
        agent_outputs = {}
        
        # Clean up the response to extract just the content
        # Remove any AutoGen-specific formatting and extract the actual content
        if "content='" in response:
            # Extract content from AutoGen response format
            start_idx = response.find("content='")
            if start_idx != -1:
                start_idx += 9  # Skip "content='"
                end_idx = response.find("', type='TextMessage'", start_idx)
                if end_idx != -1:
                    response = response[start_idx:end_idx]
        
        # Parse the actual AutoGen conversation to extract real agent responses
        # Look for specific agent responses in the conversation
        print(f"🔍 DEBUG: Looking for agent responses in AutoGen conversation...")
        print(f"🔍 DEBUG: Response length: {len(response)}")
        print(f"🔍 DEBUG: First 500 chars: {response[:500]}")
        
        # Find all agent responses in the conversation
        agent_responses = {}
        
        # Look for all TextMessage entries with agent sources
        import re
        pattern = r"source='([^']+)'.*?content='([^']+)'"
        matches = re.findall(pattern, response, re.DOTALL)
        
        print(f"🔍 DEBUG: Found {len(matches)} matches with regex pattern")
        
        for source, content in matches:
            print(f"🔍 DEBUG: Found source: {source}")
            if source in ['Job_Scheduler', 'Machine_Coordinator', 'Supervisor']:
                # Map to the expected agent names
                agent_name = f"{source.replace('_', ' ')} Agent"
                agent_responses[agent_name] = content
                print(f"🔍 DEBUG: Found {agent_name} response:")
                print("=" * 60)
                print(content)
                print("=" * 60)
        
        # If we found agent responses, use them
        if agent_responses:
            agent_outputs.update(agent_responses)
        else:
            # Fallback: look for the final response from JSSP_Coordinator
            if "source='JSSP_Coordinator'" in response:
                coordinator_start = response.find("source='JSSP_Coordinator'")
                if coordinator_start != -1:
                    content_start = response.find("content='", coordinator_start)
                    if content_start != -1:
                        content_start += 9  # Skip "content='"
                        content_end = response.find("', type='TextMessage'", content_start)
                        if content_end != -1:
                            coordinator_output = response[content_start:content_end]
                            agent_outputs['Supervisor Agent'] = coordinator_output
                            print(f"🔍 DEBUG: JSSP_Coordinator final response:")
                            print("=" * 60)
                            print(coordinator_output)
                            print("=" * 60)
        
        # If no real agent responses were found, fall back to the entire response
        if not agent_outputs:
            agent_outputs = {
                'Job Scheduler Agent': response,
                'Machine Coordinator Agent': response,
                'Supervisor Agent': response
            }
        
        return agent_outputs
    
    def _extract_swarm_agent_outputs(self, response: str) -> Dict[str, str]:
        """Extract individual agent outputs from OpenAI Swarm response"""
        agent_outputs = {}
        
        # Split response by agent sections
        if "Job Scheduler Analysis:" in response:
            job_scheduler_start = response.find("Job Scheduler Analysis:")
            machine_coordinator_start = response.find("Machine Coordinator Analysis:")
            supervisor_start = response.find("Supervisor Final Coordination:")
            
            if job_scheduler_start != -1:
                if machine_coordinator_start != -1:
                    job_scheduler_output = response[job_scheduler_start:machine_coordinator_start].strip()
                else:
                    job_scheduler_output = response[job_scheduler_start:].strip()
                agent_outputs['Job Scheduler Agent'] = job_scheduler_output
            
            if machine_coordinator_start != -1:
                if supervisor_start != -1:
                    machine_coordinator_output = response[machine_coordinator_start:supervisor_start].strip()
                else:
                    machine_coordinator_output = response[machine_coordinator_start:].strip()
                agent_outputs['Machine Coordinator Agent'] = machine_coordinator_output
            
            if supervisor_start != -1:
                supervisor_output = response[supervisor_start:].strip()
                agent_outputs['Supervisor Agent'] = supervisor_output
        else:
            # Fallback: assign the entire response to all 3 agents
            agent_outputs = {
                'Job Scheduler Agent': response,
                'Machine Coordinator Agent': response,
                'Supervisor Agent': response
            }
        
        return agent_outputs
    
    def _extract_langgraph_agent_outputs(self, response: str) -> Dict[str, str]:
        """Extract individual agent outputs from LangGraph response"""
        agent_outputs = {}
        
        # Debug: Show what LangGraph is actually returning
        print(f"🔍 DEBUG: LangGraph response length: {len(response)}")
        print(f"🔍 DEBUG: Full LangGraph response:")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        # LangGraph typically has a single agent with multiple workflow nodes
        # We'll create outputs for each workflow node
        workflow_nodes = [
            'Problem_Analysis',
            'Job_Scheduling', 
            'Conflict_Resolution',
            'Optimization',
            'Validation'
        ]
        
        # Split response by workflow sections if they exist
        if "Problem Analysis" in response or "Job Scheduling" in response:
            # Try to extract individual node outputs
            for node in workflow_nodes:
                node_name = node.replace('_', ' ')
                if node_name in response:
                    start_idx = response.find(node_name)
                    # Find the next node or end of response
                    next_nodes = [n.replace('_', ' ') for n in workflow_nodes if n != node]
                    next_idx = len(response)
                    for next_node in next_nodes:
                        if next_node in response[start_idx + len(node_name):]:
                            next_idx = min(next_idx, response.find(next_node, start_idx + len(node_name)))
                    
                    node_output = response[start_idx:next_idx].strip()
                    agent_outputs[node] = node_output
        else:
            # Fallback: assign the entire response to the 3-agent structure
            agent_outputs = {
                'Job Scheduler Agent': response,
                'Machine Coordinator Agent': response,
                'Supervisor Agent': response
            }
        
        return agent_outputs
    
    def _extract_crewai_agent_outputs(self, response: str, jobs: List[Dict]) -> Dict[str, str]:
        """Extract individual agent outputs from CrewAI response"""
        agent_outputs = {}
        
        # Parse CrewAI response to extract individual agent outputs
        lines = response.split('\n')
        current_agent = None
        
        for line in lines:
            line = line.strip()
            # Look for the new 3-agent structure
            if 'Job Scheduler' in line:
                current_agent = "Job Scheduler Agent"
            elif 'Machine Coordinator' in line:
                current_agent = "Machine Coordinator Agent"
            elif 'Supervisor' in line:
                current_agent = "Supervisor Agent"
            elif current_agent and line:
                if current_agent not in agent_outputs:
                    agent_outputs[current_agent] = ""
                agent_outputs[current_agent] += line + "\n"
        
        # If no specific agent outputs found, create generic ones for 3-agent structure
        if not agent_outputs:
            agent_outputs = {
                "Job Scheduler Agent": "CrewAI Job Scheduler Agent processed job scheduling",
                "Machine Coordinator Agent": "CrewAI Machine Coordinator Agent processed machine coordination", 
                "Supervisor Agent": "CrewAI Supervisor coordinated all agents"
            }
        
        return agent_outputs
    
    def run_comparison(self, dataset_name: str, dataset_path: str, frameworks: List[str] = None) -> Dict[str, Any]:
        """Run comparison across specified frameworks"""
        if frameworks is None:
            frameworks = list(self.frameworks.keys())
        
        print(f"\n{'='*80}")
        print(f"JSSP Framework Comparison - Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load dataset
        jobs = load_dmu_dataset(dataset_path)
        print(f"Loaded {len(jobs)} jobs from {dataset_name}")
        
        # Run each framework
        comparison_results = {
            'dataset': dataset_name,
            'num_jobs': len(jobs),
            'timestamp': datetime.now().isoformat(),
            'frameworks': {}
        }
        
        for framework_name in frameworks:
            if framework_name in self.frameworks:
                print(f"\n🔄 Testing {framework_name}...")
                result = self.frameworks[framework_name](dataset_name, jobs)
                comparison_results['frameworks'][framework_name] = result
                
                if result['success']:
                    print(f"✅ {framework_name}: Makespan = {result.get('makespan', 'N/A')}, Time = {result['execution_time']:.2f}s")
                else:
                    print(f"❌ {framework_name}: Failed - {result.get('error', 'Unknown error')}")
            else:
                print(f"⚠️ Framework {framework_name} not found")
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save comparison results to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comparison summary"""
        print(f"\n{'='*80}")
        print("JSSP FRAMEWORK COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Dataset: {results['dataset']}")
        print(f"Jobs: {results['num_jobs']}")
        print(f"Timestamp: {results['timestamp']}")
        
        print(f"\n{'Framework':<15} {'Success':<8} {'Makespan':<10} {'Schedule':<10} {'Time (s)':<10} {'Status'}")
        print("-" * 80)
        
        for framework, result in results['frameworks'].items():
            success = "✅" if result['success'] else "❌"
            makespan = str(result.get('makespan', 'N/A')) if result['success'] else 'N/A'
            schedule_count = len(result.get('structured_schedule', [])) if result['success'] else 0
            exec_time = f"{result['execution_time']:.2f}"
            status = "Success" if result['success'] else f"Error: {result.get('error', 'Unknown')[:20]}..."
            
            print(f"{framework:<15} {success:<8} {makespan:<10} {schedule_count:<10} {exec_time:<10} {status}")
    
    def print_agent_details(self, results: Dict[str, Any]):
        """Print detailed agent prompts and outputs for each framework"""
        print(f"\n{'='*80}")
        print("AGENT PROMPTS AND OUTPUTS DETAILS")
        print(f"{'='*80}")
        
        for framework, result in results['frameworks'].items():
            print(f"\n🔍 {framework} Framework Details:")
            print("-" * 50)
            
            if 'agent_prompts' in result and 'agent_outputs' in result:
                agent_prompts = result['agent_prompts']
                agent_outputs = result['agent_outputs']
                
                for agent_name, prompt_info in agent_prompts.items():
                    print(f"\n📝 {agent_name} Prompt:")
                    for key, value in prompt_info.items():
                        if key != 'query':  # Skip the full query to avoid clutter
                            print(f"  {key}: {value}")
                    
                    if agent_name in agent_outputs:
                        output = agent_outputs[agent_name]
                        print(f"\n💬 {agent_name} Output:")
                        # Show full output, not truncated
                        print(f"  {output}")
                    else:
                        print(f"\n💬 {agent_name} Output: No output generated")
            else:
                print("  No agent details available")
    
    def save_results_to_txt(self, results: Dict[str, Any], output_file: str):
        """Save results to a detailed text file"""
        txt_file = output_file.replace('.json', '.txt')
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("JSSP FRAMEWORK COMPARISON RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {results['dataset']}\n")
            f.write(f"Jobs: {results['num_jobs']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            f.write("FRAMEWORK PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            for framework, result in results['frameworks'].items():
                success = "✅" if result['success'] else "❌"
                makespan = str(result.get('makespan', 'N/A')) if result['success'] else 'N/A'
                schedule_count = len(result.get('structured_schedule', [])) if result['success'] else 0
                exec_time = f"{result['execution_time']:.2f}"
                status = "Success" if result['success'] else f"Error: {result.get('error', 'Unknown')}"
                
                f.write(f"{framework}: {success} Makespan={makespan}, Schedule_Ops={schedule_count}, Time={exec_time}s, Status={status}\n")
            
            f.write("\n\nDETAILED AGENT PROMPTS AND OUTPUTS\n")
            f.write("=" * 50 + "\n")
            
            for framework, result in results['frameworks'].items():
                f.write(f"\n🔍 {framework} Framework Details:\n")
                f.write("-" * 50 + "\n")
                
                if 'agent_prompts' in result and 'agent_outputs' in result:
                    agent_prompts = result['agent_prompts']
                    agent_outputs = result['agent_outputs']
                    
                    for agent_name, prompt_info in agent_prompts.items():
                        f.write(f"\n📝 {agent_name} Prompt:\n")
                        for key, value in prompt_info.items():
                            f.write(f"  {key}: {value}\n")
                        
                        if agent_name in agent_outputs:
                            output = agent_outputs[agent_name]
                            f.write(f"\n💬 {agent_name} Output:\n")
                            f.write(f"  {output}\n")
                else:
                    f.write("  No agent details available\n")
                
                # Add structured schedule information
                if 'structured_schedule' in result and result['structured_schedule']:
                    f.write(f"\n📅 Structured Schedule ({len(result['structured_schedule'])} operations):\n")
                    for i, op in enumerate(result['structured_schedule'][:10]):  # Show first 10 operations
                        f.write(f"  {i+1}. Job: {op['job']}, Step: {op['step']}, Machine: {op['machine']}, Start: {op['start']}, End: {op['end']}, Duration: {op['duration']}\n")
                    if len(result['structured_schedule']) > 10:
                        f.write(f"  ... and {len(result['structured_schedule']) - 10} more operations\n")
                else:
                    f.write(f"\n📅 Structured Schedule: No structured schedule available\n")
        
        print(f"📁 Detailed results saved to: {txt_file}")
    
    def save_terminal_output_to_txt(self, results: Dict[str, Any], output_file: str):
        """Save complete terminal output to TXT file"""
        txt_file = output_file.replace('.json', '_terminal_output.txt')
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("COMPLETE TERMINAL OUTPUT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {results['dataset']}\n")
            f.write(f"Jobs: {results['num_jobs']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            for framework, result in results['frameworks'].items():
                f.write(f"\n{'='*60}\n")
                f.write(f"FRAMEWORK: {framework}\n")
                f.write(f"{'='*60}\n\n")
                
                if 'success' in result:
                    success = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                    f.write(f"Status: {success}\n")
                    
                    if result['success']:
                        f.write(f"Makespan: {result.get('makespan', 'N/A')}\n")
                        f.write(f"Execution Time: {result.get('execution_time', 0):.2f}s\n")
                    else:
                        f.write(f"Error: {result.get('error', 'Unknown')}\n")
                
                f.write(f"\n{'-'*60}\n")
                f.write("COMPLETE FRAMEWORK OUTPUT:\n")
                f.write(f"{'-'*60}\n")
                
                # Store the complete response/terminal output
                if 'terminal_output' in result:
                    f.write(str(result['terminal_output']))
                elif 'response' in result:
                    f.write(str(result['response']))
                else:
                    f.write("No terminal output captured")
                
                f.write(f"\n\n{'-'*60}\n")
                f.write("AGENT DETAILS:\n")
                f.write(f"{'-'*60}\n")
                
                if 'agent_prompts' in result and 'agent_outputs' in result:
                    agent_prompts = result['agent_prompts']
                    agent_outputs = result['agent_outputs']
                    
                    for agent_name, prompt_info in agent_prompts.items():
                        f.write(f"\n📝 {agent_name} Prompt:\n")
                        for key, value in prompt_info.items():
                            f.write(f"  {key}: {value}\n")
                        
                        if agent_name in agent_outputs:
                            output = agent_outputs[agent_name]
                            f.write(f"\n💬 {agent_name} Output:\n")
                            f.write(f"  {output}\n")
                else:
                    f.write("  No agent details available\n")
        
        print(f"📄 Terminal output saved to TXT: {txt_file}")
    
    def save_agent_details_to_txt(self, results: Dict[str, Any], output_file: str):
        """Save agent prompts and outputs to TXT file"""
        txt_file = output_file.replace('.json', '_agent_details.txt')
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("AGENT PROMPTS AND OUTPUTS DETAILS\n")
            f.write("=" * 50 + "\n\n")
            
            for framework, result in results['frameworks'].items():
                f.write(f"🔍 {framework} Framework Details:\n")
                f.write("-" * 50 + "\n")
                
                if 'agent_prompts' in result and 'agent_outputs' in result:
                    agent_prompts = result['agent_prompts']
                    agent_outputs = result['agent_outputs']
                    
                    for agent_name, prompt_info in agent_prompts.items():
                        f.write(f"\n📝 {agent_name} Prompt:\n")
                        for key, value in prompt_info.items():
                            if key != 'query':  # Skip the full query to avoid clutter
                                f.write(f"  {key}: {value}\n")
                        
                        if agent_name in agent_outputs:
                            output = agent_outputs[agent_name]
                            f.write(f"\n💬 {agent_name} Output:\n")
                            f.write(f"  {output}\n")
                        else:
                            f.write(f"\n💬 {agent_name} Output: No output generated\n")
                else:
                    f.write("  No agent details available\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"📄 Agent details saved to TXT: {txt_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='JSSP Framework Comparison System')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., rcmax_20_15_5)')
    parser.add_argument('--api-key', type=str, required=True, help='OpenAI API key')
    parser.add_argument('--frameworks', nargs='+', default=['MAPLE', 'AutoGen', 'CrewAI', 'OpenAI_Swarm', 'LangGraph'],
                       help='Frameworks to compare')
    parser.add_argument('--output', type=str, default='./results/jssp_comparison.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Determine dataset path
    dataset_path = None
    for subdir in ['DMU', 'TA', 'abzswvyn']:
        potential_path = os.path.join(project_root, 'applications', subdir, f'{args.dataset}.txt')
        if os.path.exists(potential_path):
            dataset_path = potential_path
            break
    
    if not dataset_path:
        print(f"❌ Dataset {args.dataset} not found in DMU, TA, or ABZSWVYN directories")
        return
    
    # Initialize comparison system
    comparison_system = JSSPComparisonSystem(args.api_key)
    
    # Run comparison
    results = comparison_system.run_comparison(args.dataset, dataset_path, args.frameworks)
    
    # Generate framework-specific output files
    for framework in args.frameworks:
        if framework in results['frameworks']:
            # Create framework-specific output files
            json_output = f"./results/jssp_results_{args.dataset}_{framework}.json"
            txt_output = f"./results/jssp_results_{args.dataset}_{framework}.txt"
            agent_output = f"./results/jssp_results_{args.dataset}_{framework}_agent_details.txt"
            
            # Create framework-specific results
            framework_results = {
                'dataset': results['dataset'],
                'num_jobs': results['num_jobs'],
                'timestamp': results['timestamp'],
                'frameworks': {framework: results['frameworks'][framework]}
            }
            
            # Save framework-specific results
            comparison_system.save_results(framework_results, json_output)
            comparison_system.save_results_to_txt(framework_results, txt_output)
            comparison_system.save_agent_details_to_txt(framework_results, agent_output)
            comparison_system.save_terminal_output_to_txt(framework_results, json_output)
    
    # Also save combined results
    comparison_system.save_results(results, args.output)
    comparison_system.save_results_to_txt(results, args.output)
    comparison_system.save_agent_details_to_txt(results, args.output)
    comparison_system.save_terminal_output_to_txt(results, args.output)
    comparison_system.print_summary(results)
    comparison_system.print_agent_details(results)

if __name__ == "__main__":
    main()
