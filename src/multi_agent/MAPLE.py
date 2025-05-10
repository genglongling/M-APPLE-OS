import sys
import os
from collections import deque
from colorama import Fore
from graphviz import Digraph  # type: ignore
from multi_agent.agent import Agent  # Add import for base Agent class

# Layer 1: Specification Construction (Workflow Construction)
class WorkflowSpecification:
    """
    Handles the construction of the workflow graph from task specifications.
    Nodes represent task steps; edges represent dependencies.
    """
    def __init__(self, task_spec):
        self.task_spec = task_spec
        self.nodes = []
        self.edges = []
        self._build_workflow()

    def _build_workflow(self):
        # Placeholder: parse task_spec to build nodes and edges
        # In practice, this would parse a structured task specification
        self.nodes = self.task_spec.get('nodes', [])
        self.edges = self.task_spec.get('edges', [])

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

# Layer 2: Inter-Agent Coordination (Dependency Management)
class InterAgentCoordinator:
    """
    Manages agent instantiation, dependency resolution, and coordination.
    """
    def __init__(self, nodes, edges):
        # Ensure all agents inherit from base Agent class
        self.agents = []
        for node in nodes:
            agent = node['agent']
            if not isinstance(agent, Agent):
                # Convert to base Agent if not already
                base_agent = Agent(
                    name=agent.name,
                    backstory=getattr(agent, 'backstory', ''),
                    task_description=getattr(agent, 'task_description', ''),
                    task_expected_output=getattr(agent, 'task_expected_output', '')
                )
                # Copy over any additional attributes
                for attr in dir(agent):
                    if not attr.startswith('_') and not hasattr(base_agent, attr):
                        setattr(base_agent, attr, getattr(agent, attr))
                self.agents.append(base_agent)
            else:
                self.agents.append(agent)
        
        self.dependencies = {node['agent'].name: [dep for dep in node.get('dependencies', [])] for node in nodes}
        self._set_agent_dependencies()

    def _set_agent_dependencies(self):
        # Set dependencies and dependents for each agent
        name_to_agent = {agent.name: agent for agent in self.agents}
        for agent in self.agents:
            agent.dependencies = [name_to_agent[dep] for dep in self.dependencies.get(agent.name, [])]
            agent.dependents = [a for a in self.agents if agent.name in self.dependencies.get(a.name, [])]

    def topological_sort(self):
        in_degree = {agent: len(agent.dependencies) for agent in self.agents}
        queue = deque([agent for agent in self.agents if in_degree[agent] == 0])
        sorted_agents = []
        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)
            for dependent in current_agent.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        if len(sorted_agents) != len(self.agents):
            raise ValueError("Circular dependencies detected, preventing execution order.")
        return sorted_agents

# Dynamic Adaptation Manager
class DynamicAdaptationManager:
    """
    Handles dynamic adaptation when disruptions occur during execution.
    Attempts local compensation, and if that fails, triggers global replanning.
    """
    def __init__(self, workflow, coordinator, executor):
        self.workflow = workflow
        self.coordinator = coordinator
        self.executor = executor

    def handle_disruption(self, failed_agent, error):
        print(f"⚡ Disruption detected in {failed_agent.name}: {error}")
        # 1. Try local compensation
        if hasattr(failed_agent, 'compensate'):
            try:
                print(f"🛠 Attempting local compensation for {failed_agent.name}...")
                failed_agent.compensate()
                print(f"✅ Local compensation succeeded for {failed_agent.name}.")
                return True
            except Exception as comp_error:
                print(f"❌ Local compensation failed: {comp_error}")
        # 2. Global replanning
        print("🔄 Initiating global replanning...")
        # (a) Update agent states, invalidate future assignments, etc.
        self.executor.context[failed_agent.name] = "invalid"
        # (b) Re-run dependency resolution and agent assignment
        self.coordinator._set_agent_dependencies()
        # (c) Optionally, re-run the workflow from the failed point or from scratch
        try:
            self.executor.execute(with_rollback=True, adaptation_manager=self)
        except Exception as e:
            print(f"🚨 Global replanning failed: {e}")
            return False
        return True

# Layer 3: Execution, Rollback, and Self-Validation
class ExecutionManager:
    """
    Handles agent execution, rollback on failure, and self-validation of workflow execution.
    Now supports dynamic adaptation via an adaptation manager.
    """
    def __init__(self, agents):
        self.agents = agents
        self.context = {}  # Execution context for rollback and validation

    def execute(self, with_rollback=True, adaptation_manager=None):
        sorted_agents = self._topological_sort()
        executed_agents = []
        try:
            for agent in sorted_agents:
                print(f"\n🚀 Running Agent: {agent.name}")
                # Use the base Agent's run method which includes LLM integration
                result = agent.run()
                self.context[agent.name] = result
                executed_agents.append(agent)
                print(Fore.GREEN + f"✅ {agent.name} completed successfully.")

                # # Print all candidate schedules for JSSPAgent if present
                # if hasattr(agent, 'context') and isinstance(agent.context, dict) and 'candidates' in agent.context:
                #     candidates = agent.context['candidates']
                #     print(f"\n--- {agent.name} generated {len(candidates)} candidate schedules ---")
                #     for idx, schedule in enumerate(candidates):
                #         print(f"\nCandidate {idx+1}:")
                #         for entry in schedule:
                #             print(f"  Job: {entry['job']}, Step: {entry['step']}, Machine: {entry['machine']}, Start: {entry['start']}, End: {entry['end']}")

                # # print each agent result
                # print(Fore.CYAN + "📊 Agent Results:")
                # if isinstance(result, dict):
                #     for key, value in result.items():
                #         if key == 'schedule':
                #             print(f"  Schedule entries: {len(value)}")
                #             # Print first few entries as example
                #             for entry in value[:3]:
                #                 print(f"    - {entry}")
                #             if len(value) > 3:
                #                 print(f"    ... and {len(value)-3} more entries")
                #         elif key == 'errors':
                #             print(f"  Errors: {len(value)}")
                #             for error in value:
                #                 print(f"    - {error}")
                #         else:
                #             print(f"  {key}: {value}")
                # else:
                #     print(f"  {result}")
                # print(Fore.RESET)
        except Exception as e:
            print(Fore.RED + f"❌ ERROR in {agent.name}: {str(e)}")
            if adaptation_manager:
                adaptation_manager.handle_disruption(agent, e)
            elif with_rollback:
                print(Fore.YELLOW + "🔄 Rolling back executed agents...")
                for completed_agent in reversed(executed_agents):
                    try:
                        completed_agent.rollback()
                        print(Fore.BLUE + f"↩️ Rolled back: {completed_agent.name}")
                    except AttributeError:
                        print(Fore.RED + f"⚠️ {completed_agent.name} has no rollback method.")
                    except Exception as rollback_error:
                        print(Fore.RED + f"⚠️ Error rolling back {completed_agent.name}: {rollback_error}")
            print(Fore.RED + "🚨 Execution halted due to error.")

    def _topological_sort(self):
        in_degree = {agent: len(agent.dependencies) for agent in self.agents}
        queue = deque([agent for agent in self.agents if in_degree[agent] == 0])
        sorted_agents = []
        while queue:
            current_agent = queue.popleft()
            sorted_agents.append(current_agent)
            for dependent in current_agent.dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        if len(sorted_agents) != len(self.agents):
            raise ValueError("Circular dependencies detected, preventing execution order.")
        return sorted_agents

    def self_validate(self):
        """
        Validates workflow execution for structural validity, constraint satisfaction, and compensation coverage.
        Implements JSSP validation logic. If validation fails, re-run supervisor agent with error context once.
        """
        print("🔍 Self-Validation: Checking workflow structure, constraints, and compensation coverage...")
        # Find supervisor agent
        supervisor_agent = [a for a in self.agents if 'Supervisor' in a.name][0]
        supervisor_output = self.context.get(supervisor_agent.name, {})
        all_schedules = supervisor_output.get('schedule', [])

        # --- Agent Output Consistency Validation ---
        errors = []
        # Get jobs and machine_names from context if available
        jobs = self.context.get('jobs', None)
        machine_names = self.context.get('machine_names', None)

        # If jobs are not in context, try to load from task specification
        if not jobs and hasattr(self, 'task_spec'):
            jobs = self.task_spec.get('jobs', [])
            if not jobs:
                print("Warning: No jobs found in context or task specification")
                return False

        # Try to infer machine names if not present
        if not machine_names and all_schedules:
            machine_names = list({entry.get('machine') for entry in all_schedules if 'machine' in entry})
        elif not machine_names and jobs:
            # Extract machine names from job specifications
            machine_indices = set()
            for job in jobs:
                for machine, _ in job['steps']:
                    idx = int(machine.replace('Machine', ''))
                    machine_indices.add(idx)
            machine_names = [f"Machine{idx}" for idx in sorted(machine_indices)]

        # 1. Check agent output consistency
        if jobs and all_schedules:
            # Create a mapping of job steps from input specification
            job_specs = {}
            for job in jobs:
                if not job.get('steps'):
                    print(f"Warning: No steps found for job {job['name']}")
                    continue
                job_specs[job['name']] = {
                    'steps': [(machine, duration) for machine, duration in job['steps']],
                    'total_steps': len(job['steps'])
                }
                print(f"Loaded specification for {job['name']}: {len(job['steps'])} steps")

            # Validate each schedule entry against job specifications
            for entry in all_schedules:
                job_name = entry.get('job')
                step_num = entry.get('step')
                machine = entry.get('machine')
                start_time = entry.get('start')
                end_time = entry.get('end')

                # Check if job exists in specification
                if job_name not in job_specs:
                    errors.append(f"Invalid job name in schedule: {job_name}")
                    continue

                # Check if step number is valid
                if step_num > job_specs[job_name]['total_steps']:
                    errors.append(f"Invalid step number for {job_name}: Step {step_num} exceeds total steps {job_specs[job_name]['total_steps']}")
                    continue

                # Check if machine matches specification
                expected_machine, expected_duration = job_specs[job_name]['steps'][step_num - 1]
                if machine != expected_machine:
                    errors.append(f"Machine mismatch for {job_name} Step {step_num}: Expected {expected_machine}, got {machine}")

                # Check if duration matches specification
                actual_duration = end_time - start_time
                if actual_duration != expected_duration:
                    errors.append(f"Duration mismatch for {job_name} Step {step_num}: Expected {expected_duration}, got {actual_duration}")

                # Check if start and end times are valid
                if start_time < 0:
                    errors.append(f"Invalid start time for {job_name} Step {step_num}: {start_time}")
                if end_time <= start_time:
                    errors.append(f"Invalid end time for {job_name} Step {step_num}: {end_time} (start: {start_time})")

        # If there are consistency errors, add them to context and re-run supervisor
        if errors:
            print("\n=== Agent Output Consistency Validation ===")
            print("❌ Validation failed with the following consistency errors:")
            for error in errors:
                print(f"- {error}")
            self.context['validation_errors'] = errors
            print("🔄 Re-running supervisor agent with error context...")
            supervisor_agent.context = self.context
            self.context[supervisor_agent.name] = supervisor_agent.run()
            # Get updated schedule after re-run
            supervisor_output = self.context.get(supervisor_agent.name, {})
            all_schedules = supervisor_output.get('schedule', [])
            errors = []  # Reset errors for next validation phase

        # --- JSSP Validation Logic ---
        # 1. Check machine constraints (no overlapping operations)
        machine_schedules = {name: [] for name in machine_names}
        for entry in all_schedules:
            machine = entry.get('machine')
            if machine in machine_schedules:
                machine_schedules[machine].append(entry)
        for machine, schedule in machine_schedules.items():
            schedule.sort(key=lambda x: x.get('start', 0))
            # Track processed pairs to avoid duplicates
            processed_pairs = set()
            for i in range(len(schedule)-1):
                for j in range(i+1, len(schedule)):
                    pair_key = (schedule[i].get('job'), schedule[i].get('step'), 
                                schedule[j].get('job'), schedule[j].get('step'))
                    if pair_key in processed_pairs:
                        continue
                    if schedule[i].get('end', 0) > schedule[j].get('start', 0):
                        error_msg = f"Overlap detected on {machine}: {schedule[i].get('job')} Step {schedule[i].get('step')} and {schedule[j].get('job')} Step {schedule[j].get('step')}"
                        if error_msg not in errors:
                            errors.append(error_msg)
                        processed_pairs.add(pair_key)
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
        if jobs:
            for job in jobs:
                job_name = job['name']
                if job_name not in job_steps or len(job_steps[job_name]) != len(job.get('steps', [])):
                    errors.append(f"Incomplete schedule for {job_name}")
        # 4. Check makespan
        makespan = max(entry.get('end', 0) for entry in all_schedules) if all_schedules else None
        # Print validation results
        print("\n=== Validation Results ===")
        if errors:
            print("❌ Validation failed with the following errors:")
            for error in errors:
                print(f"- {error}")
            # Add errors to context and re-run supervisor ONCE
            self.context['validation_errors'] = errors
            print("🔄 Re-running supervisor agent with error context...")
            supervisor_agent.context = self.context
            self.context[supervisor_agent.name] = supervisor_agent.run()
            # Validate again
            supervisor_output = self.context.get(supervisor_agent.name, {})
            all_schedules = supervisor_output.get('schedule', [])
            # Repeat validation logic
            errors2 = []
            machine_schedules = {name: [] for name in machine_names}
            for entry in all_schedules:
                machine = entry.get('machine')
                if machine in machine_schedules:
                    machine_schedules[machine].append(entry)
            for machine, schedule in machine_schedules.items():
                schedule.sort(key=lambda x: x.get('start', 0))
                processed_pairs = set()
                for i in range(len(schedule)-1):
                    for j in range(i+1, len(schedule)):
                        pair_key = (schedule[i].get('job'), schedule[i].get('step'), 
                                    schedule[j].get('job'), schedule[j].get('step'))
                        if pair_key in processed_pairs:
                            continue
                        if schedule[i].get('end', 0) > schedule[j].get('start', 0):
                            error_msg = f"Overlap detected on {machine}: {schedule[i].get('job')} Step {schedule[i].get('step')} and {schedule[j].get('job')} Step {schedule[j].get('step')}"
                            if error_msg not in errors2:
                                errors2.append(error_msg)
                            processed_pairs.add(pair_key)
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
                        errors2.append(f"Precedence violation in {job}: Step {steps[i].get('step')} ends after Step {steps[i+1].get('step')} starts")
            if jobs:
                for job in jobs:
                    job_name = job['name']
                    if job_name not in job_steps or len(job_steps[job_name]) != len(job.get('steps', [])):
                        errors2.append(f"Incomplete schedule for {job_name}")
            makespan2 = max(entry.get('end', 0) for entry in all_schedules) if all_schedules else None
            print("\n=== Validation Results (Second Attempt) ===")
            if errors2:
                print("❌ Validation failed again. Initiating global replanning...")
                # Add errors to context for global replanning
                self.context['validation_errors'] = errors2
                self.context['global_replan'] = True
                
                # Reset supervisor agent's state
                if hasattr(supervisor_agent, 'has_run'):
                    supervisor_agent.has_run = False
                
                # Run supervisor agent with error context
                print("🔄 Running supervisor agent with global replanning...")
                supervisor_agent.context = self.context
                self.context[supervisor_agent.name] = supervisor_agent.run()
                
                # Validate one final time
                supervisor_output = self.context.get(supervisor_agent.name, {})
                all_schedules = supervisor_output.get('schedule', [])
                errors3 = []
                
                # Perform final validation
                if all_schedules:
                    # Check job completion
                    job_steps = {}
                    for entry in all_schedules:
                        job = entry.get('job')
                        if job not in job_steps:
                            job_steps[job] = []
                        job_steps[job].append(entry)
                    
                    for job in jobs:
                        job_name = job['name']
                        if job_name not in job_steps or len(job_steps[job_name]) != len(job.get('steps', [])):
                            errors3.append(f"Incomplete schedule for {job_name}")
                    
                    # Check machine constraints
                    machine_schedules = {name: [] for name in machine_names}
                    for entry in all_schedules:
                        machine = entry.get('machine')
                        if machine in machine_schedules:
                            machine_schedules[machine].append(entry)
                    
                    for machine, schedule in machine_schedules.items():
                        schedule.sort(key=lambda x: x.get('start', 0))
                        for i in range(len(schedule)-1):
                            if schedule[i].get('end', 0) > schedule[i+1].get('start', 0):
                                errors3.append(f"Overlap detected on {machine}: {schedule[i].get('job')} Step {schedule[i].get('step')} and {schedule[i+1].get('job')} Step {schedule[i+1].get('step')}")
                    
                    # Check precedence constraints
                    for job, steps in job_steps.items():
                        steps.sort(key=lambda x: x.get('step', 0))
                        for i in range(len(steps)-1):
                            if steps[i].get('end', 0) > steps[i+1].get('start', 0):
                                errors3.append(f"Precedence violation in {job}: Step {steps[i].get('step')} ends after Step {steps[i+1].get('step')} starts")
                
                print("\n=== Validation Results (Final Attempt) ===")
                if errors3:
                    print("❌ Validation failed after global replanning. Stopping MAPLE.")
                    for error in errors3:
                        print(f"- {error}")
                    return False
                else:
                    print("✅ Schedule is valid after global replanning!")
                    makespan3 = max(entry.get('end', 0) for entry in all_schedules) if all_schedules else None
                    print(f"Final Makespan: {makespan3}")
                    return True
            else:
                print("✅ Schedule is valid after second attempt!")
                print(f"Final Makespan: {makespan2}")
                return True
        else:
            print("✅ Schedule is valid!")
            print(f"Final Makespan: {makespan}")
            return True

# Main MAPLE Orchestrator
class MAPLE:
    """
    General-purpose planning framework with three-layer architecture:
    1. Specification Construction
    2. Inter-Agent Coordination
    3. Execution, Rollback, and Self-Validation
    Supports static and dynamic tasks.
    """
    def __init__(self, task_spec):
        self.workflow = WorkflowSpecification(task_spec)
        self.coordinator = InterAgentCoordinator(self.workflow.get_nodes(), self.workflow.get_edges())
        self.executor = ExecutionManager(self.coordinator.agents)
        self.adaptation_manager = DynamicAdaptationManager(self.workflow, self.coordinator, self.executor)

    def LCSR_replanning_and_optimized(self): # with disruptions
        """
        Implements Local Reactive Compensation Protocol (LRCP) as in Algorithm 2.
        This function should be called before self-validation.
        """
        print("\n=== [MAPLE] Local Reactive Compensation Protocol (LRCP) ===")
        W_exec_completed = False
        agent_status = {agent.name: 'active' for agent in self.coordinator.agents}
        disruptions = getattr(self, 'disruptions', set())  # Placeholder: set of disruptions
        persistent_log = {}
        queue_reordered = False
        max_iterations = 100
        iteration = 0
        
        # For demonstration, we assume disruptions is empty unless set externally
        while not W_exec_completed:
            iteration += 1
            print(f"\n[LRCP] Iteration {iteration}")
            if iteration > max_iterations:
                print("[LRCP] Max iterations reached, breaking to avoid infinite loop.")
                break
            W_exec_completed = True  # Will set to False if any agent is still active
            for agent in self.coordinator.agents:
                print(f"  Agent {agent.name} status: {agent_status[agent.name]}")
                if agent_status[agent.name] != 'active':
                    continue
                W_exec_completed = False
                # 3: if new task assigned to agent
                # (Assume new task if agent.context is empty or not set)
                if not hasattr(agent, 'context') or not agent.context:
                    print(f"    New task assigned to {agent.name}, running agent...")
                    # 4: Execute role; update persistent log
                    try:
                        result = agent.run()
                        agent.context = result
                        persistent_log[agent.name] = result
                        print(f"    {agent.name} executed successfully.")
                    except Exception as e:
                        print(f"❌ ERROR in {agent.name}: {e}")
                        agent_status[agent.name] = 'failed'
                        continue
                # 6: if disruption affects agent
                if disruptions and agent.name in disruptions:
                    print(f"⚡ Disruption detected for {agent.name}")
                    # 7: Activate compensator
                    if hasattr(agent, 'compensate'):
                        agent.compensate()
                    # 8: if temporal delay is feasible (simulate always feasible for now)
                    temporal_delay_feasible = True
                    if temporal_delay_feasible:
                        # 9: Perform local compensation (simulate by re-running agent)
                        print(f"🔄 Local compensation for {agent.name}")
                        try:
                            result = agent.run()
                            agent.context = result
                            persistent_log[agent.name] = result
                            print(f"    Local compensation for {agent.name} succeeded.")
                        except Exception as e:
                            print(f"❌ Compensation failed for {agent.name}: {e}")
                            agent_status[agent.name] = 'failed'
                            continue
                    else:
                        # 10: else if queue reordering improves slack
                        if not queue_reordered:
                            print(f"🔃 Applying dynamic queue reordering for {agent.name}")
                            # Simulate queue reordering by shuffling agent order
                            import random
                            random.shuffle(self.coordinator.agents)
                            queue_reordered = True
                        else:
                            # 12: else estimate reallocation cost (simulate cost always below threshold)
                            t_wip = 1  # Simulated cost
                            threshold = 2
                            if t_wip <= threshold:
                                print(f"🔁 Reassigning task locally for {agent.name}")
                                try:
                                    result = agent.run()
                                    agent.context = result
                                    persistent_log[agent.name] = result
                                    print(f"    Local reassignment for {agent.name} succeeded.")
                                except Exception as e:
                                    print(f"❌ Local reassignment failed for {agent.name}: {e}")
                                    agent_status[agent.name] = 'failed'
                                    continue
                            else:
                                print(f"⛔ Terminating and returning updated W_exec for {agent.name}")
                                agent_status[agent.name] = 'terminated'
                                continue
            # End for each agent
            # If all agents are not active, break
            if all(status != 'active' for status in agent_status.values()):
                print("[LRCP] All agents are inactive. Breaking loop.")
                break
        print("=== LRCP protocol completed ===")

    def run(self, with_rollback=True, validate=True):
        print("\n=== [MAPLE] Specification Construction ===")
        print(f"Nodes: {[n['agent'].name for n in self.workflow.get_nodes()]}")
        print(f"Edges: {self.workflow.get_edges()}")
        print("\n=== [MAPLE] Inter-Agent Coordination ===")
        self.coordinator._set_agent_dependencies()
        print("\n=== [MAPLE] Execution & Validation ===")
        self.executor.execute(with_rollback=with_rollback, adaptation_manager=self.adaptation_manager)
        # Call LRCP protocol before self-validation
        self.LCSR_replanning_and_optimized()
        if validate:
            self.executor.self_validate() 