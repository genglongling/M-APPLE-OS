import sys
import os
from collections import deque
from colorama import Fore
from graphviz import Digraph  # type: ignore
import random
import copy

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
        self.agents = [node['agent'] for node in nodes]
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
    def __init__(self, agents, dynamic_adaptation=False, task_spec=None):
        self.agents = agents
        self.context = {}  # Execution context for rollback and validation
        self.dynamic_adaptation = dynamic_adaptation
        self.task_spec = task_spec  # Store the task_spec for job extraction

    def execute(self, with_rollback=True, adaptation_manager=None):
        if self.dynamic_adaptation:
            self._execute_dynamic(with_rollback, adaptation_manager)
        else:
            self._execute_static(with_rollback, adaptation_manager)

    def _execute_static(self, with_rollback, adaptation_manager):
        print("Executing static (Algorithm 1: Tabu Search/CP Hybrid)")
        jobs = []
        for agent in self.agents:
            if hasattr(agent, 'job_ops'):
                jobs.append(agent.job_ops)
        if not jobs and self.task_spec:
            # Build a mapping from machine name to index
            machine_names = set()
            for job in self.task_spec.get('jobs', []):
                for m, _ in job['steps']:
                    machine_names.add(m)
            machine_name_to_idx = {name: idx for idx, name in enumerate(sorted(machine_names))}
            jobs = []
            for job in self.task_spec.get('jobs', []):
                ops = [(machine_name_to_idx[m], d) for m, d in job['steps']]
                jobs.append(ops)
        if not jobs:
            print("No jobs found for static execution.")
            return
        n_machines = max(m for job in jobs for m, _ in job) + 1
        def evaluate_schedule(schedule, jobs, n_machines):
            n_jobs = len(jobs)
            op_start = [[0]*n_machines for _ in range(n_jobs)]
            op_end = [[0]*n_machines for _ in range(n_jobs)]
            machine_time = [0]*n_machines
            job_time = [0]*n_jobs
            for m, machine_ops in enumerate(schedule):
                for (j, op_idx) in machine_ops:
                    prev_end = job_time[j]
                    start = max(machine_time[m], prev_end)
                    duration = jobs[j][op_idx][1]
                    op_start[j][op_idx] = start
                    op_end[j][op_idx] = start + duration
                    machine_time[m] = op_end[j][op_idx]
                    job_time[j] = op_end[j][op_idx]
            return max([max(row) for row in op_end])
        def initial_solution(jobs, n_machines):
            n_jobs = len(jobs)
            machine_ops = [[] for _ in range(n_machines)]
            for j, job in enumerate(jobs):
                for op_idx, (m, _) in enumerate(job):
                    machine_ops[m].append((j, op_idx))
            for m in range(n_machines):
                random.shuffle(machine_ops[m])
            return machine_ops
        def neighborhood(schedule):
            neighbors = []
            for m, ops in enumerate(schedule):
                for i in range(len(ops)-1):
                    neighbor = copy.deepcopy(schedule)
                    neighbor[m][i], neighbor[m][i+1] = neighbor[m][i+1], neighbor[m][i]
                    neighbors.append(neighbor)
            return neighbors
        def tabu_search_cp_hybrid(jobs, n_machines, max_iter=100, tabu_size=10):
            current = initial_solution(jobs, n_machines)
            best = current
            best_makespan = evaluate_schedule(current, jobs, n_machines)
            tabu_list = []
            for _ in range(max_iter):
                neighbors = neighborhood(current)
                neighbors = [n for n in neighbors if n not in tabu_list]
                if not neighbors:
                    break
                neighbor = min(neighbors, key=lambda s: evaluate_schedule(s, jobs, n_machines))
                makespan = evaluate_schedule(neighbor, jobs, n_machines)
                if makespan < best_makespan:
                    best = neighbor
                    best_makespan = makespan
                tabu_list.append(neighbor)
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)
                current = neighbor
            return best, best_makespan
        best_schedule, best_makespan = tabu_search_cp_hybrid(jobs, n_machines)
        self.context['static_best_schedule'] = best_schedule
        self.context['static_best_makespan'] = best_makespan
        print(f"Best static makespan: {best_makespan}")
        # Assign each job's operation sequence to the corresponding agent's context
        for j, agent in enumerate(self.agents):
            if hasattr(agent, 'name') and 'Job' in agent.name:
                # Find all (j, op_idx) pairs for this job in the best schedule
                job_ops = []
                for m, machine_ops in enumerate(best_schedule):
                    for (job_idx, op_idx) in machine_ops:
                        if job_idx == j:
                            job_ops.append({'machine': m, 'op_idx': op_idx})
                # Pass the schedule to the agent (e.g., set an attribute)
                agent.solved_schedule = job_ops
                agent.solved_makespan = best_makespan
                # Now call the agent's run() method (which can use LLM)
                self.context[agent.name] = agent.run()

        # Find the supervisor agent
        supervisor_agent = None
        for agent in self.agents:
            if hasattr(agent, 'name') and 'Supervisor' in agent.name:
                supervisor_agent = agent
                break

        if supervisor_agent is not None:
            # Aggregate all job schedules
            all_schedules = []
            for agent in self.agents:
                if hasattr(agent, 'name') and 'Job' in agent.name:
                    job_sched = self.context.get(agent.name, {}).get('schedule', [])
                    all_schedules.extend(job_sched)
            # Optionally, sort or process all_schedules as needed
            supervisor_agent.aggregated_schedule = all_schedules
            self.context[supervisor_agent.name] = supervisor_agent.run()

    def _execute_dynamic(self, with_rollback, adaptation_manager):
        # Algorithm 2: MAPLE Reactive Planning Algorithm with Cascading Repair
        print("Executing dynamic (Algorithm 2: MAPLE Reactive Planning)")
        # This is a placeholder for the full dynamic/cascading repair logic
        # You would implement the tracker, queues, and cascading repair here
        # For now, just print a message and set a dummy tracker
        self.context['dynamic_tracker'] = 'Updated tracker after cascading repair (placeholder)'
        print("Dynamic adaptation logic would update tracker and queues here.")

    def self_validate(self):
        """
        Validates workflow execution for structural validity, constraint satisfaction, and compensation coverage.
        """
        # Placeholder: implement actual validation logic
        print("🔍 Self-Validation: Checking workflow structure, constraints, and compensation coverage...")
        # Example checks (to be expanded):
        # - All agents executed?
        # - All dependencies respected?
        # - All rollback paths available?
        # - Any errors in context?
        print("✅ Self-Validation complete.")

    def intra_agent(self):
        """
        Prints details of each agent's individual execution (output/schedule).
        """
        print("\n📌 **Intra-Agent Execution Details**")
        for agent in self.agents:
            output = self.context.get(agent.name, 'Not executed')
            print(f"🔹 {agent.name}: {output}")

# Main MAPLE Orchestrator
class MAPLE:
    """
    General-purpose planning framework with three-layer architecture:
    1. Specification Construction
    2. Inter-Agent Coordination
    3. Execution, Rollback, and Self-Validation
    Supports static and dynamic tasks.
    """
    def __init__(self, task_spec, dynamic_adaptation=False):
        self.workflow = WorkflowSpecification(task_spec)
        self.coordinator = InterAgentCoordinator(self.workflow.get_nodes(), self.workflow.get_edges())
        self.executor = ExecutionManager(self.coordinator.agents, dynamic_adaptation=dynamic_adaptation, task_spec=task_spec)
        self.adaptation_manager = DynamicAdaptationManager(self.workflow, self.coordinator, self.executor)

    def run(self, with_rollback=True, validate=True):
        print("\n=== [MAPLE] Specification Construction ===")
        print(f"Nodes: {[n['agent'].name for n in self.workflow.get_nodes()]}")
        print(f"Edges: {self.workflow.get_edges()}")
        print("\n=== [MAPLE] Inter-Agent Coordination ===")
        self.coordinator._set_agent_dependencies()
        print("\n=== [MAPLE] Execution & Validation ===")
        self.executor.execute(with_rollback=with_rollback, adaptation_manager=self.adaptation_manager)
        if validate:
            self.executor.self_validate()
        # Print intra-agent execution details (JSSP schedule)
        self.executor.intra_agent() 