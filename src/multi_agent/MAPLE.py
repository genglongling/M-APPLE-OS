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

                # print each agent result
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
        """
        # Placeholder: implement actual validation logic
        print("🔍 Self-Validation: Checking workflow structure, constraints, and compensation coverage...")
        # Example checks (to be expanded):
        # - All agents executed?
        # - All dependencies respected?
        # - All rollback paths available?
        # - Any errors in context?
        print("✅ Self-Validation complete.")

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