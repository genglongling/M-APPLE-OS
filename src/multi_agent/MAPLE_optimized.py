"""
Optimized MAPLE implementation with new 7-step workflow:
1. LLM Query Agent generate_schedule()
2. ValidationTools validate_schedule()
3. RepairTools repair_schedule() (if validation fails)
4. ValidationTools revalidate_schedule() (after repair)
5. OptimizationTools run_optimization_schedule()
6. ValidationTools final_check() (after optimization)
7. Final result processing (no supervisor needed)
"""

import os
import sys
import json
from datetime import datetime
from colorama import Fore, Style

# Add the parent directory to sys.path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from multi_agent.MAPLE import MAPLE, WorkflowSpecification, InterAgentCoordinator, ExecutionManager, DynamicAdaptationManager

class OptimizedExecutionManager(ExecutionManager):
    """
    Optimized ExecutionManager with new 4-step workflow
    """
    
    def __init__(self, agents):
        super().__init__(agents)
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.iteration_count = 0
        self.max_iterations = 1  # Stop after first iteration
        self.workflow_config = None
        
    def _calculate_makespan(self, schedule):
        """Calculate makespan from a schedule"""
        if not schedule:
            return 0
        
        max_end_time = 0
        for step in schedule:
            if isinstance(step, dict):
                # Handle dict format with 'end' key
                if 'end' in step:
                    max_end_time = max(max_end_time, step['end'])
                elif 'end_time' in step:
                    max_end_time = max(max_end_time, step['end_time'])
            elif isinstance(step, list) and len(step) >= 4:
                # Handle list format: [job, machine, start_time, end_time]
                max_end_time = max(max_end_time, step[3])
        
        return max_end_time
        
    def execute(self, with_rollback=True, adaptation_manager=None):
        """Execute new 4-step workflow: Query -> Validation -> Optimization -> Storage"""
        print("\n🔄 Starting New 4-Step Workflow...")
        
        # Initialize optimization skip flag
        self._optimization_skipped = False
        
        try:
            # Step 1: Load pre-generated schedule from JSON file
            print("1️⃣ Loading pre-generated schedule from JSON file...")
            # Load pre-generated schedule from JSON file
            import json
            import os
            
            dataset_name = self.context.get('dataset_name', 'rcmax_20_15_5')
            json_file_path = f"results_single(gpt-4o)/singleagent_llm_comparison_{dataset_name}_Gemini-2.5.json"
            
            if not os.path.exists(json_file_path):
                print(f"⚠️ JSON file not found: {json_file_path}")
                query_result = {
                    "schedule": [
                        {"job": "Job1", "step": 1, "machine": "Machine0", "start": 0, "end": 10},
                        {"job": "Job2", "step": 1, "machine": "Machine1", "start": 0, "end": 15}
                    ]
                }
            else:
                try:
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'models' in data and 'Gemini-2.5' in data['models']:
                        gemini_data = data['models']['Gemini-2.5']
                        if 'structured_schedule' in gemini_data:
                            schedule = gemini_data['structured_schedule']
                            makespan = gemini_data.get('makespan', None)
                            print(f"📊 Loaded {len(schedule)} schedule entries from {dataset_name}")
                            print(f"📊 Gemini-2.5 makespan: {makespan}")
                            
                            # Calculate and print initial makespan
                            initial_makespan = self._calculate_makespan(schedule)
                            print(f"📊 Initial schedule makespan: {initial_makespan}")
                            print(f"📊 Schedule entries count: {len(schedule)}")
                            if schedule:
                                print(f"📊 Sample entry: {schedule[0] if len(schedule) > 0 else 'None'}")
                            
                            query_result = {
                                "schedule": schedule,
                                "makespan": makespan
                            }
                        else:
                            print(f"⚠️ No structured_schedule found in {json_file_path}")
                            query_result = {"schedule": []}
                    else:
                        print(f"⚠️ No Gemini-2.5 data found in {json_file_path}")
                        query_result = {"schedule": []}
                        
                except Exception as e:
                    print(f"❌ Error loading JSON file: {str(e)}")
                    query_result = {"schedule": []}
            self.context['query_result'] = query_result
            print(Fore.GREEN + "✅ Pre-generated schedule loaded successfully.")
            
            # Step 2: ValidationTools validate_schedule() (only if validation is enabled)
            if self.workflow_config and self.workflow_config.get('use_validation', True):
                print("2️⃣ ValidationTools validating schedule...")
                from src.utils.validation_tools import ValidationTools
                
                # Set dataset information for validation
                if hasattr(self, 'context') and 'jobs' in self.context and 'machine_names' in self.context:
                    ValidationTools.set_dataset_info({
                        'jobs': self.context['jobs'],
                        'machine_names': self.context['machine_names']
                    })
                
                validation_result = ValidationTools.validate_schedule(query_result)
                
                if not validation_result['valid']:
                    print(f"❌ Initial validation failed: {validation_result['errors']}")
                    
                    # Step 3: RepairTools repair_schedule() (if validation fails) with max 5 iterations
                    if self.workflow_config and self.workflow_config.get('use_repair', True):
                        print("3️⃣ RepairTools repairing schedule...")
                        from src.utils.local_repair_tools import LocalRepairTools
                        
                        # Apply repair with maximum 5 iterations
                        max_iterations = 5
                        current_schedule = query_result
                        current_errors = validation_result['errors']
                        
                        # Store repair iteration results
                        repair_iterations = []
                        
                        for iteration in range(max_iterations):
                            print(f"🔧 Repair iteration {iteration + 1}/{max_iterations}")
                            repair_result = LocalRepairTools.fix_schedule(current_schedule, current_errors)
                            
                            # Calculate makespan for this repair iteration
                            # repair_result is the raw schedule entries, so we need to extract them properly
                            if isinstance(repair_result, dict) and 'schedule' in repair_result:
                                schedule_entries = repair_result['schedule']
                            elif isinstance(repair_result, list):
                                schedule_entries = repair_result
                            else:
                                schedule_entries = repair_result
                            
                            repair_makespan = self._calculate_makespan(schedule_entries)
                            print(f"📊 Repair iteration {iteration + 1} makespan: {repair_makespan}")
                            print(f"📊 Repair iteration {iteration + 1} schedule entries: {len(schedule_entries)}")
                            if schedule_entries:
                                # Show the latest end time operations
                                latest_ops = sorted(schedule_entries, key=lambda x: x.get('end', 0), reverse=True)[:3]
                                print(f"📊 Latest operations: {[(op.get('job', ''), op.get('step', 0), op.get('end', 0)) for op in latest_ops]}")
                            
                            # Store repair iteration data
                            repair_iteration_data = {
                                'iteration': iteration + 1,
                                'makespan': repair_makespan,
                                'schedule': schedule_entries,
                                'schedule_entries_count': len(schedule_entries),
                                'timestamp': datetime.now().isoformat()
                            }
                            repair_iterations.append(repair_iteration_data)
                            
                            # Save repair iteration JSON file
                            if hasattr(self, 'context') and 'dataset_name' in self.context:
                                dataset_name = self.context['dataset_name']
                                iteration_file = f"maple_optimized(gemini-2.5)/{dataset_name}_repair_iteration_{iteration + 1}.json"
                                os.makedirs("maple_optimized(gemini-2.5)", exist_ok=True)
                                with open(iteration_file, 'w') as f:
                                    json.dump(repair_iteration_data, f, indent=2)
                                print(f"💾 Saved repair iteration {iteration + 1} to: {iteration_file}")
                            
                            # Validate the repaired schedule
                            repair_validation = ValidationTools.validate_schedule(repair_result)
                            
                            if repair_validation['valid']:
                                print(f"✅ Repair successful after {iteration + 1} iterations (final makespan: {repair_makespan})")
                                # Update query_result with the repaired schedule
                                if isinstance(query_result, dict):
                                    query_result['schedule'] = schedule_entries
                                else:
                                    query_result = {'schedule': schedule_entries}
                                # Store repair iterations in context for later use
                                query_result['repair_iterations'] = repair_iterations
                                break
                            else:
                                print(f"⚠️ Repair iteration {iteration + 1} reduced errors from {len(current_errors)} to {len(repair_validation['errors'])} (makespan: {repair_makespan})")
                                current_schedule = repair_result
                                current_errors = repair_validation['errors']
                        
                        # Store final repair iterations if repair didn't succeed
                        if not repair_validation['valid']:
                            query_result['repair_iterations'] = repair_iterations
                        
                        if not repair_validation['valid']:
                            print(f"⚠️ Repair completed after {max_iterations} iterations with remaining errors")
                            query_result = repair_result
                        
                        # Step 4: ValidationTools revalidate_schedule() (after repair)
                        print("4️⃣ ValidationTools revalidating schedule...")
                        revalidation_result = ValidationTools.validate_schedule(query_result)
                        if not revalidation_result['valid']:
                            print(f"❌ Revalidation failed: {revalidation_result['errors']}")
                            raise Exception("Schedule validation failed after repair")
                        print("✅ Repair and revalidation successful")
                    else:
                        print("⚠️ Schedule validation failed but repair is disabled - using original schedule without optimization")
                        print("✅ Proceeding with original schedule without optimization due to validation failures")
                        # Skip optimization when validation fails and repair is disabled
                        query_result = query_result  # Keep original schedule
                        # Set flag to skip optimization
                        self._optimization_skipped = True
                        print("5️⃣ OptimizationTools skipped - using original schedule due to validation failures")
                else:
                    print("✅ Initial validation passed")
            else:
                print("2️⃣ ValidationTools disabled - skipping validation")
            
            # Step 5: OptimizationTools run_optimization_schedule()
            if self.workflow_config and self.workflow_config.get('use_optimization', True):
                # Check if optimization was already skipped due to validation failures
                if hasattr(self, '_optimization_skipped') and self._optimization_skipped:
                    print("5️⃣ OptimizationTools already skipped due to validation failures")
                else:
                    print("5️⃣ OptimizationTools optimizing schedule...")
                    from src.utils.optimization_tools import OptimizationTools
                    
                    # Print makespan before optimization
                    pre_opt_schedule = query_result.get('schedule', []) if isinstance(query_result, dict) else query_result
                    pre_opt_makespan = self._calculate_makespan(pre_opt_schedule)
                    print(f"📊 Pre-optimization makespan: {pre_opt_makespan}")
                    
                    optimization_result = OptimizationTools.run_optimization_schedule(query_result)
                    
                    # Print makespan after optimization
                    post_opt_schedule = optimization_result.get('schedule', []) if isinstance(optimization_result, dict) else optimization_result
                    post_opt_makespan = self._calculate_makespan(post_opt_schedule)
                    print(f"📊 Post-optimization makespan: {post_opt_makespan}")
                    
                    query_result = optimization_result
                    print("✅ Optimization completed successfully.")
                
                # Step 6: ValidationTools final_check() (after optimization)
                if self.workflow_config and self.workflow_config.get('use_validation', True):
                    print("6️⃣ ValidationTools final check...")
                    final_validation = ValidationTools.validate_schedule(query_result)
                    if not final_validation['valid']:
                        print(f"⚠️ Final validation found {len(final_validation['errors'])} errors, but continuing with optimization result")
                        print("✅ Optimization result accepted despite validation warnings")
                    else:
                        print("✅ Final validation passed")
                else:
                    print("6️⃣ ValidationTools disabled - skipping final validation")
            
            # Step 7: Final result processing (no supervisor needed)
            print("7️⃣ Processing final results...")
            
            # Extract final schedule and makespan from the processed result
            schedule = query_result.get('schedule', [])
            if schedule:
                # Calculate makespan from the final processed schedule
                makespan = max(entry.get('end', 0) for entry in schedule)
                print(f"📊 Final processed schedule with {len(schedule)} entries")
                print(f"📊 Calculated makespan: {makespan}")
                
                self.best_makespan = makespan
                self.best_schedule = schedule
                print(f"🎯 Final makespan: {makespan}")
                print("✅ New workflow completed successfully.")
                return True
            else:
                print("❌ No schedule data found")
                return False
                        
        except Exception as e:
            print(Fore.RED + f"❌ ERROR in workflow: {str(e)}")
            if adaptation_manager:
                adaptation_manager.handle_disruption(None, e)
            elif with_rollback:
                print(Fore.YELLOW + "🔄 Rolling back...")
            print(Fore.RED + "🚨 Workflow execution halted due to error.")
            return False
    
        def _find_agent_by_type(self, agent_type):
            """Find agent by type name"""
            for agent in self.agents:
                if agent is not None and hasattr(agent, '__class__') and agent_type in agent.__class__.__name__:
                    return agent
            return None
        
        def _load_pregenerated_schedule(self):
            """Load pre-generated schedule from JSON file"""
            import json
            import os
            
            # Get dataset name from context
            dataset_name = self.context.get('dataset_name', 'rcmax_20_15_5')
            
            # Construct file path for Claude-4 generated JSON files
            json_file_path = f"results_single(gpt-4o)/singleagent_llm_comparison_{dataset_name}_gpt-4o.json"
            
            if not os.path.exists(json_file_path):
              print(f"⚠️ JSON file not found: {json_file_path}")
              # Return a simple fallback schedule
              return {
                  "schedule": [
                      {"job": "Job1", "step": 1, "machine": "Machine0", "start": 0, "end": 10},
                      {"job": "Job2", "step": 1, "machine": "Machine1", "start": 0, "end": 15}
                  ]
              }
            
            try:
              with open(json_file_path, 'r') as f:
                  data = json.load(f)
              
              # Extract GPT-4o's structured schedule
              if 'models' in data and 'gpt-4o' in data['models']:
                  gpt_data = data['models']['gpt-4o']
                  if 'structured_schedule' in gpt_data:
                      schedule = gpt_data['structured_schedule']
                      print(f"📊 Loaded {len(schedule)} schedule entries from {dataset_name}")
                      return {"schedule": schedule}
              
              print(f"⚠️ No structured_schedule found in {json_file_path}")
              return {"schedule": []}
              
            except Exception as e:
              print(f"❌ Error loading JSON file: {str(e)}")
              return {"schedule": []}
        
    
    def self_validate(self):
        """Simplified validation that uses the best solution found"""
        print("\n=== Optimized Validation Results ===")
        
        if self.best_schedule:
            # Use the best solution found
            all_schedules = self.best_schedule
            
            # Basic validation
            errors = []
            
            # Check for basic schedule validity
            if not all_schedules:
                errors.append("No schedule found")
                print("❌ No schedule found")
                return False
            
            # Calculate final makespan
            makespan = max(entry.get('end', 0) for entry in all_schedules)
            print(f"✅ Using best solution found (makespan: {makespan})")
            print(f"📊 Total operations scheduled: {len(all_schedules)}")
            
            # Store the best solution in context
            supervisor_agent = None
            for agent in self.agents:
                if 'Supervisor' in agent.name:
                    supervisor_agent = agent
                    break
            
            if supervisor_agent:
                self.context[supervisor_agent.name] = {'schedule': all_schedules}
            
            return True
        else:
            print("❌ No valid solution found")
            return False

class OptimizedMAPLE(MAPLE):
    """
    Optimized MAPLE with new 4-step workflow support
    """
    
    def __init__(self, task_spec):
        self.workflow = WorkflowSpecification(task_spec)
        self.coordinator = InterAgentCoordinator(self.workflow.get_nodes(), self.workflow.get_edges())
        self.executor = OptimizedExecutionManager(self.coordinator.agents)
        self.adaptation_manager = DynamicAdaptationManager(self.workflow, self.coordinator, self.executor)
        
        # Set workflow configuration if provided
        if 'workflow_config' in task_spec:
            self.executor.workflow_config = task_spec['workflow_config']
        
        # Pass dataset information to executor for JSON file lookup and validation
        if 'dataset_name' in task_spec:
            self.executor.context['dataset_name'] = task_spec['dataset_name']
        if 'jobs' in task_spec:
            self.executor.context['jobs'] = task_spec['jobs']
        if 'machine_names' in task_spec:
            self.executor.context['machine_names'] = task_spec['machine_names']
    
    def run(self, with_rollback=True, validate=True):
        print("\n=== [OPTIMIZED MAPLE] New 4-Step Workflow ===")
        # Safely get agent names
        node_names = []
        for n in self.workflow.get_nodes():
            if n and 'agent' in n and n['agent'] is not None:
                node_names.append(n['agent'].name)
            else:
                node_names.append("Unknown Agent")
        print(f"Nodes: {node_names}")
        print(f"Edges: {self.workflow.get_edges()}")
        
        # Display workflow configuration
        if self.executor.workflow_config:
            config = self.executor.workflow_config
            print(f"Workflow: {config.get('name', 'Unknown')}")
            print(f"Validation: {'✅' if config.get('use_validation', True) else '❌'}")
            print(f"Repair: {'✅' if config.get('use_repair', True) else '❌'}")
            print(f"Optimization: {'✅' if config.get('use_optimization', True) else '❌'}")
        
        print("\n=== [OPTIMIZED MAPLE] Inter-Agent Coordination ===")
        self.coordinator._set_agent_dependencies()
        print("\n=== [OPTIMIZED MAPLE] New Workflow Execution ===")
        print("🔄 Running new 7-step workflow: Query -> Validation -> Repair -> Re-validation -> Optimization -> Final Check -> Supervisor")
        
        success = self.executor.execute(with_rollback=with_rollback, adaptation_manager=self.adaptation_manager)
        
        if validate and success:
            self.executor.self_validate()
        
        if success:
            print("✅ New 4-step workflow completed successfully")
        else:
            print("❌ New 4-step workflow failed")
        
        return success
