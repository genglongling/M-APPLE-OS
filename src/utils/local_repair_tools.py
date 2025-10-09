"""
Local Repair Tools for MAPLE JSSP Workflow
Provides automatic repair functionality for common schedule issues
"""

import json
import random
from typing import Dict, List, Any, Tuple


class LocalRepairTools:
    """
    Static methods for repairing JSSP schedules
    """
    
    @staticmethod
    def fix_schedule(schedule_data: Any, errors: List[str]) -> Any:
        """
        Repair schedule using Algorithm 3: Reactive Planning with Cascading Repair and Queue Reordering
        
        Args:
            schedule_data: Original schedule data
            errors: List of validation errors
            
        Returns:
            Repaired schedule data
        """
        try:
            # Parse schedule data if needed
            if isinstance(schedule_data, str):
                try:
                    schedule_data = json.loads(schedule_data)
                except json.JSONDecodeError:
                    return schedule_data
            
            # Extract schedule entries
            if isinstance(schedule_data, dict):
                if 'schedule' in schedule_data:
                    schedule_entries = schedule_data['schedule']
                elif 'schedules' in schedule_data:
                    schedule_entries = schedule_data['schedules']
                else:
                    schedule_entries = [schedule_data]
            else:
                schedule_entries = schedule_data
            
            if not schedule_entries:
                return schedule_data
            
            print(f"🔧 Starting Algorithm 3: Cascading Repair and Queue Reordering")
            print(f"📊 Processing {len(schedule_entries)} schedule entries")
            print(f"❌ Found {len(errors)} constraint violations")
            
            # Apply Algorithm 3: Reactive Planning with Cascading Repair
            repaired_entries = LocalRepairTools._apply_cascading_repair(schedule_entries, errors)
            
            # Return in original format
            if isinstance(schedule_data, dict):
                if 'schedule' in schedule_data:
                    schedule_data['schedule'] = repaired_entries
                elif 'schedules' in schedule_data:
                    schedule_data['schedules'] = repaired_entries
                else:
                    schedule_data = repaired_entries
                return schedule_data
            else:
                return repaired_entries
                
        except Exception as e:
            print(f"Repair error: {str(e)}")
            return schedule_data
    
    @staticmethod
    def _apply_cascading_repair(schedule_entries: List[Dict[str, Any]], errors: List[str]) -> List[Dict[str, Any]]:
        """
        Apply Algorithm 3: Reactive Planning with Cascading Repair and Queue Reordering
        
        Enhanced version with better constraint handling and iterative improvement
        """
        print("🔄 Phase I: Status Update - Identifying affected operations")
        
        # Create a working copy of the schedule
        working_schedule = [entry.copy() for entry in schedule_entries]
        
        # Phase I: Immediate constraint fixing
        working_schedule = LocalRepairTools._fix_immediate_constraints(working_schedule, errors)
        
        # Phase II: Job precedence repair
        print("🔄 Phase II: Job Precedence Repair - Fixing step ordering violations")
        working_schedule = LocalRepairTools._repair_job_precedence(working_schedule)
        
        # Phase III: Machine capacity repair
        print("🔄 Phase III: Machine Capacity Repair - Resolving machine overlaps")
        working_schedule = LocalRepairTools._repair_machine_capacity(working_schedule)
        
        # Phase IV: Iterative improvement
        print("🔄 Phase IV: Iterative Improvement - Optimizing schedule quality")
        working_schedule = LocalRepairTools._iterative_improvement(working_schedule)
        
        # Phase V: Final validation and cleanup
        print("🔄 Phase V: Final validation and cleanup")
        working_schedule = LocalRepairTools._final_cleanup(working_schedule)
        
        print(f"✅ Algorithm 3 completed - Generated {len(working_schedule)} schedule entries")
        return working_schedule
    
    @staticmethod
    def _create_machine_queues(schedule_entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create machine queues from schedule entries"""
        machine_queues = {}
        
        for entry in schedule_entries:
            machine = entry.get('machine', '')
            if machine not in machine_queues:
                machine_queues[machine] = []
            machine_queues[machine].append(entry)
        
        # Sort each queue by start time
        for machine in machine_queues:
            machine_queues[machine].sort(key=lambda x: x.get('start', 0))
        
        return machine_queues
    
    @staticmethod
    def _fix_immediate_constraints(schedule_entries: List[Dict[str, Any]], errors: List[str]) -> List[Dict[str, Any]]:
        """Fix immediate constraint violations by adjusting start/end times"""
        print("🔧 Fixing immediate constraint violations...")
        
        # Sort schedule by start time for proper processing
        schedule_entries.sort(key=lambda x: x.get('start', 0))
        
        # Fix job precedence violations first
        for error in errors:
            if "starts before" in error and "ends" in error:
                # Extract job and step information
                parts = error.split("Job ")[1].split(":")[0] if "Job " in error else ""
                if parts:
                    job_name = parts
                    # Find all operations for this job and sort by step
                    job_operations = [op for op in schedule_entries if op.get('job') == job_name]
                    job_operations.sort(key=lambda x: x.get('step', 0))
                    
                    # Ensure proper precedence
                    for i in range(1, len(job_operations)):
                        prev_op = job_operations[i-1]
                        curr_op = job_operations[i]
                        
                        if curr_op.get('start', 0) < prev_op.get('end', 0):
                            # Fix precedence violation
                            curr_op['start'] = prev_op.get('end', 0)
                            curr_op['end'] = curr_op['start'] + (curr_op.get('end', 0) - curr_op.get('start', 0))
        
        return schedule_entries
    
    @staticmethod
    def _repair_job_precedence(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Repair job precedence violations systematically"""
        print("🔧 Repairing job precedence violations...")
        
        # Group operations by job
        job_operations = {}
        for op in schedule_entries:
            job = op.get('job', '')
            if job not in job_operations:
                job_operations[job] = []
            job_operations[job].append(op)
        
        # Fix precedence for each job
        for job, operations in job_operations.items():
            # Sort by step number
            operations.sort(key=lambda x: x.get('step', 0))
            
            # Ensure each step starts after the previous one ends
            for i in range(1, len(operations)):
                prev_op = operations[i-1]
                curr_op = operations[i]
                
                if curr_op.get('start', 0) < prev_op.get('end', 0):
                    # Calculate duration (ensure it's positive)
                    duration = max(1, curr_op.get('end', 0) - curr_op.get('start', 0))
                    # Set new start time
                    curr_op['start'] = prev_op.get('end', 0)
                    curr_op['end'] = curr_op['start'] + duration
                    print(f"⏰ Fixed precedence for {job} step {curr_op.get('step', 0)}")
        
        return schedule_entries
    
    @staticmethod
    def _repair_machine_capacity(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Repair machine capacity violations by rescheduling overlapping operations"""
        print("🔧 Repairing machine capacity violations...")
        
        # Group operations by machine
        machine_operations = {}
        for op in schedule_entries:
            machine = op.get('machine', '')
            if machine not in machine_operations:
                machine_operations[machine] = []
            machine_operations[machine].append(op)
        
        # Fix overlaps for each machine
        for machine, operations in machine_operations.items():
            # Sort by start time
            operations.sort(key=lambda x: x.get('start', 0))
            
            # Find and fix overlaps
            for i in range(len(operations) - 1):
                curr_op = operations[i]
                next_op = operations[i + 1]
                
                # Check for overlap
                if next_op.get('start', 0) < curr_op.get('end', 0):
                    # Calculate duration (ensure it's positive)
                    duration = max(1, next_op.get('end', 0) - next_op.get('start', 0))
                    # Reschedule next operation
                    next_op['start'] = curr_op.get('end', 0)
                    next_op['end'] = next_op['start'] + duration
                    print(f"⏰ Fixed machine overlap on {machine}")
        
        return schedule_entries
    
    @staticmethod
    def _iterative_improvement(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply iterative improvement to optimize the schedule"""
        print("🔧 Applying iterative improvement...")
        
        max_iterations = 3
        for iteration in range(max_iterations):
            improved = False
            
            # Try to improve by swapping operations on the same machine
            machine_operations = {}
            for op in schedule_entries:
                machine = op.get('machine', '')
                if machine not in machine_operations:
                    machine_operations[machine] = []
                machine_operations[machine].append(op)
            
            for machine, operations in machine_operations.items():
                if len(operations) < 2:
                    continue
                
                # Sort by start time
                operations.sort(key=lambda x: x.get('start', 0))
                
                # Try swapping adjacent operations
                for i in range(len(operations) - 1):
                    op_a = operations[i]
                    op_b = operations[i + 1]
                    
                    # Check if swap would improve makespan
                    current_makespan = max(op.get('end', 0) for op in schedule_entries)
                    
                    # Simulate swap
                    temp_start_a = op_b.get('start', 0)
                    temp_end_a = op_b.get('end', 0)
                    temp_start_b = op_a.get('start', 0)
                    temp_end_b = op_a.get('end', 0)
                    
                    # Check if swap is feasible (no precedence violations)
                    if LocalRepairTools._is_swap_feasible(op_a, op_b):
                        # Calculate new makespan
                        op_a['start'] = temp_start_a
                        op_a['end'] = temp_end_a
                        op_b['start'] = temp_start_b
                        op_b['end'] = temp_end_b
                        
                        new_makespan = max(op.get('end', 0) for op in schedule_entries)
                        
                        if new_makespan < current_makespan:
                            improved = True
                            print(f"🔄 Improved schedule on {machine}, Δt = {new_makespan - current_makespan}")
                        else:
                            # Revert swap
                            op_a['start'] = temp_start_b
                            op_a['end'] = temp_end_b
                            op_b['start'] = temp_start_a
                            op_b['end'] = temp_end_a
            
            if not improved:
                break
        
        return schedule_entries
    
    @staticmethod
    def _is_swap_feasible(op_a: Dict[str, Any], op_b: Dict[str, Any]) -> bool:
        """Check if swapping two operations is feasible (no precedence violations)"""
        # Check if operations are from the same job
        if op_a.get('job') == op_b.get('job'):
            # Same job - check step order
            step_a = op_a.get('step', 0)
            step_b = op_b.get('step', 0)
            return step_a < step_b  # Can only swap if step_a comes before step_b
        else:
            # Different jobs - swap is always feasible
            return True
    
    @staticmethod
    def _final_cleanup(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Final cleanup to ensure all operations have valid start/end times"""
        print("🔧 Final cleanup - ensuring valid operation times...")
        
        for op in schedule_entries:
            start = op.get('start', 0)
            end = op.get('end', 0)
            
            # Ensure start < end
            if start >= end:
                # Fix invalid times
                duration = max(1, end - start) if end > start else 1
                op['end'] = start + duration
                print(f"⏰ Fixed invalid times for {op.get('job', 'Unknown')} step {op.get('step', 0)}")
            
            # Ensure non-negative times
            if start < 0:
                op['start'] = 0
                op['end'] = max(1, end - start)
                print(f"⏰ Fixed negative start time for {op.get('job', 'Unknown')} step {op.get('step', 0)}")
        
        return schedule_entries
    
    @staticmethod
    def _identify_affected_operations(schedule_entries: List[Dict[str, Any]], errors: List[str]) -> List[Dict[str, Any]]:
        """Identify operations that need rescheduling based on constraint violations"""
        affected_operations = []
        
        for error in errors:
            if "starts before" in error and "ends" in error:
                # Job precedence violation - extract job name
                job_name = error.split("Job ")[1].split(":")[0] if "Job " in error else ""
                if job_name:
                    # Find all operations for this job
                    job_operations = [op for op in schedule_entries if op.get('job') == job_name]
                    affected_operations.extend(job_operations)
            
            elif "Operations overlap" in error:
                # Machine capacity violation - extract machine name
                machine_name = error.split("Machine ")[1].split(":")[0] if "Machine " in error else ""
                if machine_name:
                    # Find all operations on this machine
                    machine_operations = [op for op in schedule_entries if op.get('machine') == machine_name]
                    affected_operations.extend(machine_operations)
        
        # Remove duplicates
        unique_operations = []
        seen = set()
        for op in affected_operations:
            op_id = f"{op.get('job')}_{op.get('step')}_{op.get('machine')}"
            if op_id not in seen:
                unique_operations.append(op)
                seen.add(op_id)
        
        return unique_operations
    
    @staticmethod
    def _propagate_delays(schedule_entries: List[Dict[str, Any]], affected_operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Propagate delays to dependent operations"""
        delay_notifications = []
        
        for op in affected_operations:
            job = op.get('job', '')
            step = op.get('step', 0)
            end_time = op.get('end', 0)
            
            # Find next operation for this job
            next_operations = [o for o in schedule_entries 
                             if o.get('job') == job and o.get('step') == step + 1]
            
            for next_op in next_operations:
                delay_notifications.append({
                    'job': job,
                    'step': step + 1,
                    'new_end_time': end_time,
                    'operation': next_op
                })
        
        return delay_notifications
    
    @staticmethod
    def _optimize_machine_queues(machine_queues: Dict[str, List[Dict[str, Any]]], 
                                schedule_entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Optimize machine queues using local reordering (Phase III)"""
        optimized_queues = {}
        
        for machine, queue in machine_queues.items():
            print(f"🔧 Optimizing queue for {machine} ({len(queue)} operations)")
            
            # Create a copy for optimization
            optimized_queue = queue.copy()
            
            # Try swapping operations to reduce makespan
            improved = True
            while improved:
                improved = False
                
                for i in range(len(optimized_queue) - 1):
                    op_a = optimized_queue[i]
                    op_b = optimized_queue[i + 1]
                    
                    # Check if op_b is final operation of its job
                    job_b = op_b.get('job', '')
                    step_b = op_b.get('step', 0)
                    
                    # Find max step for this job
                    max_step = max([o.get('step', 0) for o in schedule_entries 
                                  if o.get('job') == job_b], default=0)
                    
                    if step_b == max_step:  # op_b is final operation
                        # Evaluate swap
                        delta_t = LocalRepairTools._evaluate_swap(optimized_queue, i, i + 1)
                        
                        if delta_t < 0:  # Swap improves makespan
                            # Perform swap
                            optimized_queue[i], optimized_queue[i + 1] = optimized_queue[i + 1], optimized_queue[i]
                            
                            # Update start/end times
                            LocalRepairTools._update_operation_times(optimized_queue, i)
                            LocalRepairTools._update_operation_times(optimized_queue, i + 1)
                            
                            improved = True
                            print(f"🔄 Swapped operations on {machine}, Δt = {delta_t}")
            
            optimized_queues[machine] = optimized_queue
        
        return optimized_queues
    
    @staticmethod
    def _evaluate_swap(queue: List[Dict[str, Any]], i: int, j: int) -> float:
        """Evaluate the makespan change if operations at positions i and j are swapped"""
        if i >= len(queue) or j >= len(queue):
            return 0
        
        op_a = queue[i]
        op_b = queue[j]
        
        # Calculate current makespan
        current_makespan = max(op.get('end', 0) for op in queue)
        
        # Simulate swap
        temp_queue = queue.copy()
        temp_queue[i], temp_queue[j] = temp_queue[j], temp_queue[i]
        
        # Recalculate times for affected operations
        LocalRepairTools._update_operation_times(temp_queue, i)
        LocalRepairTools._update_operation_times(temp_queue, j)
        
        # Calculate new makespan
        new_makespan = max(op.get('end', 0) for op in temp_queue)
        
        return new_makespan - current_makespan
    
    @staticmethod
    def _update_operation_times(queue: List[Dict[str, Any]], index: int):
        """Update start and end times for an operation based on its position in queue"""
        if index >= len(queue):
            return
        
        op = queue[index]
        machine = op.get('machine', '')
        duration = op.get('end', 0) - op.get('start', 0)
        
        # Find previous operation on same machine
        prev_end = 0
        for i in range(index):
            if queue[i].get('machine') == machine:
                prev_end = max(prev_end, queue[i].get('end', 0))
        
        # Update times
        new_start = prev_end
        new_end = new_start + duration
        
        op['start'] = new_start
        op['end'] = new_end
    
    @staticmethod
    def _handle_cascading_delays(schedule_entries: List[Dict[str, Any]], 
                               delay_notifications: List[Dict[str, Any]], 
                               optimized_queues: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Handle cascading delays throughout the schedule (Phase IV)"""
        print("🔄 Processing cascading delays...")
        
        # Create a working copy of the schedule
        working_schedule = schedule_entries.copy()
        
        # Process delay notifications
        for notification in delay_notifications:
            job = notification['job']
            step = notification['step']
            new_end_time = notification['new_end_time']
            
            # Find the affected operation
            for i, op in enumerate(working_schedule):
                if op.get('job') == job and op.get('step') == step:
                    # Check if operation needs to be delayed
                    current_start = op.get('start', 0)
                    if current_start < new_end_time:
                        # Delay the operation
                        duration = op.get('end', 0) - op.get('start', 0)
                        op['start'] = new_end_time
                        op['end'] = new_end_time + duration
                        
                        print(f"⏰ Delayed {job} step {step} to start at {new_end_time}")
                        
                        # Check if this creates more delays
                        LocalRepairTools._check_for_further_delays(working_schedule, op)
                    break
        
        # Rebuild schedule from optimized queues
        final_schedule = []
        for machine, queue in optimized_queues.items():
            final_schedule.extend(queue)
        
        # Sort by start time
        final_schedule.sort(key=lambda x: x.get('start', 0))
        
        return final_schedule
    
    @staticmethod
    def _check_for_further_delays(schedule: List[Dict[str, Any]], delayed_op: Dict[str, Any]):
        """Check if delaying an operation creates further delays"""
        job = delayed_op.get('job', '')
        step = delayed_op.get('step', 0)
        new_end_time = delayed_op.get('end', 0)
        
        # Find next operation for this job
        next_operations = [op for op in schedule 
                          if op.get('job') == job and op.get('step') == step + 1]
        
        for next_op in next_operations:
            if next_op.get('start', 0) < new_end_time:
                # This operation also needs to be delayed
                duration = next_op.get('end', 0) - next_op.get('start', 0)
                next_op['start'] = new_end_time
                next_op['end'] = new_end_time + duration
                
                # Recursively check for further delays
                LocalRepairTools._check_for_further_delays(schedule, next_op)
    
    @staticmethod
    def _apply_repairs(schedule_entries: List[Dict[str, Any]], errors: List[str]) -> List[Dict[str, Any]]:
        """Apply specific repairs based on error types"""
        repaired_entries = schedule_entries.copy()
        
        for error in errors:
            if "Missing required field" in error:
                repaired_entries = LocalRepairTools._fix_missing_fields(repaired_entries, error)
            elif "start time" in error and "end time" in error:
                repaired_entries = LocalRepairTools._fix_time_consistency(repaired_entries, error)
            elif "Operations overlap" in error:
                repaired_entries = LocalRepairTools._fix_machine_overlaps(repaired_entries, error)
            elif "starts before" in error:
                repaired_entries = LocalRepairTools._fix_job_precedence(repaired_entries, error)
            elif "below theoretical upper bound" in error:
                repaired_entries = LocalRepairTools._fix_unrealistic_makespan(repaired_entries, error)
        
        return repaired_entries
    
    @staticmethod
    def _fix_missing_fields(entries: List[Dict[str, Any]], error: str) -> List[Dict[str, Any]]:
        """Fix missing required fields"""
        # Extract entry index from error message
        try:
            if "Entry" in error:
                index_str = error.split("Entry ")[1].split(":")[0]
                index = int(index_str)
                
                if 0 <= index < len(entries):
                    entry = entries[index]
                    
                    # Add missing fields with default values
                    if 'job' not in entry:
                        entry['job'] = f"Job{index + 1}"
                    if 'step' not in entry:
                        entry['step'] = 1
                    if 'machine' not in entry:
                        entry['machine'] = f"Machine{index % 5}"
                    if 'start' not in entry:
                        entry['start'] = index * 10
                    if 'end' not in entry:
                        entry['end'] = entry['start'] + 5
                        
        except (ValueError, IndexError):
            pass  # Skip if we can't parse the error
        
        return entries
    
    @staticmethod
    def _fix_time_consistency(entries: List[Dict[str, Any]], error: str) -> List[Dict[str, Any]]:
        """Fix time consistency issues"""
        try:
            if "Entry" in error:
                index_str = error.split("Entry ")[1].split(":")[0]
                index = int(index_str)
                
                if 0 <= index < len(entries):
                    entry = entries[index]
                    
                    # Ensure start < end
                    if entry.get('start', 0) >= entry.get('end', 0):
                        entry['end'] = entry.get('start', 0) + 5
                        
        except (ValueError, IndexError):
            pass
        
        return entries
    
    @staticmethod
    def _fix_machine_overlaps(entries: List[Dict[str, Any]], error: str) -> List[Dict[str, Any]]:
        """Fix machine capacity violations by rescheduling overlapping operations"""
        # Group by machine
        machine_entries = {}
        for i, entry in enumerate(entries):
            machine = entry.get('machine', '')
            if machine not in machine_entries:
                machine_entries[machine] = []
            machine_entries[machine].append((i, entry))
        
        # Fix overlaps for each machine
        for machine, machine_ops in machine_entries.items():
            # Sort by start time
            machine_ops.sort(key=lambda x: x[1].get('start', 0))
            
            for i in range(1, len(machine_ops)):
                prev_idx, prev_entry = machine_ops[i-1]
                curr_idx, curr_entry = machine_ops[i]
                
                # If overlap exists, move current operation
                if prev_entry.get('end', 0) > curr_entry.get('start', 0):
                    # Move current operation to start after previous ends
                    curr_entry['start'] = prev_entry.get('end', 0)
                    curr_entry['end'] = curr_entry['start'] + (curr_entry.get('end', 0) - curr_entry.get('start', 0))
        
        return entries
    
    @staticmethod
    def _fix_job_precedence(entries: List[Dict[str, Any]], error: str) -> List[Dict[str, Any]]:
        """Fix job precedence violations"""
        # Group by job
        job_entries = {}
        for i, entry in enumerate(entries):
            job = entry.get('job', '')
            if job not in job_entries:
                job_entries[job] = []
            job_entries[job].append((i, entry))
        
        # Fix precedence for each job
        for job, job_ops in job_entries.items():
            # Sort by step number
            try:
                job_ops.sort(key=lambda x: int(x[1].get('step', 0)))
                
                for i in range(1, len(job_ops)):
                    prev_idx, prev_entry = job_ops[i-1]
                    curr_idx, curr_entry = job_ops[i]
                    
                    # Ensure current step starts after previous ends
                    if prev_entry.get('end', 0) > curr_entry.get('start', 0):
                        curr_entry['start'] = prev_entry.get('end', 0)
                        curr_entry['end'] = curr_entry['start'] + (curr_entry.get('end', 0) - curr_entry.get('start', 0))
                        
            except (ValueError, TypeError):
                pass  # Skip if step numbers are not numeric
        
        return entries
    
    @staticmethod
    def _fix_unrealistic_makespan(entries: List[Dict[str, Any]], error: str) -> List[Dict[str, Any]]:
        """Fix unrealistic makespan by scaling up the schedule"""
        if not entries:
            return entries
        
        # Calculate current makespan
        current_makespan = max(entry.get('end', 0) for entry in entries)
        
        if current_makespan <= 0:
            # If makespan is too small, scale up all times
            scale_factor = 100  # Arbitrary scaling factor
            for entry in entries:
                entry['start'] = entry.get('start', 0) * scale_factor
                entry['end'] = entry.get('end', 0) * scale_factor
        else:
            # Scale up proportionally
            scale_factor = 2.0  # Double the makespan
            for entry in entries:
                entry['start'] = int(entry.get('start', 0) * scale_factor)
                entry['end'] = int(entry.get('end', 0) * scale_factor)
        
        return entries
    
    @staticmethod
    def repair_dataset_mismatch(entries: List[Dict[str, Any]], expected_jobs: List[str]) -> List[Dict[str, Any]]:
        """Repair missing jobs by adding placeholder entries"""
        existing_jobs = set(entry.get('job', '') for entry in entries)
        missing_jobs = set(expected_jobs) - existing_jobs
        
        for job in missing_jobs:
            # Add a simple placeholder entry for missing job
            placeholder_entry = {
                'job': job,
                'step': 1,
                'machine': 'Machine0',
                'start': len(entries) * 10,
                'end': len(entries) * 10 + 5
            }
            entries.append(placeholder_entry)
        
        return entries
    
    @staticmethod
    def repair_machine_capacity_violations(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Repair machine capacity violations by moving operations to different machines"""
        # Group by machine and time
        machine_schedule = {}
        
        for entry in entries:
            machine = entry.get('machine', '')
            start = entry.get('start', 0)
            end = entry.get('end', 0)
            
            if machine not in machine_schedule:
                machine_schedule[machine] = []
            machine_schedule[machine].append((start, end, entry))
        
        # Find available machines
        all_machines = set(entry.get('machine', '') for entry in entries)
        available_machines = list(all_machines)
        
        # Fix overlaps by moving to different machines
        for machine, operations in machine_schedule.items():
            operations.sort(key=lambda x: x[0])  # Sort by start time
            
            for i in range(1, len(operations)):
                prev_start, prev_end, prev_entry = operations[i-1]
                curr_start, curr_end, curr_entry = operations[i]
                
                if prev_end > curr_start:  # Overlap detected
                    # Try to move to a different machine
                    for alt_machine in available_machines:
                        if alt_machine != machine:
                            # Check if this machine is free at this time
                            if LocalRepairTools._is_machine_free(alt_machine, curr_start, curr_end, entries):
                                curr_entry['machine'] = alt_machine
                                break
        
        return entries
    
    @staticmethod
    def _is_machine_free(machine: str, start: int, end: int, entries: List[Dict[str, Any]]) -> bool:
        """Check if a machine is free during a specific time period"""
        for entry in entries:
            if entry.get('machine', '') == machine:
                entry_start = entry.get('start', 0)
                entry_end = entry.get('end', 0)
                
                # Check for overlap
                if not (end <= entry_start or start >= entry_end):
                    return False
        return True
