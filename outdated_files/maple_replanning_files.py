import os
import csv
import sys
from typing import List, Dict, Tuple

# Add src directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

class JSSPAgent(Agent):
    def __init__(self, name: str, operation: Dict):
        super().__init__(
            name=name,
            backstory=f"Agent responsible for optimizing operation {operation['job']} Step {operation['step']}",
            task_description=f"Optimize the schedule for {operation['job']} Step {operation['step']} on {operation['machine']}",
            task_expected_output="Optimized start and end times for the operation"
        )
        self.operation = operation

    def run(self):
        # Parse context string into dictionary if it's a string
        if isinstance(self.context, str):
            try:
                # Try to parse as JSON first
                import json
                schedule = json.loads(self.context)
            except:
                # If not JSON, try to parse as a list of operations
                schedule = []
                lines = self.context.strip().split('\n')
                for line in lines:
                    if ',' in line:
                        parts = line.split(',')
                        if len(parts) >= 5:  # job,step,machine,start,end,precedence
                            schedule.append({
                                'job': parts[0],
                                'step': int(parts[1]),
                                'machine': parts[2],
                                'start': int(parts[3]),
                                'end': int(parts[4]),
                                'precedence': parts[5] if len(parts) > 5 else ''
                            })
        else:
            schedule = self.context.get('schedule', []) if isinstance(self.context, dict) else []

        if not schedule:
            return {'start': self.operation['start'], 'end': self.operation['end']}

        # Try to optimize this operation's timing
        best_start = self.operation['start']
        best_end = self.operation['end']
        best_makespan = max(op['end'] for op in schedule)

        # Find earliest possible start time based on precedence
        earliest_start = 0
        if self.operation['precedence']:
            pred_job, pred_step = self.operation['precedence'].split("After ")[1].split(" Step ")
            pred = next(op for op in schedule 
                       if op['job'] == pred_job and op['step'] == int(pred_step))
            earliest_start = pred['end']

        # Try different start times
        current_start = earliest_start
        while current_start < best_start:
            # Check if this start time is feasible
            feasible = True
            for op in schedule:
                if op['machine'] == self.operation['machine']:
                    if not (current_start >= op['end'] or 
                           current_start + (self.operation['end'] - self.operation['start']) <= op['start']):
                        feasible = False
                        break
            
            if feasible:
                # Calculate new makespan
                new_schedule = schedule.copy()
                new_schedule.append({
                    'job': self.operation['job'],
                    'step': self.operation['step'],
                    'machine': self.operation['machine'],
                    'start': current_start,
                    'end': current_start + (self.operation['end'] - self.operation['start']),
                    'precedence': self.operation['precedence']
                })
                new_makespan = max(op['end'] for op in new_schedule)
                
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best_start = current_start
                    best_end = current_start + (self.operation['end'] - self.operation['start'])
            
            current_start += 1

        return {'start': best_start, 'end': best_end}

class SupervisorAgent(Agent):
    def __init__(self, name: str, operations: List[Dict]):
        super().__init__(
            name=name,
            backstory="Supervisor agent that coordinates all operations to find the minimum makespan solution",
            task_description="""Find the minimum makespan schedule for all operations while strictly following these rules:
1. Each job's steps must be completed in strict order (e.g., Job1's step 2 can only start after step 1 is completed)
2. Each machine can only process one operation at a time
3. The goal is to minimize the total completion time (makespan) while ensuring all operations are completed""",
            task_expected_output="A complete schedule with minimum makespan that satisfies all constraints"
        )
        self.operations = operations

    def run(self):
        # Get optimized times from all operation agents
        optimized_times = {}
        for agent in self.dependencies:
            if hasattr(agent, 'operation'):
                result = agent.context
                if isinstance(result, dict) and 'start' in result and 'end' in result:
                    optimized_times[(agent.operation['job'], agent.operation['step'])] = {
                        'start': result['start'],
                        'end': result['end']
                    }

        # Create final schedule
        final_schedule = []
        for op in self.operations:
            job_step = (op['job'], op['step'])
            if job_step in optimized_times:
                times = optimized_times[job_step]
                final_schedule.append({
                    'job': op['job'],
                    'step': op['step'],
                    'machine': op['machine'],
                    'start': times['start'],
                    'end': times['end'],
                    'precedence': op['precedence']
                })
            else:
                final_schedule.append(op)

        # Calculate makespan
        makespan = max(op['end'] for op in final_schedule)
        print(f"Final makespan: {makespan}")

        return {'schedule': final_schedule}

def read_schedule_file(file_path: str) -> List[Dict]:
    """Read a schedule file and return list of operations"""
    operations = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            operations.append({
                'job': row['job'],
                'step': int(row['step']),
                'machine': row['machine'],
                'start': int(row['start']),
                'end': int(row['end']),
                'precedence': row['precedence']
            })
    return operations

def write_schedule_file(operations: List[Dict], output_path: str):
    """Write operations to a schedule file"""
    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['job', 'step', 'machine', 'start', 'end', 'precedence'])
        for op in operations:
            writer.writerow([op['job'], op['step'], op['machine'], op['start'], op['end'], op['precedence']])

def main():
    # Create output directory if it doesn't exist
    output_dir = 'results_baselines_optimized'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Read schedule files from claude-3.7-sonnet-sim1
    input_dir = 'results_baselines/claude-3.7-sonnet-sim1'
    schedule_files = [f for f in os.listdir(input_dir) if f.endswith('_5.csv') and not f.startswith('meta_')]

    for schedule_file in schedule_files:
        print(f"\nProcessing {schedule_file}...")
        
        # Read operations from schedule file
        operations = read_schedule_file(os.path.join(input_dir, schedule_file))
        
        # Create agents for each operation
        operation_agents = []
        for op in operations:
            agent_name = f"{op['job']}_Step{op['step']}_Agent"
            agent = JSSPAgent(name=agent_name, operation=op)
            operation_agents.append(agent)
        
        # Create supervisor agent
        supervisor = SupervisorAgent(
            name="SupervisorAgent",
            operations=operations
        )
        
        # Set up dependencies
        for agent in operation_agents:
            supervisor.add_dependency(agent)
        
        # Create task specification for MAPLE
        task_spec = {
            'nodes': [
                {'agent': agent, 'dependencies': []} for agent in operation_agents
            ] + [{'agent': supervisor, 'dependencies': [agent.name for agent in operation_agents]}],
            'edges': [],
            'jobs': operations,
            'disruptions': [],
            'rules': [
                'Each job must perform its steps strictly in order.',
                'Each machine can only handle one operation at a time.',
                'No two operations use the same machine at the same time.'
            ]
        }
        
        # Initialize and run MAPLE
        maple = MAPLE(task_spec)
        maple.run(with_rollback=True, validate=True)
        
        # Get optimized schedule from supervisor
        optimized_operations = supervisor.context.get('schedule', operations)
        
        # Write optimized schedule to output directory
        output_file = os.path.join(output_dir, schedule_file)
        write_schedule_file(optimized_operations, output_file)
        
        # Calculate and print makespan
        makespan = max(op['end'] for op in optimized_operations)
        print(f"Optimized makespan for {schedule_file}: {makespan}")

if __name__ == "__main__":
    main() 