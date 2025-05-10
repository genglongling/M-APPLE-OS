import pandas as pd
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model
import os
import argparse
import csv

def load_dmu_dataset(filepath):
    jobs = []
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        n_jobs, n_machines = map(int, lines[0].split())
        for job_idx, line in enumerate(lines[1:]):
            tokens = list(map(int, line.split()))
            steps = [(f"Machine{machine}", duration) for machine, duration in zip(tokens[::2], tokens[1::2])]
            jobs.append({'name': f'Job{job_idx+1}', 'steps': steps})
    return jobs, n_jobs, n_machines

def load_schedule(filepath):
    schedule = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule.append({
                'job': row['job'],
                'step': int(row['step'].split(' ')[1]) if ' ' in row['step'] else int(row['step']),
                'machine': row['machine'],
                'start': int(row['start']),
                'end': int(row['end'])
            })
    return schedule

def write_schedule(schedule, output_file):
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['job', 'step', 'machine', 'start', 'end', 'precedence'])
        writer.writeheader()
        for entry in schedule:
            writer.writerow(entry)

def lrcp_optimize(jobs, schedule, n_jobs, n_machines):
    """
    A simpler LRCP optimization that tries to move each job one unit ahead while respecting constraints.
    Returns the best schedule found.
    """
    # Group operations by job for easier processing
    job_operations = {}
    for entry in schedule:
        job = entry['job']
        if job not in job_operations:
            job_operations[job] = []
        job_operations[job].append(entry)
    
    # Sort operations within each job by step
    for job in job_operations:
        job_operations[job].sort(key=lambda x: x['step'])
    
    # Try to move each job one unit ahead
    best_schedule = schedule.copy()
    best_makespan = max(op['end'] for op in schedule)
    
    for job in job_operations:
        # Create a copy of the current schedule
        current_schedule = best_schedule.copy()
        
        # Try to move this job's operations one unit earlier
        can_move = True
        for op in job_operations[job]:
            # Find the operation in current_schedule
            for current_op in current_schedule:
                if (current_op['job'] == op['job'] and 
                    current_op['step'] == op['step']):
                    # Try to move one unit earlier
                    new_start = current_op['start'] - 1
                    new_end = current_op['end'] - 1
                    
                    # Check precedence constraints
                    if op['step'] > 1:
                        prev_op = next(o for o in current_schedule 
                                     if o['job'] == op['job'] and o['step'] == op['step'] - 1)
                        if new_start < prev_op['end']:
                            can_move = False
                            break
                    
                    # Check machine constraints
                    for other_op in current_schedule:
                        if other_op['machine'] == op['machine'] and other_op['job'] != op['job']:
                            if not (new_end <= other_op['start'] or new_start >= other_op['end']):
                                can_move = False
                                break
                    
                    if can_move:
                        current_op['start'] = new_start
                        current_op['end'] = new_end
                    else:
                        break
            
            if not can_move:
                break
        
        # If we successfully moved the job, check if it improved makespan
        if can_move:
            current_makespan = max(op['end'] for op in current_schedule)
            if current_makespan < best_makespan:
                best_schedule = current_schedule
                best_makespan = current_makespan
    
    return best_schedule

def process_all_schedules(model_name, dmu_dir='applications/DMU'):
    # Get all schedule files for the model
    schedule_dir = f'results_baselines/{model_name}'
    if not os.path.exists(schedule_dir):
        print(f"Error: Schedule directory {schedule_dir} does not exist")
        return
    
    # Create output directory for optimized schedules
    output_dir = f'results_baselines/{model_name}-optimized'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Process each schedule file
    for filename in os.listdir(schedule_dir):
        if filename.endswith('_5.csv') and not filename.startswith('meta_'):
            print(f"\nProcessing {filename}...")
            
            # Extract dataset name from filename
            # Format: rcmax_20_15_5_claude-3.7-sonnet-sim1_5.csv
            parts = filename.split('_')
            if len(parts) >= 4:
                dataset_name = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
            else:
                print(f"Error: Could not extract dataset name from {filename}")
                continue
            
            # Load DMU data
            dmu_file = os.path.join(dmu_dir, f'{dataset_name}.txt')
            if not os.path.exists(dmu_file):
                print(f"Error: DMU file {dmu_file} does not exist")
                continue
            
            # Load data
            jobs, n_jobs, n_machines = load_dmu_dataset(dmu_file)
            schedule = load_schedule(os.path.join(schedule_dir, filename))
            
            # Optimize schedule
            optimized_schedule = lrcp_optimize(jobs, schedule, n_jobs, n_machines)
            
            # Write optimized schedule to new directory
            output_file = os.path.join(output_dir, filename)
            write_schedule(optimized_schedule, output_file)
            print(f"Optimized schedule written to {output_file}")
            
            # Also create meta file
            meta_file = os.path.join(output_dir, f'meta_{dataset_name}_{model_name}.csv')
            with open(meta_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Dataset', 'Algorithm', 'Iteration', 'Makespan'])
                writer.writerow([dataset_name, 'LRCP', 1, max(op['end'] for op in optimized_schedule)])
            print(f"Meta file written to {meta_file}")

def main():
    parser = argparse.ArgumentParser(description='Optimize JSSP schedules using LRCP.')
    parser.add_argument('--model_name', type=str, help='Name of the model to process')
    parser.add_argument('--dmu_dir', type=str, default='applications/DMU', help='Directory containing DMU files')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    args = parser.parse_args()
    
    if args.model_name:
        process_all_schedules(args.model_name, args.dmu_dir)
    else:
        print("Error: --model_name is required")

if __name__ == '__main__':
    main()
