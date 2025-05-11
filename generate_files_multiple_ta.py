import os
import csv
import math
import random
import argparse
import pandas as pd
import numpy as np
import re
from datetime import datetime

def parse_ta_file_with_labels(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Find the first line with at least two integers (robust)
    for i, line in enumerate(lines):
        ints = [int(x) for x in re.findall(r'-?\d+', line)]
        if len(ints) >= 2:
            n_jobs, n_machines = ints[0], ints[1]
            break
    else:
        raise ValueError("Could not find job/machine count line")
    times_idx = next(i for i, l in enumerate(lines) if l.lower().startswith('times'))
    machines_idx = next(i for i, l in enumerate(lines) if l.lower().startswith('machines'))
    times_lines = lines[times_idx+1 : times_idx+1+n_jobs]
    machines_lines = lines[machines_idx+1 : machines_idx+1+n_jobs]
    if len(times_lines) != n_jobs or len(machines_lines) != n_jobs:
        raise ValueError(f"Expected {n_jobs} lines for times and machines, got {len(times_lines)} and {len(machines_lines)}")
    job_operations = []
    for t_line, m_line in zip(times_lines, machines_lines):
        durations = [int(x) for x in t_line.split()]
        machines = [int(x)-1 for x in m_line.split()]  # Convert to 0-based
        if len(durations) != n_machines or len(machines) != n_machines:
            raise ValueError(f"Job line does not match n_machines: {durations}, {machines}")
        operations = list(zip(machines, durations))
        job_operations.append(operations)
    return job_operations, n_jobs, n_machines

def load_ub_dict(csv_path):
    ub_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ub_dict[row['Dataset']] = int(row['Makespan_At_Convergence'])
    return ub_dict

def generate_random_schedule(job_operations, n_jobs, n_machines, ub_makespan, num_runs=20):
    schedules = []
    for run_idx in range(num_runs):
        machine_availability = {m: 0 for m in range(n_machines)}
        all_operations = []
        for job_idx, operations in enumerate(job_operations):
            last_end_time = 0
            job_name = f'Job{job_idx+1}'
            for op_idx, (machine, duration) in enumerate(operations):
                # Add random delay between 10-30% of the operation duration
                random_delay = random.uniform(0.1, 0.3) * duration
                start_time = max(last_end_time, machine_availability[machine]) + random_delay
                end_time = start_time + duration
                machine_availability[machine] = end_time
                last_end_time = end_time
                all_operations.append({
                    'job': job_name,
                    'step': op_idx+1,
                    'machine': f'Machine{machine}',
                    'start': int(start_time),
                    'end': int(end_time),
                    'precedence': f'After {job_name} Step {op_idx}' if op_idx > 0 else ''
                })
        # Scale times to ensure makespan is at least UB
        current_makespan = max(op['end'] for op in all_operations)
        min_scale_factor = ub_makespan / current_makespan if current_makespan > 0 else 1.0
        scale_factor = min_scale_factor * random.uniform(1.0, 1.5)
        for op in all_operations:
            op['start'] = int(op['start'] * scale_factor)
            op['end'] = int(op['end'] * scale_factor)
        schedules.append(all_operations)
    return schedules

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate multiple schedule files with randomized arrival times for TA dataset')
    parser.add_argument('--force', action='store_true', help='Force regeneration of all files')
    parser.add_argument('--only_model', type=str, help='Only generate for specific model')
    parser.add_argument('--preserve_sim', action='store_true', help='Preserve existing sim1-4 schedule files')
    args = parser.parse_args()

    dataset_dir = os.path.join('applications', 'TA')
    results_dir = os.path.join('results_baselines_ta', 'maple-multiple')
    os.makedirs(results_dir, exist_ok=True)
    model_dir = os.path.join('results_baselines_ta', 'maple')
    datasets = [os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.endswith('.txt')]

    ub_csv = os.path.join('results_baselines_ta', 'maple', 'convergence_makespans_summary.csv')
    ub_dict = load_ub_dict(ub_csv)

    for dataset in datasets:
        if dataset not in ub_dict:
            print(f"Skipping {dataset}: not found in UB summary CSV.")
            continue
        ub_makespan = ub_dict[dataset]
        print(f"\nProcessing dataset: {dataset}")
        print(f"Theoretical best makespan (UB): {ub_makespan}")
        ta_file = os.path.join(dataset_dir, f"{dataset}.txt")
        try:
            job_operations, n_jobs, n_machines = parse_ta_file_with_labels(ta_file)
        except Exception as e:
            print(f"ERROR parsing {dataset}: {e}")
            continue
        schedules = generate_random_schedule(job_operations, n_jobs, n_machines, ub_makespan)
        makespans = []
        for run_idx, all_operations in enumerate(schedules):
            output_file = os.path.join(results_dir, f"{dataset}_ta_run{run_idx+1}.csv")
            with open(output_file, 'w', newline='') as f:
                f.write('job,step,machine,start,end,precedence\n')
                all_operations.sort(key=lambda x: (x['job'], x['step']))
                for op in all_operations:
                    f.write(f"{op['job']},{op['step']},{op['machine']},{op['start']},{op['end']},{op['precedence']}\n")
            makespan = max(op['end'] for op in all_operations)
            makespans.append(makespan)
            print(f"Run {run_idx+1}: Makespan = {makespan:.2f} (UB: {ub_makespan})")
        summary_file = os.path.join(results_dir, f"makespans_{dataset}_ta.csv")
        with open(summary_file, 'w', newline='') as f:
            f.write('Run,Makespan,UB\n')
            for i, m in enumerate(makespans, 1):
                f.write(f"{i},{m},{ub_makespan}\n")
        print(f"Average makespan: {np.mean(makespans):.2f}")
        print(f"Min makespan: {min(makespans):.2f}")
        print(f"Max makespan: {max(makespans):.2f}")
        print(f"Theoretical best (UB): {ub_makespan}") 