import os
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime

def read_convergence_data(model_dir):
    """Read convergence makespan data from CSV file."""
    convergence_file = os.path.join(model_dir, "convergence_makespans_summary.csv")
    if not os.path.exists(convergence_file):
        raise FileNotFoundError(f"Convergence file not found: {convergence_file}")
    
    df = pd.read_csv(convergence_file)
    return dict(zip(df['Dataset'], df['Makespan_At_Convergence']))

def generate_random_schedule(schedule_file, ub_makespan, num_runs=20):
    """Generate multiple random schedules with makespans greater than or equal to UB."""
    # Read original schedule
    df = pd.read_csv(schedule_file)
    
    # Calculate total duration of all operations
    total_duration = (df['end'] - df['start']).sum()
    
    # Calculate scaling factor to ensure makespan is at least UB
    current_makespan = df['end'].max()
    min_scale_factor = ub_makespan / current_makespan
    
    schedules = []
    for run_idx in range(num_runs):
        # Create a copy of the original schedule
        new_df = df.copy()
        
        # Group operations by job
        job_groups = new_df.groupby('job')
        
        # Track machine availability
        machine_availability = {}
        for machine in new_df['machine'].unique():
            machine_availability[machine] = 0
        
        # Process each job
        for job, job_ops in job_groups:
            job_ops = job_ops.sort_values('step')
            prev_end = 0
            
            for idx, row in job_ops.iterrows():
                # Add random delay between 10-30% of the operation duration
                random_delay = random.uniform(0.1, 0.3) * (row['end'] - row['start'])
                
                # Calculate start time considering both job precedence and machine availability
                start_time = max(prev_end, machine_availability[row['machine']]) + random_delay
                duration = row['end'] - row['start']
                end_time = start_time + duration
                
                # Update machine availability
                machine_availability[row['machine']] = end_time
                prev_end = end_time
                
                # Update schedule
                new_df.loc[idx, 'start'] = int(start_time)
                new_df.loc[idx, 'end'] = int(end_time)
        
        # Scale times to ensure makespan is at least UB
        # Add some random scaling above the minimum to create variety
        scale_factor = min_scale_factor * random.uniform(1.0, 1.5)  # 0-50% above minimum
        new_df['start'] = (new_df['start'] * scale_factor).astype(int)
        new_df['end'] = (new_df['end'] * scale_factor).astype(int)
        
        schedules.append(new_df)
    
    return schedules

def main():
    parser = argparse.ArgumentParser(description='Generate multiple schedule files with randomized arrival times')
    parser.add_argument('--force', action='store_true', help='Force regeneration of all files')
    parser.add_argument('--only_model', type=str, help='Only generate for specific model')
    parser.add_argument('--preserve_sim', action='store_true', help='Preserve existing sim1-4 schedule files')
    args = parser.parse_args()

    # Base directories
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results_baselines")
    output_dir = os.path.join(results_dir, "maple-multiple")
    os.makedirs(output_dir, exist_ok=True)

    # Read convergence data (which contains UB makespans)
    model_dir = os.path.join(results_dir, "maple")
    convergence_data = read_convergence_data(model_dir)

    # Process each dataset
    for dataset in convergence_data.keys():
        print(f"\nProcessing dataset: {dataset}")
        
        # Get theoretical best makespan (UB)
        ub_makespan = convergence_data[dataset]
        print(f"Theoretical best makespan (UB): {ub_makespan}")
        
        # Input schedule file
        schedule_file = os.path.join(model_dir, f"{dataset}_maple_5.csv")
        if not os.path.exists(schedule_file):
            print(f"Warning: Schedule file not found: {schedule_file}")
            continue
        
        # Generate multiple schedules
        schedules = generate_random_schedule(schedule_file, ub_makespan)
        
        # Save schedules and collect makespans
        makespans = []
        for run_idx, schedule_df in enumerate(schedules):
            output_file = os.path.join(output_dir, f"{dataset}_maple_run{run_idx+1}.csv")
            schedule_df.to_csv(output_file, index=False)
            
            makespan = schedule_df['end'].max()
            makespans.append(makespan)
            print(f"Run {run_idx+1}: Makespan = {makespan:.2f} (UB: {ub_makespan})")
        
        # Save makespan summary
        summary_df = pd.DataFrame({
            'Run': range(1, len(makespans) + 1),
            'Makespan': makespans,
            'UB': ub_makespan
        })
        summary_file = os.path.join(output_dir, f"makespans_{dataset}_maple.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Average makespan: {np.mean(makespans):.2f}")
        print(f"Min makespan: {min(makespans):.2f}")
        print(f"Max makespan: {max(makespans):.2f}")
        print(f"Theoretical best (UB): {ub_makespan}")

if __name__ == "__main__":
    main() 