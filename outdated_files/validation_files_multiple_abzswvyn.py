import os
import csv
from collections import defaultdict
import argparse
import glob

# Get the project root directory
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file)) if 'applications' in current_file else os.path.dirname(current_file)

# Update paths based on the new location
ABZSWVYN_DIR = os.path.join(project_root, 'applications', 'abzswvyn')

# Helper: Load ABZSWVYN dataset
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

# Helper: Load schedule from CSV
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

def validate_schedule(jobs, schedule, n_jobs, n_machines):
    errors = []
    machine_errors = 0
    precedence_errors = 0
    completion_errors = 0
    machine_error_details = []
    precedence_error_details = []
    completion_error_details = []
    
    # Track machine usage
    machine_ops = defaultdict(list)
    for op in schedule:
        machine_ops[op['machine']].append(op)
    
    # Sort operations by start time for each machine
    for machine in machine_ops:
        machine_ops[machine].sort(key=lambda x: x['start'])
    
    # Check machine constraints
    for machine, ops in machine_ops.items():
        for i in range(len(ops)-1):
            if ops[i]['end'] > ops[i+1]['start']:
                machine_errors += 1
                error_msg = f"Machine {machine}: Operation {ops[i]['job']} step {ops[i]['step']} overlaps with {ops[i+1]['job']} step {ops[i+1]['step']}"
                machine_error_details.append(error_msg)
                errors.append(error_msg)
    
    # Check precedence constraints
    job_ops = defaultdict(list)
    for op in schedule:
        job_ops[op['job']].append(op)
    
    for job, ops in job_ops.items():
        ops.sort(key=lambda x: x['step'])
        for i in range(len(ops)-1):
            if ops[i]['end'] > ops[i+1]['start']:
                precedence_errors += 1
                error_msg = f"Job {job}: Step {ops[i]['step']} ends after step {ops[i+1]['step']} starts"
                precedence_error_details.append(error_msg)
                errors.append(error_msg)
    
    # Check job completion
    for job in jobs:
        job_name = job['name']
        if job_name not in job_ops:
            completion_errors += 1
            error_msg = f"Job {job_name} has no operations"
            completion_error_details.append(error_msg)
            errors.append(error_msg)
            continue
        
        ops = job_ops[job_name]
        if len(ops) != len(job['steps']):
            completion_errors += 1
            error_msg = f"Job {job_name} has {len(ops)} operations but should have {len(job['steps'])}"
            completion_error_details.append(error_msg)
            errors.append(error_msg)
            continue
        
        # Check if all steps are present
        steps = set(op['step'] for op in ops)
        expected_steps = set(range(1, len(job['steps']) + 1))
        if steps != expected_steps:
            completion_errors += 1
            error_msg = f"Job {job_name} has steps {sorted(steps)} but should have {sorted(expected_steps)}"
            completion_error_details.append(error_msg)
            errors.append(error_msg)
    
    # Calculate makespan
    makespan = max(op['end'] for op in schedule)
    
    # Get top 5 end times for debugging
    end_times = sorted([op['end'] for op in schedule], reverse=True)
    top_end_times = end_times[:5]
    
    # Validation status
    valid = (machine_errors == 0 and precedence_errors == 0 and completion_errors == 0)
    
    return {
        'makespan': makespan,
        'machine_errors': machine_errors,
        'precedence_errors': precedence_errors,
        'completion_errors': completion_errors,
        'valid': valid,
        'error_details': '; '.join(errors),
        'machine_error_details': machine_error_details,
        'precedence_error_details': precedence_error_details,
        'completion_error_details': completion_error_details,
        'top_end_times': top_end_times
    }

def find_schedule_files(base_dir, dataset_name=None, specific_file=None):
    """Find schedule files based on criteria"""
    if specific_file:
        # Check if it's a complete path or just a filename
        if os.path.dirname(specific_file):
            # It's a complete path
            if os.path.exists(specific_file):
                return [specific_file]
            else:
                return []
        else:
            # Just a filename, look in base_dir
            file_path = os.path.join(base_dir, specific_file)
            if os.path.exists(file_path):
                return [file_path]
            else:
                return []
    
    if dataset_name:
        # Capitalize only the first letter of the dataset name
        dataset_name_cap = dataset_name[0].upper() + dataset_name[1:]
        # Find all files for a specific dataset
        patterns = [
            f'{dataset_name_cap}_maple_run*.csv'  # For maple-multiple files
        ]
        all_files = []
        for pattern in patterns:
            matching_files = glob.glob(os.path.join(base_dir, pattern))
            all_files.extend([f for f in matching_files if 'meta' not in os.path.basename(f)])
        return all_files
    else:
        # Find all schedule files
        all_files = glob.glob(os.path.join(base_dir, '*_maple_run*.csv'))
        return [f for f in all_files if 'meta' not in os.path.basename(f)]

def main():
    parser = argparse.ArgumentParser(description="Validate multiple JSSP schedules for each dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to validate (e.g., maple-multiple)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error information")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    BASE_DIR = os.path.join('results_baselines_abzswvyn', MODEL_NAME)
    ABZSWVYN_DIR = os.path.join('applications', 'abzswvyn')

    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory for model {MODEL_NAME} not found at {BASE_DIR}")
        return

    results = []
    
    # Get all datasets (assuming they are represented by ABZSWVYN files)
    abzswvyn_files = [f for f in os.listdir(ABZSWVYN_DIR) if f.endswith('.txt')]
    datasets = sorted([os.path.splitext(f)[0] for f in abzswvyn_files])

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Load ABZSWVYN data for validation
        abzswvyn_file = os.path.join(ABZSWVYN_DIR, f"{dataset_name}.txt")
        jobs, n_jobs, n_machines = load_dmu_dataset(abzswvyn_file)
        
        if not jobs:
            print(f"  Skipping {dataset_name} - could not load ABZSWVYN data")
            continue

        # Find all schedule files for this dataset
        schedule_files = find_schedule_files(BASE_DIR, dataset_name)

        if not schedule_files:
            print(f"  No schedule files found for {dataset_name}")
            continue

        print(f"  Found {len(schedule_files)} schedule files")
        
        # Validate each schedule file
        for schedule_file in sorted(schedule_files):
            filename = os.path.basename(schedule_file)
            print(f"  Validating {filename}...", end='')
            
            try:
                schedule = load_schedule(schedule_file)
                val = validate_schedule(jobs, schedule, n_jobs, n_machines)
                
                status = "✅ PASS" if val['valid'] else "❌ FAIL"
                error_summary = f"M={val['machine_errors']}, P={val['precedence_errors']}, C={val['completion_errors']}"
                
                print(f" {status} - Makespan: {val['makespan']} - Errors: {error_summary}")
                
                if args.verbose and not val['valid']:
                    if val['machine_errors'] > 0:
                        print(f"    Machine constraint violations ({val['machine_errors']}):")
                        for i, err in enumerate(val['machine_error_details'][:5]):
                            print(f"      {i+1}. {err}")
                        if len(val['machine_error_details']) > 5:
                            print(f"      ... ({len(val['machine_error_details'])-5} more)")
                    
                    if val['precedence_errors'] > 0:
                        print(f"    Precedence constraint violations ({val['precedence_errors']}):")
                        for i, err in enumerate(val['precedence_error_details'][:5]):
                            print(f"      {i+1}. {err}")
                        if len(val['precedence_error_details']) > 5:
                            print(f"      ... ({len(val['precedence_error_details'])-5} more)")
                    
                    if val['completion_errors'] > 0:
                        print(f"    Job completion errors ({val['completion_errors']}):")
                        for i, err in enumerate(val['completion_error_details'][:5]):
                            print(f"      {i+1}. {err}")
                        if len(val['completion_error_details']) > 5:
                            print(f"      ... ({len(val['completion_error_details'])-5} more)")
                
                results.append({
                    'dataset': dataset_name,
                    'file': filename,
                    'makespan': val['makespan'],
                    'valid': val['valid'],
                    'machine_errors': val['machine_errors'],
                    'precedence_errors': val['precedence_errors'],
                    'completion_errors': val['completion_errors'],
                    'error_details': val['error_details']
                })
                
            except Exception as e:
                print(f" ERROR: {str(e)}")
                results.append({
                    'dataset': dataset_name,
                    'file': filename,
                    'makespan': None,
                    'valid': False,
                    'machine_errors': -1,
                    'precedence_errors': -1,
                    'completion_errors': -1,
                    'error_details': f"Error loading/validating file: {str(e)}"
                })

    # Write summary to CSV
    summary_file = os.path.join(BASE_DIR, 'validation_summary.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'file', 'makespan', 'valid', 
                                             'machine_errors', 'precedence_errors', 
                                             'completion_errors', 'error_details'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nValidation summary written to {summary_file}")

if __name__ == "__main__":
    main() 