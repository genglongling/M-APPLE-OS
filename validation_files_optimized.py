import os
import csv
from collections import defaultdict
import argparse
import glob

# Get the project root directory
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file)) if 'applications' in current_file else os.path.dirname(current_file)

# Update paths based on the new location
DMU_DIR = os.path.join(project_root, 'applications', 'DMU')

# Helper: Load DMU dataset
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
        # Find all files for a specific dataset
        patterns = [
            f'{dataset_name}_*_optimized.csv',
            f'{dataset_name}*_optimized.csv'
        ]
        all_files = []
        for pattern in patterns:
            matching_files = glob.glob(os.path.join(base_dir, pattern))
            all_files.extend([f for f in matching_files if 'meta' not in os.path.basename(f)])
        return all_files
    else:
        # Find all schedule files
        all_files = glob.glob(os.path.join(base_dir, '*_optimized.csv'))
        return [f for f in all_files if 'meta' not in os.path.basename(f)]

def main():
    parser = argparse.ArgumentParser(description="Validate optimized JSSP schedules.")
    parser.add_argument("--model_name", type=str, help="Name of the model to validate (e.g., claude-3.7-sonnet-sim1-optimized)")
    parser.add_argument("--dataset_name", type=str, help="Name of specific dataset to check (e.g., rcmax_20_15_5)")
    parser.add_argument("--specific_file", type=str, help="Path to a specific schedule file to validate")
    parser.add_argument("--all_models", action="store_true", help="Check all model directories")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error information")
    args = parser.parse_args()

    # Determine which models to check
    if args.all_models:
        model_dirs = [d for d in os.listdir(os.path.join(project_root, 'results_baselines_optimized')) 
                     if os.path.isdir(os.path.join(project_root, 'results_baselines_optimized', d))]
    elif args.model_name:
        model_dirs = [args.model_name]
    elif args.specific_file:
        # Extract model from filename if possible
        filename = os.path.basename(args.specific_file)
        parts = filename.split('_')
        if len(parts) >= 3 and parts[-1] == 'optimized.csv':
            possible_model = '_'.join(parts[2:-1])
            if os.path.isdir(os.path.join(project_root, 'results_baselines_optimized', possible_model)):
                model_dirs = [possible_model]
            else:
                model_dirs = ['']  # Will use specific file directly
        else:
            model_dirs = ['']  # Will use specific file directly
    else:
        print("Error: Must specify either --model_name, --specific_file, or --all_models")
        return

    all_results = []
    
    for model_name in model_dirs:
        if model_name:
            model_base_dir = os.path.join(project_root, 'results_baselines_optimized', model_name)
            if not os.path.isdir(model_base_dir):
                print(f"Error: Model directory not found: {model_base_dir}")
                continue
        else:
            # For specific file with no model dir
            model_base_dir = os.path.dirname(args.specific_file)
        
        # Find files to check
        if args.specific_file:
            files_to_check = [args.specific_file]
        else:
            files_to_check = find_schedule_files(model_base_dir, args.dataset_name)
        
        if not files_to_check:
            print(f"No schedule files found for model {model_name}" + 
                  (f" and dataset {args.dataset_name}" if args.dataset_name else ""))
            continue
        
        print(f"\nChecking {len(files_to_check)} schedule files for model {model_name}:")
        
        model_results = []
        
        for schedule_file_path in files_to_check:
            filename = os.path.basename(schedule_file_path)
            
            # Extract dataset name from filename
            # File format example: rcmax_50_20_9_claude-3.7-sonnet-sim1_5_optimized.csv
            # We need to extract rcmax_50_20_9
            dataset_parts = filename.split('_')
            if len(dataset_parts) >= 4 and (dataset_parts[0] == 'rcmax' or dataset_parts[0] == 'cscmax'):
                dataset_name = f"{dataset_parts[0]}_{dataset_parts[1]}_{dataset_parts[2]}_{dataset_parts[3]}"
            else:
                # Fallback for other formats
                dataset_name = dataset_parts[0]
                
            print(f"  Extracted dataset name: {dataset_name} from {filename}")
            
            # Load DMU data
            dmu_file = os.path.join(DMU_DIR, f'{dataset_name}.txt')
            if not os.path.exists(dmu_file):
                print(f"  ⚠️ Warning: DMU file not found for {dataset_name}, skipping {filename}")
                continue
            
            jobs, n_jobs, n_machines = load_dmu_dataset(dmu_file)
            
            # Load and validate schedule
            try:
                schedule = load_schedule(schedule_file_path)
                val = validate_schedule(jobs, schedule, n_jobs, n_machines)
                
                status = "✅ PASS" if val['valid'] else "❌ FAIL"
                error_summary = f"M={val['machine_errors']}, P={val['precedence_errors']}, C={val['completion_errors']}"
                
                print(f"  {status} {filename} - Makespan: {val['makespan']} - Errors: {error_summary}")
                
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
                
                # Print top 5 end times to help understand makespan
                if args.verbose and val['top_end_times']:
                    print(f"    Top 5 end times: {val['top_end_times'][:5]}")
                
                model_results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Filename': filename,
                    'Makespan': val['makespan'],
                    'MachineErrors': val['machine_errors'],
                    'PrecedenceErrors': val['precedence_errors'],
                    'CompletionErrors': val['completion_errors'],
                    'Valid': 'PASS' if val['valid'] else 'FAIL',
                    'ErrorDetails': val['error_details']
                })
                
            except Exception as e:
                print(f"  ❌ Error validating {filename}: {str(e)}")
                model_results.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Filename': filename,
                    'Makespan': None,
                    'MachineErrors': 'ERROR',
                    'PrecedenceErrors': 'ERROR',
                    'CompletionErrors': 'ERROR',
                    'Valid': 'ERROR',
                    'ErrorDetails': str(e)
                })
        
        all_results.extend(model_results)
        
        # Write summary CSV for this model
        if model_name and not args.specific_file:
            out_csv = os.path.join(model_base_dir, 'convergence_validation_summary.csv')
            # Define fieldnames, ensuring ErrorDetails is last
            fieldnames = ['Dataset', 'Makespan', 'MachineErrors', 'PrecedenceErrors', 'CompletionErrors', 'Valid', 'ErrorDetails']
            
            # Convert to CSV format
            csv_results = []
            for result in model_results:
                csv_results.append({
                    'Dataset': result['Dataset'],
                    'Makespan': result['Makespan'],
                    'MachineErrors': result['MachineErrors'],
                    'PrecedenceErrors': result['PrecedenceErrors'],
                    'CompletionErrors': result['CompletionErrors'],
                    'Valid': result['Valid'],
                    'ErrorDetails': result['ErrorDetails']
                })
            
            with open(out_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in csv_results:
                    writer.writerow(row)
            print(f"\nValidation summary written to {out_csv}")
    
    # Write overall summary if checking multiple models
    if args.all_models:
        all_out_csv = os.path.join(project_root, 'results_baselines_optimized', 'all_models_validation_summary.csv')
        fieldnames = ['Model', 'Dataset', 'Filename', 'Makespan', 'MachineErrors', 'PrecedenceErrors', 'CompletionErrors', 'Valid', 'ErrorDetails']
        with open(all_out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
        print(f"\nOverall validation summary written to {all_out_csv}")

if __name__ == '__main__':
    main() 