import os
import csv
from collections import defaultdict
import argparse # Import argparse

DMU_DIR = os.path.join(os.path.dirname(__file__), 'DMU')
# BASE_DIR will be set via command-line argument

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
    return jobs

# Helper: Load schedule from CSV
def load_schedule(filepath):
    schedule = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            schedule.append({
                'job': row['job'],
                'step': int(row['step']),
                'machine': row['machine'],
                'start': int(row['start']),
                'end': int(row['end'])
            })
    return schedule

def validate_schedule(jobs, schedule):
    errors = []
    # Build job specs
    job_specs = {job['name']: job for job in jobs}
    # 1. Machine constraints
    machine_schedules = defaultdict(list)
    for entry in schedule:
        machine_schedules[entry['machine']].append(entry)
    machine_errors = 0
    for machine, ops in machine_schedules.items():
        ops.sort(key=lambda x: x['start'])
        for i in range(len(ops)-1):
            if ops[i]['end'] > ops[i+1]['start']:
                machine_errors += 1
                errors.append(f"Overlap on {machine}: {ops[i]['job']} Step {ops[i]['step']} and {ops[i+1]['job']} Step {ops[i+1]['step']}")
    # 2. Job precedence
    precedence_errors = 0
    job_steps = defaultdict(list)
    for entry in schedule:
        job_steps[entry['job']].append(entry)
    for job_name_key, steps in job_steps.items(): # Renamed 'job' to 'job_name_key' to avoid conflict
        steps.sort(key=lambda x: x['step'])
        for i in range(len(steps)-1):
            # Check if the job exists in job_specs before accessing it
            if job_name_key in job_specs:
                current_job_spec = job_specs[job_name_key]
                # Ensure step numbers are within bounds
                if steps[i]['step'] < len(current_job_spec['steps']) and steps[i+1]['step'] <= len(current_job_spec['steps']):
                    if steps[i]['end'] > steps[i+1]['start']:
                        precedence_errors += 1
                        errors.append(f"Precedence violation in {job_name_key}: Step {steps[i]['step']} ends after Step {steps[i+1]['step']} starts")
            else:
                # This case should ideally not happen if schedule files are correct
                errors.append(f"Job {job_name_key} from schedule not found in DMU specs.")

    # 3. Job completion
    completion_errors = 0
    for job_spec_item in jobs: # Iterate through jobs from DMU
        job_name = job_spec_item['name']
        expected_steps = len(job_spec_item['steps'])
        # Filter schedule entries for the current job_name
        actual_steps = len([s for s in schedule if s['job'] == job_name])
        if actual_steps != expected_steps:
            completion_errors += 1
            errors.append(f"Incomplete schedule for {job_name}: expected {expected_steps}, got {actual_steps}")

    # 4. Makespan
    makespan = max(entry['end'] for entry in schedule) if schedule else None
    # Validation status
    valid = (machine_errors == 0 and precedence_errors == 0 and completion_errors == 0)
    return {
        'makespan': makespan,
        'machine_errors': machine_errors,
        'precedence_errors': precedence_errors,
        'completion_errors': completion_errors,
        'valid': valid,
        'error_details': '; '.join(errors)
    }

def main():
    parser = argparse.ArgumentParser(description="Validate JSSP schedules for a given model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to validate (e.g., gemini-2.5-sim1)")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    BASE_DIR = os.path.join(os.path.dirname(__file__), f'../results_baselines/{MODEL_NAME}')

    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory for model {MODEL_NAME} not found at {BASE_DIR}")
        return

    results = []
    
    # Get all datasets (assuming they are represented by DMU files)
    dmu_files = [f for f in os.listdir(DMU_DIR) if f.endswith('.txt')]
    datasets = sorted([os.path.splitext(f)[0] for f in dmu_files])

    for dataset_name in datasets:
        dmu_file = os.path.join(DMU_DIR, f'{dataset_name}.txt')
        jobs = load_dmu_dataset(dmu_file)
        
        # Construct schedule file name based on new convention
        schedule_file_name = f'{dataset_name}_{MODEL_NAME}_5.csv'
        schedule_file_path = os.path.join(BASE_DIR, schedule_file_name)

        if not os.path.exists(schedule_file_path):
            print(f"No schedule file found for {dataset_name} at {schedule_file_path}")
            results.append({
                'Dataset': dataset_name,
                'Makespan': None,
                'MachineErrors': 'N/A',
                'PrecedenceErrors': 'N/A',
                'CompletionErrors': 'N/A',
                'Valid': 'FAIL',
                'ErrorDetails': 'Schedule file not found.'
            })
            continue
            
        schedule = load_schedule(schedule_file_path)
        val = validate_schedule(jobs, schedule)
        results.append({
            'Dataset': dataset_name,
            'Makespan': val['makespan'],
            'MachineErrors': val['machine_errors'],
            'PrecedenceErrors': val['precedence_errors'],
            'CompletionErrors': val['completion_errors'],
            'Valid': 'PASS' if val['valid'] else 'FAIL',
            'ErrorDetails': val['error_details']
        })
        print(f"{dataset_name}: {'PASS' if val['valid'] else 'FAIL'} (Makespan: {val['makespan']}) Errors: M={val['machine_errors']}, P={val['precedence_errors']}, C={val['completion_errors']}")

    # Write summary CSV
    out_csv = os.path.join(BASE_DIR, 'convergence_validation_summary.csv')
    # Define fieldnames, ensuring ErrorDetails is last
    fieldnames = ['Dataset', 'Makespan', 'MachineErrors', 'PrecedenceErrors', 'CompletionErrors', 'Valid', 'ErrorDetails']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nValidation summary written to {out_csv}")

if __name__ == '__main__':
    main() 