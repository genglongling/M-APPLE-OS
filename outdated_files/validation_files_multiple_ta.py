import os
import csv
from collections import defaultdict
import argparse
import re

def parse_ta_file_with_labels(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
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
    jobs = []
    for job_idx, (t_line, m_line) in enumerate(zip(times_lines, machines_lines)):
        durations = [int(x) for x in t_line.split()]
        machines = [int(x)-1 for x in m_line.split()]
        steps = [(f"Machine{machine}", duration) for machine, duration in zip(machines, durations)]
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

def validate_schedule(jobs, schedule, n_jobs, n_machines):
    errors = []
    machine_errors = 0
    precedence_errors = 0
    completion_errors = 0
    machine_error_details = []
    precedence_error_details = []
    completion_error_details = []
    machine_ops = defaultdict(list)
    for op in schedule:
        machine_ops[op['machine']].append(op)
    for machine in machine_ops:
        machine_ops[machine].sort(key=lambda x: x['start'])
    for machine, ops in machine_ops.items():
        for i in range(len(ops)-1):
            if ops[i]['end'] > ops[i+1]['start']:
                machine_errors += 1
                error_msg = f"Machine {machine}: Operation {ops[i]['job']} step {ops[i]['step']} overlaps with {ops[i+1]['job']} step {ops[i+1]['step']}"
                machine_error_details.append(error_msg)
                errors.append(error_msg)
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
        steps = set(op['step'] for op in ops)
        expected_steps = set(range(1, len(job['steps']) + 1))
        if steps != expected_steps:
            completion_errors += 1
            error_msg = f"Job {job_name} has steps {sorted(steps)} but should have {sorted(expected_steps)}"
            completion_error_details.append(error_msg)
            errors.append(error_msg)
    makespan = max(op['end'] for op in schedule) if schedule else None
    end_times = sorted([op['end'] for op in schedule], reverse=True)
    top_end_times = end_times[:5]
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

def main():
    parser = argparse.ArgumentParser(description="Validate multiple JSSP schedules for each TA dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to validate (e.g., maple-multiple)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed error information")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    BASE_DIR = os.path.join('results_baselines_ta', MODEL_NAME)
    TA_DIR = os.path.join('applications', 'TA')

    if not os.path.isdir(BASE_DIR):
        print(f"Error: Base directory for model {MODEL_NAME} not found at {BASE_DIR}")
        return

    results = []
    ta_files = [f for f in os.listdir(TA_DIR) if f.endswith('.txt')]
    datasets = sorted([os.path.splitext(f)[0] for f in ta_files])

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        ta_file = os.path.join(TA_DIR, f'{dataset_name}.txt')
        try:
            jobs, n_jobs, n_machines = parse_ta_file_with_labels(ta_file)
        except Exception as e:
            print(f"  Skipping {dataset_name} - could not load TA data: {e}")
            continue
        schedule_files = []
        for file in os.listdir(BASE_DIR):
            if file.startswith(f"{dataset_name}_") and file.endswith(".csv"):
                schedule_files.append(os.path.join(BASE_DIR, file))
        if not schedule_files:
            print(f"  No schedule files found for {dataset_name}")
            continue
        print(f"  Found {len(schedule_files)} schedule files")
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