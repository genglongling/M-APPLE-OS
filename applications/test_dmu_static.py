import os
import sys
import random
import copy
import re

# Set up project root and src path
# (Keep for MAPLE import, but not used in synthetic generation)
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# List of DMU cases (for demonstration)
dmu_cases = [
    'DMU03', 'DMU04', 'DMU08', 'DMU09', 'DMU13', 'DMU14', 'DMU18', 'DMU19',
    'DMU23', 'DMU24', 'DMU28', 'DMU29', 'DMU33', 'DMU34', 'DMU38', 'DMU39'
]

# Static planning methods (restore previous static ones)
methods = [
    'Random', 'LPT', 'SPT', 'STPT', 'MPSR', 'DRL-Liu', 'GP', 'GEP', 'SeEvo(GLM3)', 'SeEvo(GPT3.5)'
]

def build_dmu_case_mapping(dmu_dir):
    mapping = {}
    for fname in os.listdir(dmu_dir):
        match = re.match(r'(rcmax|cscmax)_(\d+)_(\d+)_(\d+)\.txt', fname)
        if match:
            typ, jobs, machines, case_num = match.groups()
            dmu_case = f"DMU{int(case_num):02d}_{typ}"
            mapping[dmu_case] = fname
    return mapping

dmu_dir = os.path.join(dir_path, 'DMU')
dmu_case_to_filename = build_dmu_case_mapping(dmu_dir)

def load_dmu_instance(case, typ='rcmax'):
    # case: e.g. 'DMU03', typ: 'rcmax' or 'cscmax'
    key = f"{case}_{typ}"
    filename = dmu_case_to_filename.get(key)
    if not filename:
        raise ValueError(f"No file found for {case} with type {typ}")
    filepath = os.path.join(dir_path, 'DMU', filename)
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n_jobs, n_machines = map(int, lines[0].split())
    jobs = []
    for line in lines[1:1+n_jobs]:
        tokens = list(map(int, line.split()))
        ops = [(tokens[i], tokens[i+1]) for i in range(0, len(tokens), 2)]
        jobs.append(ops)
    if 'cscmax_50_20_1.txt' in filename or (case == 'DMU01' and typ == 'cscmax'):
        print(f"DEBUG: Jobs: {len(jobs)}, Machines: {n_machines}")
        for i, job in enumerate(jobs[:3]):  # print first 3 jobs
            print(f"Job {i}: {job}")
        # Print initial solution makespan
        init_sched = initial_solution(jobs, n_machines)
        init_makespan = evaluate_schedule(init_sched, jobs, n_machines)
        print(f"DEBUG: Initial solution makespan: {init_makespan}")
    return jobs, n_machines, typ

# List all DMU cases and types
all_cases = sorted(set(k.split('_')[0] for k in dmu_case_to_filename.keys()))
all_types = sorted(set(k.split('_')[1] for k in dmu_case_to_filename.keys()))

# Print all datasets to be run
print("\n=== DMU Datasets to be Evaluated ===")
for key, fname in sorted(dmu_case_to_filename.items()):
    case, typ = key.split('_')
    print(f"Case: {case}, Type: {typ}, File: {fname}")
print(f"Total datasets: {len(dmu_case_to_filename)}\n")

# --- Schedule Evaluation ---
def evaluate_schedule(schedule, jobs, n_machines):
    n_jobs = len(jobs)
    op_start = [[0]*n_machines for _ in range(n_jobs)]
    op_end = [[0]*n_machines for _ in range(n_jobs)]
    machine_time = [0]*n_machines
    job_time = [0]*n_jobs
    for m, machine_ops in enumerate(schedule):
        for (j, op_idx) in machine_ops:
            prev_end = job_time[j]
            start = max(machine_time[m], prev_end)
            duration = jobs[j][op_idx][1]
            op_start[j][op_idx] = start
            op_end[j][op_idx] = start + duration
            machine_time[m] = op_end[j][op_idx]
            job_time[j] = op_end[j][op_idx]
    return max([max(row) for row in op_end])

# --- Heuristics ---
def initial_solution(jobs, n_machines):
    n_jobs = len(jobs)
    machine_ops = [[] for _ in range(n_machines)]
    for j, job in enumerate(jobs):
        for op_idx, (m, _) in enumerate(job):
            machine_ops[m].append((j, op_idx))
    for m in range(n_machines):
        random.shuffle(machine_ops[m])
    return machine_ops

def lpt_heuristic(jobs, n_machines):
    # Longest total processing time first
    job_order = sorted(range(len(jobs)), key=lambda j: -sum(d for _, d in jobs[j]))
    return schedule_by_job_order(jobs, n_machines, job_order)

def spt_heuristic(jobs, n_machines):
    # Shortest total processing time first
    job_order = sorted(range(len(jobs)), key=lambda j: sum(d for _, d in jobs[j]))
    return schedule_by_job_order(jobs, n_machines, job_order)

def stpt_heuristic(jobs, n_machines):
    # Shortest processing time of first operation
    job_order = sorted(range(len(jobs)), key=lambda j: jobs[j][0][1])
    return schedule_by_job_order(jobs, n_machines, job_order)

def mpsr_heuristic(jobs, n_machines):
    # Most process steps remaining (just job length, descending)
    job_order = sorted(range(len(jobs)), key=lambda j: -len(jobs[j]))
    return schedule_by_job_order(jobs, n_machines, job_order)

def schedule_by_job_order(jobs, n_machines, job_order):
    # Greedy scheduler: jobs in job_order, ops in order
    machine_time = [0] * n_machines
    job_time = [0] * len(jobs)
    for j in job_order:
        for op_idx, (m, d) in enumerate(jobs[j]):
            start = max(machine_time[m], job_time[j])
            machine_time[m] = start + d
            job_time[j] = start + d
    return max(job_time)

# --- Placeholders for DRL-Liu, GP, GEP ---
def placeholder_heuristic(jobs, n_machines):
    # Just use random for now
    return evaluate_schedule(initial_solution(jobs, n_machines), jobs, n_machines)

def drl_liu_heuristic(jobs, n_machines):
    # Schedule jobs by always picking the next available operation with the earliest possible start time
    n_jobs = len(jobs)
    n_ops = len(jobs[0])
    machine_time = [0] * n_machines
    job_time = [0] * n_jobs
    op_indices = [0] * n_jobs
    total_ops = n_jobs * n_ops
    scheduled = 0
    while scheduled < total_ops:
        earliest = float('inf')
        chosen = -1
        for j in range(n_jobs):
            if op_indices[j] < n_ops:
                m, d = jobs[j][op_indices[j]]
                ready = max(machine_time[m], job_time[j])
                if ready < earliest:
                    earliest = ready
                    chosen = j
        m, d = jobs[chosen][op_indices[chosen]]
        start = max(machine_time[m], job_time[chosen])
        machine_time[m] = start + d
        job_time[chosen] = start + d
        op_indices[chosen] += 1
        scheduled += 1
    return max(job_time)

def gp_heuristic(jobs, n_machines):
    n_jobs = len(jobs)
    n_ops = len(jobs[0])
    machine_time = [0] * n_machines
    job_time = [0] * n_jobs
    op_indices = [0] * n_jobs
    total_ops = n_jobs * n_ops
    scheduled = 0
    while scheduled < total_ops:
        best_score = float('inf')
        chosen = -1
        for j in range(n_jobs):
            if op_indices[j] < n_ops:
                m, d = jobs[j][op_indices[j]]
                remaining = n_ops - op_indices[j]
                score = d * remaining  # GP: combine features
                if score < best_score:
                    best_score = score
                    chosen = j
        m, d = jobs[chosen][op_indices[chosen]]
        start = max(machine_time[m], job_time[chosen])
        machine_time[m] = start + d
        job_time[chosen] = start + d
        op_indices[chosen] += 1
        scheduled += 1
    return max(job_time)

def gep_heuristic(jobs, n_machines):
    n_jobs = len(jobs)
    n_ops = len(jobs[0])
    machine_time = [0] * n_machines
    job_time = [0] * n_jobs
    op_indices = [0] * n_jobs
    total_ops = n_jobs * n_ops
    scheduled = 0
    while scheduled < total_ops:
        best_score = float('inf')
        chosen = -1
        for j in range(n_jobs):
            if op_indices[j] < n_ops:
                m, d = jobs[j][op_indices[j]]
                remaining = n_ops - op_indices[j]
                score = d + remaining  # GEP: different combination
                if score < best_score:
                    best_score = score
                    chosen = j
        m, d = jobs[chosen][op_indices[chosen]]
        start = max(machine_time[m], job_time[chosen])
        machine_time[m] = start + d
        job_time[chosen] = start + d
        op_indices[chosen] += 1
        scheduled += 1
    return max(job_time)

# --- MAPLE-Static (use random for demonstration) ---
def maple_static(jobs, n_machines):
    # Build agents and task_spec for MAPLE
    agents = []
    for j, job in enumerate(jobs):
        agent = Agent(
            name=f"Job{j+1} Agent",
            backstory=f"Agent for Job{j+1} scheduling.",
            task_description=f"Schedule steps for Job{j+1} on required machines with precedence and strict non-overlap.",
            task_expected_output=f"Step schedule for Job{j+1} respecting machine, precedence, and non-overlap constraints."
        )
        agents.append(agent)
    supervisor_agent = Agent(
        name="Supervisor Agent",
        backstory="Aggregates all job schedules and produces the overall JSSP schedule.",
        task_description="Combine all job agent schedules into a single overall JSSP schedule.",
        task_expected_output="Overall JSSP schedule as a table."
    )
    agents.append(supervisor_agent)
    task_spec = {
        'nodes': [
            {'agent': agent, 'dependencies': []} for agent in agents[:-1]
        ] + [{'agent': supervisor_agent, 'dependencies': [a.name for a in agents[:-1]]}],
        'edges': [],
        'jobs': [{'name': f'Job{j+1}', 'steps': [(f'Machine{m+1}', d) for m, d in job]} for j, job in enumerate(jobs)],
        'disruptions': [],
        'rules': [
            'Each job must perform its steps strictly in order.',
            'Each machine can only handle one operation at a time.',
            'No two operations use the same machine at the same time.'
        ]
    }
    maple = MAPLE(task_spec)
    maple.run(with_rollback=True, validate=True)
    # Extract makespan from supervisor agent's context
    context = maple.executor.context
    output = context.get(supervisor_agent.name, {})
    if isinstance(output, dict) and 'schedule' in output:
        return max([step['end'] for step in output['schedule']])
    return -1  # fallback

# Prepare to store results
results_table = {}

header = ['Cases', 'Type', 'Size'] + methods + ['MAPLE-Static', 'UB']
print('| ' + ' | '.join(header) + ' |')
print('|' + '------|' * len(header))

for case in all_cases:
    for typ in all_types:
        key = f"{case}_{typ}"
        if key not in dmu_case_to_filename:
            continue
        jobs, n_machines, typ = load_dmu_instance(case, typ)
        row = [case, typ, f'{len(jobs)} x {n_machines}']
        method_results = {}
        # Random
        random_val = evaluate_schedule(initial_solution(jobs, n_machines), jobs, n_machines)
        row.append(str(random_val)); method_results['Random'] = random_val
        # LPT
        lpt_val = lpt_heuristic(jobs, n_machines)
        row.append(str(lpt_val)); method_results['LPT'] = lpt_val
        # SPT
        spt_val = spt_heuristic(jobs, n_machines)
        row.append(str(spt_val)); method_results['SPT'] = spt_val
        # STPT
        stpt_val = stpt_heuristic(jobs, n_machines)
        row.append(str(stpt_val)); method_results['STPT'] = stpt_val
        # MPSR
        mpsr_val = mpsr_heuristic(jobs, n_machines)
        row.append(str(mpsr_val)); method_results['MPSR'] = mpsr_val
        # DRL-Liu, GP, GEP
        drl_val = drl_liu_heuristic(jobs, n_machines)
        row.append(str(drl_val)); method_results['DRL-Liu'] = drl_val
        gp_val = gp_heuristic(jobs, n_machines)
        row.append(str(gp_val)); method_results['GP'] = gp_val
        gep_val = gep_heuristic(jobs, n_machines)
        row.append(str(gep_val)); method_results['GEP'] = gep_val
        # SeEvo columns: skip
        row.append('-'); method_results['SeEvo(GLM3)'] = '-'
        row.append('-'); method_results['SeEvo(GPT3.5)'] = '-'
        # MAPLE-Static
        maple_val = maple_static(jobs, n_machines)
        row.append(str(maple_val)); method_results['MAPLE-Static'] = maple_val
        # UB (use best of all above as a pseudo-UB)
        numeric_results = [v for k, v in method_results.items() if isinstance(v, int)]
        ub_val = min(numeric_results) if numeric_results else '-'
        row.append(str(ub_val)); method_results['UB'] = ub_val
        print('| ' + ' | '.join(row) + ' |')
        # Store in results table
        results_table[key] = {'case': case, 'type': typ, 'size': f'{len(jobs)} x {n_machines}', **method_results}

# After all runs, print a summary table
print("\n=== Summary Table (All Results) ===")
summary_header = ['Case', 'Type', 'Size'] + methods + ['MAPLE-Static', 'UB']
print('\t'.join(summary_header))
for key, res in sorted(results_table.items()):
    row = [res['case'], res['type'], res['size']]
    for m in methods:
        row.append(str(res.get(m, '-')))
    row.append(str(res.get('MAPLE-Static', '-')))
    row.append(str(res.get('UB', '-')))
    print('\t'.join(row)) 