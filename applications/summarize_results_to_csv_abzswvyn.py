import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Get the project root directory
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file)) if 'applications' in current_file else os.path.dirname(current_file)

# Define paths
static_results_path = os.path.join(project_root, 'results_baselines_abzswvyn', 'maple', 'validation_summary.csv')
dynamic_results_path = os.path.join(project_root, 'results_baselines_abzswvyn', 'maple-multiple', 'validation_summary.csv')
ub_values_path = os.path.join(project_root, 'results_baselines_abzswvyn', 'maple', 'convergence_makespans_summary.csv')
output_dir = os.path.join(project_root, 'results_baselines_abzswvyn')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read UB values
ub_values = {}
with open(ub_values_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        ub_values[row['Dataset'].lower()] = float(row['Makespan_At_Convergence'])

# Read static results
static_results = {}
with open(static_results_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dataset = row['Dataset'].lower()  # Convert to lowercase
        makespan = int(row['Makespan'])
        valid = row['Valid'] == 'PASS'
        # Only update if this is a valid result or we haven't seen a valid result yet
        if valid or dataset not in static_results:
            static_results[dataset] = {
                'makespan': makespan,
                'valid': valid
            }

# Read dynamic results
dynamic_results = []
with open(dynamic_results_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['valid'] == 'True':  # Changed from 'Valid' to 'valid'
            dynamic_results.append({
                'dataset': row['dataset'],  # Changed from 'Dataset' to 'dataset'
                'makespan': int(row['makespan'])  # Changed from 'Makespan' to 'makespan'
            })

# Prepare table data
table_data = []
for dataset in sorted(set(r['dataset'] for r in dynamic_results)):
    # Get static result
    static_result = static_results.get(dataset, 'N/A')
    static_makespan = static_result['makespan'] if static_result != 'N/A' else 'N/A'
    
    # Get dynamic results for this dataset
    dataset_results = [r for r in dynamic_results if r['dataset'] == dataset]
    if dataset_results:
        min_makespan = min(r['makespan'] for r in dataset_results)
        max_makespan = max(r['makespan'] for r in dataset_results)
        valid_count = len(dataset_results)
        valid_rate = valid_count / 20  # Assuming 20 runs per dataset
        
        # Calculate gaps
        if static_makespan != 'N/A':
            min_gap = ((min_makespan - static_makespan) / static_makespan) * 100
            max_gap = ((max_makespan - static_makespan) / static_makespan) * 100
        else:
            min_gap = 'N/A'
            max_gap = 'N/A'
    else:
        min_makespan = 'N/A'
        max_makespan = 'N/A'
        valid_count = 0
        valid_rate = 0
        min_gap = 'N/A'
        max_gap = 'N/A'
    
    static_valid = static_result['valid'] if static_result != 'N/A' else False
    size = static_results.get(dataset, 'N/A')
    ub = ub_values[dataset]
    
    table_data.append({
        'Dataset': dataset,
        'Size': size,
        'UB': ub,
        'Static_Makespan': static_makespan,
        'Valid_Static': static_valid,
        'Dynamic_Min': min_makespan,
        'Dynamic_Max': max_makespan,
        'Valid_Rate': f"{valid_rate:.1f}%",
        'Static_Gap_%': f"{min_gap:.1f}%" if min_gap != 'N/A' else 'N/A',
        'Dynamic_Gap_%': f"{min_gap:.1f}%" if min_gap != 'N/A' else 'N/A'
    })

# Write comparison table
output_file = os.path.join(output_dir, 'maple_comparison.csv')
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Dataset', 'Size', 'UB', 'Static_Makespan', 'Valid_Static', 
                                         'Dynamic_Min', 'Dynamic_Max', 'Valid_Rate', 
                                         'Static_Gap_%', 'Dynamic_Gap_%'])
    writer.writeheader()
    writer.writerows(table_data)

# Create trend plot
plt.figure(figsize=(12, 6))
datasets = [d['Dataset'] for d in table_data]
ub_values = [d['UB'] for d in table_data]
static_makespans = [d['Static_Makespan'] for d in table_data]
dynamic_mins = [d['Dynamic_Min'] for d in table_data]
static_valid = [d['Valid_Static'] for d in table_data]

# Plot UB line
plt.plot(datasets, ub_values, 'k--', label='UB', alpha=0.5)

# Plot static makespans with validity indication
for i, (makespan, valid) in enumerate(zip(static_makespans, static_valid)):
    if makespan is not None:
        color = 'green' if valid else 'red'
        plt.scatter(datasets[i], makespan, color=color, marker='o', label='Static' if i == 0 else "")

# Plot dynamic minimum makespans
plt.plot(datasets, dynamic_mins, 'b-', label='Dynamic Min', alpha=0.7)

plt.xticks(rotation=45, ha='right')
plt.ylabel('Makespan')
plt.title('Makespan Trend Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maple_makespan_trend.png'))
plt.close()

# Create gap percentage plot
plt.figure(figsize=(12, 6))
static_gaps = [float(d['Static_Gap_%'].strip('%')) if d['Static_Gap_%'] != 'N/A' else None for d in table_data]
dynamic_gaps = [float(d['Dynamic_Gap_%'].strip('%')) if d['Dynamic_Gap_%'] != 'N/A' else None for d in table_data]

# Plot gap percentages
plt.plot(datasets, static_gaps, 'g-', label='Static Gap %', alpha=0.7)
plt.plot(datasets, dynamic_gaps, 'b-', label='Dynamic Gap %', alpha=0.7)

plt.xticks(rotation=45, ha='right')
plt.ylabel('Gap Percentage (%)')
plt.title('Gap Percentage to UB')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'maple_gap_percentage.png'))
plt.close()

# Create summary table for different models
summary_data = [
    {'Model': 'maple-static-dmu', 'Avg_Gap_%': 'N/A', 'Min_Gap_%': 'N/A', 'Max_Gap_%': 'N/A'},
    {'Model': 'maple-dynamic-dmu', 'Avg_Gap_%': 'N/A', 'Min_Gap_%': 'N/A', 'Max_Gap_%': 'N/A'},
    {'Model': 'maple-dynamic-ta', 'Avg_Gap_%': 'N/A', 'Min_Gap_%': 'N/A', 'Max_Gap_%': 'N/A'},
    {'Model': 'maple-dynamic-abz', 'Avg_Gap_%': 'N/A', 'Min_Gap_%': 'N/A', 'Max_Gap_%': 'N/A'},
    {'Model': 'maple-dynamic-swv', 'Avg_Gap_%': 'N/A', 'Min_Gap_%': 'N/A', 'Max_Gap_%': 'N/A'},
    {'Model': 'maple-dynamic-yn', 'Avg_Gap_%': 'N/A', 'Min_Gap_%': 'N/A', 'Max_Gap_%': 'N/A'}
]

# Write summary table
summary_file = os.path.join(output_dir, 'model_comparison_summary.csv')
with open(summary_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Model', 'Avg_Gap_%', 'Min_Gap_%', 'Max_Gap_%'])
    writer.writeheader()
    writer.writerows(summary_data)

print(f"Comparison table written to {output_file}")
print(f"Summary table written to {summary_file}")
print(f"Plots saved in {output_dir}") 