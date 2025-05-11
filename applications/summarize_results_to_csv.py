import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reference data from the table image
REFERENCE_DATA = [
    # [Case, Size, Random, LPT, SPT, STPT, MPSR, DRL-Liu, GP, GEP, SeEvo(GLM3), SeEvo(GPT3.5), UB]
    ["DMU03", "20 × 15", 3827, 4592, 3630, 4232, 3435, 3303, 3540, 3651, 3462, 3238, 2731],
    ["DMU04", "20 × 15", 3889, 4047, 3541, 4642, 3355, 3321, 3406, 3499, 3235, 3212, 2669],
    ["DMU08", "20 × 20", 4228, 4551, 4714, 4459, 3999, 4098, 3802, 4023, 3728, 3728, 3188],
    ["DMU09", "20 × 20", 4094, 4511, 4283, 4690, 3869, 3753, 4196, 4136, 3857, 3828, 3092],
    ["DMU13", "30 × 15", 5451, 5580, 4813, 5207, 4759, 4708, 4765, 4812, 4658, 4709, 3681],
    ["DMU14", "30 × 15", 5306, 5591, 4583, 4811, 4238, 4124, 4289, 4213, 3980, 3980, 3394],
    ["DMU18", "30 × 20", 5326, 5810, 6231, 5480, 5003, 4800, 4696, 4917, 4724, 4724, 3844],
    ["DMU19", "30 × 20", 5174, 5787, 5126, 5203, 4930, 4837, 4666, 5245, 4715, 4816, 3768],
    ["DMU23", "40 × 15", 5948, 7045, 6250, 6521, 5383, 5240, 5391, 5595, 5151, 5258, 4668],
    ["DMU24", "40 × 15", 6078, 6484, 5503, 6595, 5358, 5319, 5560, 5458, 5226, 5316, 4648],
    ["DMU28", "40 × 20", 6737, 7322, 6558, 7697, 5927, 5948, 6017, 6142, 5838, 5944, 4692],
    ["DMU29", "40 × 20", 6602, 7386, 6565, 7690, 6107, 5824, 6236, 6224, 5941, 5825, 4691],
    ["DMU33", "50 × 15", 6890, 8779, 7361, 7631, 6282, 6458, 6109, 6081, 6029, 6029, 5728],
    ["DMU34", "50 × 15", 7523, 7991, 7026, 7740, 6359, 6284, 6327, 6279, 6148, 6146, 5385],
    ["DMU38", "50 × 20", 7685, 9051, 7954, 8555, 7604, 7275, 7267, 7501, 7168, 7170, 5713],
    ["DMU39", "50 × 20", 8097, 8514, 7592, 8908, 6953, 6776, 6941, 7124, 6693, 6590, 5747],
]

# Map DMU case to result file prefix
CASE_TO_FILE = {
    "DMU03": "rcmax_20_15_5",
    "DMU04": "rcmax_20_15_8",
    "DMU08": "rcmax_20_20_7",
    "DMU09": "rcmax_20_20_8",
    "DMU13": "rcmax_30_15_5",
    "DMU14": "rcmax_30_15_4",
    "DMU18": "rcmax_30_20_9",
    "DMU19": "rcmax_30_20_8",
    "DMU23": "rcmax_40_15_10",
    "DMU24": "rcmax_40_15_8",
    "DMU28": "rcmax_40_20_6",
    "DMU29": "rcmax_40_20_2",
    "DMU33": "rcmax_50_15_2",
    "DMU34": "rcmax_50_15_4",
    "DMU38": "rcmax_50_20_6",
    "DMU39": "rcmax_50_20_9",
}

# Get the project root directory
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file)) if 'applications' in current_file else os.path.dirname(current_file)

# Update paths based on the new location
RESULTS_DIR = os.path.join(project_root, 'results_baselines')

# Load maple-dynamic results from validation summary
maple_dynamic_results = {}
maple_dynamic_dir = os.path.join(RESULTS_DIR, 'maple-multiple')
if os.path.exists(maple_dynamic_dir):
    validation_summary = os.path.join(maple_dynamic_dir, 'validation_summary.csv')
    if os.path.exists(validation_summary):
        df = pd.read_csv(validation_summary)
        for dataset in df['dataset'].unique():
            dataset_makespans = df[df['dataset'] == dataset]['makespan'].values
            if len(dataset_makespans) > 0:
                mean = np.mean(dataset_makespans)
                std = np.std(dataset_makespans)
                maple_dynamic_results[dataset] = f"{mean:.2f}±{std:.2f}"

# Load LLM(GPT-4o) makespans from convergence_makespans_summary.csv
llm_gpt4o_makespans = {}
gpt4o_csv = os.path.join(RESULTS_DIR, 'gpt-4o-sim1', 'convergence_makespans_summary.csv')
if os.path.exists(gpt4o_csv):
    with open(gpt4o_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm_gpt4o_makespans[row['Dataset']] = int(row['Makespan_At_Convergence'])

# Load MAPLE-static (Claude-3.7) makespans from convergence_makespans_summary.csv
maple_static_claude_makespans = {}
claude_csv = os.path.join(RESULTS_DIR, 'claude-3.7-sonnet-sim1', 'convergence_makespans_summary.csv')
if os.path.exists(claude_csv):
    with open(claude_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            maple_static_claude_makespans[row['Dataset']] = int(row['Makespan_At_Convergence'])

# Load MAPLE-static results from validation summary
maple_static_makespans = {}
maple_static_csv = os.path.join(RESULTS_DIR, 'maple', 'convergence_makespans_summary.csv')
if os.path.exists(maple_static_csv):
    with open(maple_static_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            maple_static_makespans[row['Dataset']] = int(row['Makespan_At_Convergence'])

# Load MAPLE-dynamic results from validation summary
maple_dynamic_data = {}
maple_dynamic_dir = os.path.join(RESULTS_DIR, 'maple-multiple')
maple_dynamic_summary = os.path.join(maple_dynamic_dir, 'validation_summary.csv')

if os.path.exists(maple_dynamic_summary):
    with open(maple_dynamic_summary, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']  # Changed from 'dataset' to 'Dataset' to match the file format
            makespan = float(row['makespan'])  # Changed from 'makespan' to 'Makespan'
            if dataset not in maple_dynamic_data:
                maple_dynamic_data[dataset] = []
            maple_dynamic_data[dataset].append(makespan)
else:
    print(f"Warning: Could not find validation summary file at {maple_dynamic_summary}")

# Calculate mean and std for each dataset in maple-dynamic
maple_dynamic_stats = {}
for dataset, makespans in maple_dynamic_data.items():
    mean = np.mean(makespans)
    std = np.std(makespans)
    maple_dynamic_stats[dataset] = f"{mean:.2f}±{std:.2f}"

# Load makespan values from other CSV files
llm_makespans = {}
gpt4o_csv = os.path.join(RESULTS_DIR, 'gpt-4o-sim1', 'convergence_makespans_summary.csv')
if os.path.exists(gpt4o_csv):
    with open(gpt4o_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm_makespans[row['Dataset']] = int(row['Makespan_At_Convergence'])

# Load MAPLE-dynamic results
maple_dynamic_df = pd.read_csv(os.path.join(RESULTS_DIR, 'maple-multiple', 'validation_summary.csv'))

# Handle both uppercase and lowercase column names
dataset_col = 'Dataset' if 'Dataset' in maple_dynamic_df.columns else 'dataset'
makespan_col = 'Makespan' if 'Makespan' in maple_dynamic_df.columns else 'makespan'

# Group by dataset and calculate minimum makespan
maple_dynamic_stats = {}
for dataset, group in maple_dynamic_df.groupby(dataset_col):
    makespans = group[makespan_col].astype(float)
    maple_dynamic_stats[dataset] = {
        'min': np.min(makespans)
    }

# Load MAPLE-static results
maple_static_df = pd.read_csv(os.path.join(RESULTS_DIR, 'claude-3.7-sonnet-sim1', 'convergence_validation_summary.csv'))

# Handle both uppercase and lowercase column names for static results
dataset_col_static = 'Dataset' if 'Dataset' in maple_static_df.columns else 'dataset'
makespan_col_static = 'Makespan' if 'Makespan' in maple_static_df.columns else 'makespan'

# Store makespans for static results
maple_static_makespans = {}
for _, row in maple_static_df.iterrows():
    dataset = row[dataset_col_static]
    makespan = float(row[makespan_col_static])
    maple_static_makespans[dataset] = makespan

# Prepare output rows
output_rows = []
for case_data in REFERENCE_DATA:
    case, size, random, lpt, spt, stpt, mpsr, drl, gp, gep, seevo_glm3, seevo_gpt35, ub = case_data
    file_prefix = CASE_TO_FILE[case]
    
    row = {
        'Dataset': case,
        'Size': size,
        'Random': random,
        'LPT': lpt,
        'SPT': spt,
        'STPT': stpt,
        'MPSR': mpsr,
        'DRL-Liu': drl,
        'GP': gp,
        'GEP': gep,
        'SeEvo(GLM3)': seevo_glm3,
        'SeEvo(GPT3.5)': seevo_gpt35,
        'UB': ub,
        'MAPLE-dynamic (Claude-3.7 heuristics)': f"{maple_dynamic_stats.get(file_prefix, {}).get('min', 'N/A')}",
        'MAPLE-static (Claude-3.7 heuristics)': f"{maple_static_makespans.get(file_prefix, 'N/A')}"
    }
    output_rows.append(row)

# Write to CSV
fieldnames = ['Dataset', 'Size', 'Random', 'LPT', 'SPT', 'STPT', 'MPSR', 'DRL-Liu', 'GP', 'GEP', 
              'SeEvo(GLM3)', 'SeEvo(GPT3.5)', 'UB', 'MAPLE-dynamic (Claude-3.7 heuristics)', 
              'MAPLE-static (Claude-3.7 heuristics)']

with open(os.path.join(RESULTS_DIR, 'summary_table.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

# Compute means
mean_row = {'Dataset': 'Mean', 'Size': ''}
for field in fieldnames[2:]:  # Skip Dataset and Size
    values = []
    for row in output_rows:
        if row[field] != 'N/A':
            if isinstance(row[field], str) and '±' in row[field]:
                # For maple-dynamic values, use the mean part
                mean_val = float(row[field].split('±')[0])
                values.append(mean_val)
            else:
                values.append(float(row[field]))
    if values:
        mean_row[field] = f"{sum(values) / len(values):.2f}"
    else:
        mean_row[field] = 'N/A'

# Append mean row
with open(os.path.join(RESULTS_DIR, 'summary_table.csv'), 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writerow(mean_row)

print(f"Summary table written to {os.path.join(RESULTS_DIR, 'summary_table.csv')}")

# Process data for plotting
df = pd.read_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'))
df_no_mean = df[df['Dataset'] != 'Mean'].copy()

# Calculate gaps for all methods
methods = ['Random', 'LPT', 'SPT', 'STPT', 'MPSR', 'DRL-Liu', 'GP', 'GEP', 
           'SeEvo(GLM3)', 'SeEvo(GPT3.5)', 'MAPLE-dynamic (Claude-3.7 heuristics)',
           'MAPLE-static (Claude-3.7 heuristics)']

for method in methods:
    df_no_mean[f'{method}_gap'] = ((df_no_mean[method].astype(float) - df_no_mean['UB'].astype(float)) / 
                                 df_no_mean['UB'].astype(float) * 100)

# Create pivot table for plotting
pivot_df = df_no_mean.pivot_table(
    index='Size',
    values=[f'{method}_gap' for method in methods],
    aggfunc='mean'
)

# Plot line plot for all methods
plt.figure(figsize=(12, 6))
for method in methods:
    plt.plot(pivot_df.index, pivot_df[f'{method}_gap'], marker='o', label=method)

plt.xlabel('Problem Size')
plt.ylabel('Gap (%)')
plt.title('Gap to Upper Bound by Method')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gap_all_methods_lineplot.png'))
plt.close()

# Plot bar plot for mean gaps
mean_gaps = df_no_mean[[f'{method}_gap' for method in methods]].mean()
plt.figure(figsize=(12, 6))
mean_gaps.plot(kind='bar')
plt.xlabel('Method')
plt.ylabel('Mean Gap (%)')
plt.title('Mean Gap to Upper Bound by Method')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mean_gap_all_methods.png'))
plt.close()

# Plot gap by size for each method
for method in methods:
    plt.figure(figsize=(10, 6))
    pivot_df[f'{method}_gap'].plot(kind='bar')
    plt.xlabel('Problem Size')
    plt.ylabel('Gap (%)')
    plt.title(f'Gap to Upper Bound by Size - {method}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'gap_by_size_{method.replace(" ", "_").replace("(", "").replace(")", "")}.png'))
    plt.close()

# --- Generate LaTeX Table ---
def csv_to_latex(csv_path, tex_path):
    df = pd.read_csv(csv_path)
    # Bold the best (minimum) value in each row (excluding UB and Mean row)
    def bold_min(row):
        # Only for data rows, not the mean row
        if row[0] == 'Mean':
            return row
        # Find the minimum among all methods (excluding UB and MAPLE-static)
        vals = []
        for v in row[1:]:  # Exclude UB and MAPLE-static
            if isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, str) and '±' in v:
                # For maple-dynamic values, use the mean part
                mean_val = float(v.split('±')[0])
                vals.append(mean_val)
        if not vals:
            return row
        min_val = min(vals)
        new_row = row.copy()
        for i, v in enumerate(row[1:]):
            if isinstance(v, (int, float)) and float(v) == min_val:
                new_row[i+1] = f"\\textbf{{{int(v)}}}"
            elif isinstance(v, str) and '±' in v:
                mean_val = float(v.split('±')[0])
                if mean_val == min_val:
                    new_row[i+1] = f"\\textbf{{{v}}}"
        return new_row
    df = df.apply(bold_min, axis=1)
    # Convert to LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='ll', longtable=False)
    with open(tex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table written to {tex_path}")

# Generate LaTeX table
csv_path = os.path.join(RESULTS_DIR, "summary_table.csv")
tex_path = os.path.join(RESULTS_DIR, "summary_table.tex")
csv_to_latex(csv_path, tex_path)

# --- Step 2: Plot Values Over Iterations (MAPLE-static) ---
def extract_iterations(filepath):
    """Extract makespan values over iterations from a result file."""
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            # Look for lines like: [Tabu] Iteration 1: New best makespan found: 3639
            m = re.search(r"Iteration (\d+):.*makespan.*?: (\d+)", line)
            if m:
                values.append(int(m.group(2)))
    return values

plt.figure(figsize=(14, 5))
legend_labels = []
for row in REFERENCE_DATA:
    case = row[0]
    file_prefix = CASE_TO_FILE[case]
    result_file = os.path.join(RESULTS_DIR, f"{file_prefix}_dmu.txt")
    if os.path.exists(result_file):
        y = extract_iterations(result_file)
        if len(y) > 0:
            plt.plot(range(1, len(y)+1), y, marker='o', label=case)
            legend_labels.append(case)

plt.xlabel('Iteration (DMU)')
plt.ylabel('Makespan')
plt.title('MAPLE-static (GPT-4o) Makespan over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'maple_static_iterations.png'))
plt.close()
print(f"Iteration plot saved to {os.path.join(RESULTS_DIR, 'maple_static_iterations.png')}")

# --- Figure 4: Gap Percentage by Size Group ---
# Define size groups based on dataset names
size_groups = {
    '20x15': ['rcmax_20_15_5', 'rcmax_20_15_8'],
    '20x20': ['rcmax_20_20_7', 'rcmax_20_20_8'],
    '30x15': ['rcmax_30_15_5', 'rcmax_30_15_4'],
    '30x20': ['rcmax_30_20_9', 'rcmax_30_20_8'],
    '40x15': ['rcmax_40_15_10', 'rcmax_40_15_8'],
    '40x20': ['rcmax_40_20_6', 'rcmax_40_20_2'],
    '50x15': ['rcmax_50_15_2', 'rcmax_50_15_4'],
    '50x20': ['rcmax_50_20_6', 'rcmax_50_20_9']
}

for size, datasets in size_groups.items():
    group = df_no_mean[df_no_mean['Dataset'].isin(datasets)]
    if group.empty:
        continue
        
    cases = group['Dataset'].tolist()
    ubs = group['MAPLE-static (Claude-3.7 heuristics)'].astype(float).tolist()
    plt.figure(figsize=(12, 6))
    
    for idx, method in enumerate(methods):
        vals = []
        for v in group[method]:
            if isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, str) and '±' in v:
                mean_val = float(v.split('±')[0])
                vals.append(mean_val)
            else:
                vals.append(None)
        gaps = [100 * (v-u)/u if v is not None and u > 0 else None for v, u in zip(vals, ubs)]
        
        # Set colors for different method types
        if 'MAPLE-dynamic' in method:
            color = 'red'
            linestyle = '--'
        elif 'MAPLE-static' in method:
            color = 'blue'
            linestyle = '-'
        else:
            color = color_map[idx % len(color_map)]
            linestyle = '-'
        
        plt.plot(cases, gaps, marker='o', label=method, color=color, linestyle=linestyle)
    
    plt.xlabel('Instance', fontsize=12)
    plt.ylabel('Gap Percentage (%)', fontsize=12)
    plt.title(f'Gap Percentage by Size Group ({size})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'gap_by_size_{size}.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Gap percentage plot for size {size} saved") 