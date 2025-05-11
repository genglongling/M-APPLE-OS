import os
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reference data from the table image (only TA cases in the red box)
REFERENCE_DATA = [
    # [Case, Size, LSO, SPT/TWKR, DRL-Chen, DRL-Zhang, DRL-Liu, GP, GEP, SeEvo(GLM3), SeEvo(GPT3.5), UB]
    ["TA01", "15 × 15", 1957, 1664, 1711, 1433, 1492, 1547, 1547, 1427, 1427, 1231],
    ["TA02", "15 × 15", 1759, 1538, 1639, 1544, 1425, 1565, 1486, 1465, 1437, 1244],
    ["TA51", "50 × 15", 3844, 3768, 3762, 3599, 3608, 3603, 3668, 3364, 3412, 2760],
    ["TA52", "50 × 15", 3715, 3588, 3511, 3341, 3524, 3346, 3324, 3286, 3245, 2756],
    ["TA61", "50 × 20", 4188, 3752, 3633, 3654, 3548, 3685, 3642, 3529, 3537, 2868],
    ["TA71", "100 × 20", 6754, 6705, 6321, 6452, 6289, 6305, 6278, 6071, 6099, 5464],
    ["TA72", "100 × 20", 6674, 6351, 6232, 5695, 6002, 5776, 5625, 5604, 5575, 5181],
]

# Map TA case to result file prefix
CASE_TO_FILE = {
    "TA01": "TA01_ta_5",
    "TA02": "TA02_ta_5",
    "TA51": "TA51_ta_5",
    "TA52": "TA52_ta_5",
    "TA61": "TA61_ta_5",
    "TA71": "TA71_ta_5",
    "TA72": "TA72_ta_5",
}

# Get the project root directory
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file)) if 'applications' in current_file else os.path.dirname(current_file)

# Update paths based on the new location
RESULTS_DIR = os.path.join(project_root, 'results_baselines_ta')

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
# llm_gpt4o_makespans = {}
# gpt4o_csv = os.path.join(RESULTS_DIR, 'gpt-4o-sim1', 'convergence_makespans_summary.csv')
# if os.path.exists(gpt4o_csv):
#     with open(gpt4o_csv, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             llm_gpt4o_makespans[row['Dataset']] = int(row['Makespan_At_Convergence'])

# Load MAPLE-static (Claude-3.7) makespans from convergence_makespans_summary.csv
maple_static_claude_makespans = {}
claude_csv = os.path.join(RESULTS_DIR, 'maple', 'validation_summary.csv')
if os.path.exists(claude_csv):
    with open(claude_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            maple_static_claude_makespans[row['dataset']] = int(row['makespan'])

# Load MAPLE-static results from validation summary
maple_static_makespans = {}
maple_static_csv = os.path.join(RESULTS_DIR, 'maple', 'validation_summary.csv')
if os.path.exists(maple_static_csv):
    with open(maple_static_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            maple_static_makespans[row['dataset']] = int(row['makespan'])

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
            llm_makespans[row['Dataset']] = int(row['makespan'])

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
maple_static_df = pd.read_csv(os.path.join(RESULTS_DIR, 'maple', 'validation_summary.csv'))

# Handle both uppercase and lowercase column names for static results
dataset_col_static = 'Dataset' if 'Dataset' in maple_static_df.columns else 'dataset'
makespan_col_static = 'Makespan' if 'Makespan' in maple_static_df.columns else 'makespan'

# Store makespans for static results
maple_static_makespans = {}
for _, row in maple_static_df.iterrows():
    dataset = row[dataset_col_static]
    makespan = float(row[makespan_col_static])
    maple_static_makespans[dataset] = makespan

# --- Load MAPLE-static (TA) ---
maple_static_ta = {}
static_csv = os.path.join(project_root, 'results_baselines_ta', 'maple', 'validation_summary.csv')
if os.path.exists(static_csv):
    with open(static_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row['dataset']
            valid = row['valid'].strip().lower() == 'true'
            if valid:
                maple_static_ta[dataset] = row['makespan']
            else:
                maple_static_ta[dataset] = 'F'
else:
    print(f"Warning: Could not find MAPLE-static TA summary at {static_csv}")

# --- Load MAPLE-dynamic (TA) ---
maple_dynamic_ta = {}
dynamic_csv = os.path.join(project_root, 'results_baselines_ta', 'maple-multiple', 'validation_summary.csv')
if os.path.exists(dynamic_csv):
    with open(dynamic_csv, 'r') as f:
        reader = csv.DictReader(f)
        ta_valid = {}
        for row in reader:
            dataset = row['dataset']
            valid = row['valid'].strip().lower() == 'true'
            if valid:
                makespan = float(row['makespan'])
                if dataset not in ta_valid:
                    ta_valid[dataset] = []
                ta_valid[dataset].append(makespan)
        for dataset in CASE_TO_FILE.keys():
            if dataset in ta_valid and ta_valid[dataset]:
                maple_dynamic_ta[dataset] = str(int(min(ta_valid[dataset])))
            else:
                maple_dynamic_ta[dataset] = 'F'
else:
    print(f"Warning: Could not find MAPLE-dynamic TA summary at {dynamic_csv}")

# Prepare output rows (only for selected TA cases)
output_rows = []
for case_data in REFERENCE_DATA:
    case, size, lso, spt_twkr, drl_chen, drl_zhang, drl_liu, gp, gep, seevo_glm3, seevo_gpt35, ub = case_data
    file_prefix = CASE_TO_FILE[case]
    row = {
        'Dataset': case,
        'Size': size,
        'LSO': lso,
        'SPT/TWKR': spt_twkr,
        'DRL-Chen': drl_chen,
        'DRL-Zhang': drl_zhang,
        'DRL-Liu': drl_liu,
        'GP': gp,
        'GEP': gep,
        'SeEvo(GLM3)': seevo_glm3,
        'SeEvo(GPT3.5)': seevo_gpt35,
        'UB': ub,
        'MAPLE-dynamic (Claude-3.7 heuristics)': maple_dynamic_ta.get(case, 'F'),
        'MAPLE-static (Claude-3.7 heuristics)': maple_static_ta.get(case, 'F')
    }
    output_rows.append(row)

# Write to CSV
fieldnames = ['Dataset', 'Size', 'LSO', 'SPT/TWKR', 'DRL-Chen', 'DRL-Zhang', 'DRL-Liu', 'GP', 'GEP', 'SeEvo(GLM3)', 'SeEvo(GPT3.5)', 'UB', 'MAPLE-dynamic (Claude-3.7 heuristics)', 'MAPLE-static (Claude-3.7 heuristics)']

with open(os.path.join(RESULTS_DIR, 'summary_table.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

# Compute means
mean_row = {'Dataset': 'Mean', 'Size': ''}
for field in fieldnames[2:]:  # Skip Dataset and Size
    values = []
    for row in output_rows:
        val = row[field]
        if val in ('N/A', 'F'):
            continue
        if isinstance(val, str) and '±' in val:
            # For maple-dynamic values, use the mean part
            mean_val = float(val.split('±')[0])
            values.append(mean_val)
        else:
            try:
                values.append(float(val))
            except Exception:
                continue
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

# Define methods before using them
methods = ['LSO', 'SPT/TWKR', 'DRL-Chen', 'DRL-Zhang', 'DRL-Liu', 'GP', 'GEP',
           'SeEvo(GLM3)', 'SeEvo(GPT3.5)', 'MAPLE-dynamic (Claude-3.7 heuristics)',
           'MAPLE-static (Claude-3.7 heuristics)']

# Convert all relevant columns to numeric, coercing errors to NaN
import numpy as np
for method in methods + ['UB']:
    df_no_mean[method] = pd.to_numeric(df_no_mean[method], errors='coerce')

# Calculate gaps for all methods
for method in methods:
    df_no_mean[f'{method}_gap'] = ((df_no_mean[method] - df_no_mean['UB']) / df_no_mean['UB'] * 100)

# Create pivot table for plotting
pivot_df = df_no_mean.pivot_table(
    index='Size',
    values=[f'{method}_gap' for method in methods],
    aggfunc='mean'
)

# Plot line plot for all methods (exclude MAPLE-static)
plt.figure(figsize=(12, 6))
plot_methods = [m for m in methods if m != 'MAPLE-static (Claude-3.7 heuristics)']
color_map = {
    'LSO': 'red',
    'SPT/TWKR': 'orange',
    'DRL-Chen': 'green',
    'DRL-Zhang': 'purple',
    'DRL-Liu': 'brown',
    'GP': 'magenta',
    'GEP': 'cyan',
    'SeEvo(GLM3)': 'black',
    'SeEvo(GPT3.5)': 'black',
    'MAPLE-dynamic (Claude-3.7 heuristics)': 'blue',
}
for method in plot_methods:
    color = color_map.get(method, 'gray')
    linestyle = '-'
    plt.plot(pivot_df.index, pivot_df[f'{method}_gap'], marker='o', label=method, color=color, linestyle=linestyle)
    # Annotate mean for all methods
    mean_val = pivot_df[f'{method}_gap'].mean()
    plt.axhline(mean_val, color=color, linestyle='--', linewidth=1)
    plt.text(pivot_df.index[-1], mean_val, f"Mean: {mean_val:.2f}", color=color, va='bottom', ha='right')

plt.xlabel('Problem Size')
plt.ylabel('Gap (%)')
plt.title('Gap to Upper Bound by Method')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gap_all_methods_lineplot.png'))
plt.close()

# Plot bar plot for mean gaps (exclude MAPLE-static)
mean_gap_methods = [m for m in methods if m != 'MAPLE-static (Claude-3.7 heuristics)']
mean_gaps = df_no_mean[[f'{method}_gap' for method in mean_gap_methods]].mean()
plt.figure(figsize=(12, 6))
bar_colors = []
for method in mean_gap_methods:
    if method == 'MAPLE-dynamic (Claude-3.7 heuristics)':
        bar_colors.append('blue')
    elif 'SeEvo' in method:
        bar_colors.append('black')
    else:
        bar_colors.append('red')
mean_gaps.plot(kind='bar', color=bar_colors)
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
    safe_method = method.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    plt.savefig(os.path.join(RESULTS_DIR, f'gap_by_size_{safe_method}.png'))
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