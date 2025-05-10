import os
import re
import csv
import matplotlib.pyplot as plt
import pandas as pd

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

# add o1-preview, o1-mini, Claude-3.5 Sonnet, Claude 3 Opus, GPT-4o, GPT-4, LLaMA-3.1 405B, LLaMA-3 70B, Gemini 1.5 Pro, Deepseek R1
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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results0')

# Extract best makespan from a result file
def extract_best_makespan(filepath):
    best = None
    with open(filepath, 'r') as f:
        for line in f:
            m = re.search(r"Best static makespan for", line)
            if m:
                # Extract the number at the end
                num = re.findall(r"\d+", line)
                if num:
                    best = int(num[-1])
    return best

# Load LLM(GPT-4o) makespans from convergence_makespans_summary.csv
llm_gpt4o_makespans = {}
gpt4o_csv = os.path.join(os.path.dirname(__file__), '../results_baselines/GPT-4o/convergence_makespans_summary.csv')
if os.path.exists(gpt4o_csv):
    with open(gpt4o_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            llm_gpt4o_makespans[row['Dataset']] = int(row['Makespan_At_Convergence'])

# Prepare output rows
rows = []
for row in REFERENCE_DATA:
    case = row[0]
    file_prefix = CASE_TO_FILE[case]
    result_file = os.path.join(RESULTS_DIR, f"{file_prefix}_dmu.txt")
    maple_static = extract_best_makespan(result_file) if os.path.exists(result_file) else None
    llm_gpt4o_val = llm_gpt4o_makespans.get(file_prefix)
    rows.append(row + [maple_static, llm_gpt4o_val])

# Update header
header = [
    "Cases", "Size", "Random", "LPT", "SPT", "STPT", "MPSR", "DRL-Liu", "GP", "GEP",
    "SeEvo(GLM3)", "SeEvo(GPT3.5)", "UB", "MAPLE-static (GPT-4o)", "LLM(GPT-4o)"
]

# Compute means
means = ["Mean", ""]
for col in range(2, len(rows[0])):
    vals = [r[col] for r in rows if isinstance(r[col], (int, float))]
    means.append(round(sum(vals)/len(vals), 2))
rows.append(means)

# Write to CSV
with open(os.path.join(RESULTS_DIR, "summary_table.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

print(f"Summary table written to {os.path.join(RESULTS_DIR, 'summary_table.csv')}")

# --- Generate LaTeX Table ---
def csv_to_latex(csv_path, tex_path):
    df = pd.read_csv(csv_path)
    # Bold the best (minimum) value in each row (excluding UB and Mean row)
    def bold_min(row):
        # Only for data rows, not the mean row
        if row[0] == 'Mean':
            return row
        # Find the minimum among all methods (excluding UB and MAPLE-static)
        vals = row[2:-2] + [row[-1]]  # Exclude UB, include MAPLE-static
        try:
            vals = [float(v) for v in vals]
        except:
            return row
        min_val = min(vals)
        new_row = row.copy()
        for i, v in enumerate(vals):
            if float(v) == min_val:
                new_row[2+i if i < len(vals)-1 else -1] = f"\\textbf{{{int(v)}}}"
        return new_row
    df = df.apply(bold_min, axis=1)
    # Convert to LaTeX
    latex = df.to_latex(index=False, escape=False, column_format='llcccccccccccc', longtable=False)
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
            m = re.search(r"Iteration (\\d+):.*makespan.*?: (\\d+)", line)
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

# --- Step 3 (line plot): Gap Percentage for All Methods Across DMUs ---
df = pd.read_csv(os.path.join(RESULTS_DIR, 'summary_table.csv'))
df_no_mean = df[df['Cases'] != 'Mean']
method_cols = [col for col in df.columns if col not in ['Cases', 'Size', 'UB', 'Mean', 'SeEvo(GLM3)', 'SeEvo(GPT3.5)']]

ubs = df_no_mean['UB'].astype(float).tolist()
cases = df_no_mean['Cases'].tolist()

plt.figure(figsize=(12, 6))
color_map = plt.cm.tab20.colors
for idx, method in enumerate(method_cols):
    vals = df_no_mean[method].astype(float).tolist()
    gaps = [100 * (v-u)/u if u > 0 else None for v, u in zip(vals, ubs)]
    color = 'blue' if method == 'MAPLE-static (GPT-4o)' else color_map[idx % len(color_map)]
    plt.plot(cases, gaps, marker='o', label=method, color=color)

plt.xlabel('Instance')
plt.ylabel('Gap Percentage (%)')
plt.title('Gap Percentage for All Methods Across DMUs')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'gap_all_methods_lineplot.png'))
plt.close()
print("Line plot of gap percentage for all methods saved as gap_all_methods_lineplot.png")

# --- Figure 4: Gap Percentage by Size Group ---
size_groups = df_no_mean.groupby('Size')
for size, group in size_groups:
    cases = group['Cases'].tolist()
    ubs = group['UB'].astype(float).tolist()
    plt.figure(figsize=(10, 5))
    for idx, method in enumerate(method_cols):
        vals = group[method].astype(float).tolist()
        gaps = [100 * (v-u)/u if u > 0 else None for v, u in zip(vals, ubs)]
        color = 'blue' if method == 'MAPLE-static (GPT-4o)' else color_map[idx % len(color_map)]
        plt.plot(cases, gaps, marker='o', label=method, color=color)
    plt.xlabel('Instance')
    plt.ylabel('Gap Percentage (%)')
    plt.title(f'Gap Percentage for Size {size}')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fname = f'gap_by_size_{size.replace(" ", "x").replace("×", "x")}.png'
    plt.savefig(os.path.join(RESULTS_DIR, fname))
    plt.close()
    print(f"Gap percentage plot for size {size} saved as {fname}")

# Mean gap bar plot (as before)
mean_gaps = []
for method in method_cols:
    vals = df_no_mean[method].astype(float).tolist()
    ubs = df_no_mean['UB'].astype(float).tolist()
    gaps = [100 * (v-u)/u if u > 0 else None for v, u in zip(vals, ubs)]
    mean_gaps.append(sum(gaps)/len(gaps))
plt.figure(figsize=(10,5))
bar_colors = ['blue' if method == 'MAPLE-static (GPT-4o)' else color_map[idx % len(color_map)] for idx, method in enumerate(method_cols)]
plt.bar(method_cols, mean_gaps, color=bar_colors)
plt.ylabel('Mean Gap Percentage (%)')
plt.title('Mean Gap Percentage for Each Method')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mean_gap_all_methods.png'))
plt.close()
print("Mean gap bar plot saved for all methods.") 