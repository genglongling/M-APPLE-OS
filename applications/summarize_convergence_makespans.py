import os
import csv

def find_meta_file(folder):
    for fname in os.listdir(folder):
        if fname.startswith('meta_') and fname.endswith('.csv'):
            return os.path.join(folder, fname)
    return None

def get_convergence_makespan(meta_file):
    with open(meta_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['iteration']) == 1:
                return int(row['makespan'])
    return None

def main():
    base_dir = os.path.join(os.path.dirname(__file__), '../results_baselines/GPT-4o')
    output_folders = [d for d in os.listdir(base_dir) if d.endswith('_outputs') and os.path.isdir(os.path.join(base_dir, d))]
    results = []
    for folder in sorted(output_folders):
        full_path = os.path.join(base_dir, folder)
        meta_file = find_meta_file(full_path)
        if not meta_file:
            print(f"No meta file found in {folder}")
            continue
        makespan = get_convergence_makespan(meta_file)
        dataset = folder.replace('_gpt-4o_outputs', '')
        results.append({'Dataset': dataset, 'Makespan_At_Convergence': makespan})
        print(f"{dataset}: {makespan}")
    # Write summary CSV
    out_csv = os.path.join(base_dir, 'convergence_makespans_summary.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Dataset', 'Makespan_At_Convergence'])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nSummary written to {out_csv}")

if __name__ == '__main__':
    main() 