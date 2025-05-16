import pandas as pd
import os

# Define upper bounds for each dataset (TA hardcoded, DMU loaded from file)
TA_UPPER_BOUNDS = {
    'TA01': 1231,
    'TA02': 1244,
    'TA51': 2760,
    'TA52': 2756,
    'TA61': 2868,
    'TA71': 5464,
    'TA72': 5181
}

def load_dmu_upper_bounds(convergence_csv_path):
    df = pd.read_csv(convergence_csv_path)
    return dict(zip(df['Dataset'], df['Makespan_At_Convergence']))

def calculate_gap_percentage(makespan, ub):
    return ((makespan - ub) / ub) * 100

def analyze_validation_summary(csv_path, upper_bounds, label=None):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group by dataset
    grouped = df.groupby('dataset')
    
    print("\n=== Validation Summary Analysis ===")
    if label:
        print(f"Dataset Group: {label}")
    print("=" * 50)
    
    min_gaps = []
    min_makespans = []  # Track minimum makespans for average calculation
    
    for dataset, group in grouped:
        # Get all makespan values for this dataset
        makespans = sorted(group['makespan'].tolist())
        min_makespan = min(makespans)
        min_makespans.append(min_makespan)  # Add to list for average calculation
        ub = upper_bounds.get(dataset)
        if ub is None:
            print(f"[Warning] No UB found for {dataset}, skipping...")
            continue
        # Calculate gap percentages
        gap_percentages = [calculate_gap_percentage(m, ub) for m in makespans]
        min_gap = min(gap_percentages)
        min_gaps.append(min_gap)
        
        print(f"\nDataset: {dataset}")
        print(f"Upper Bound: {ub}")
        print(f"Minimum Makespan: {min_makespan}")
        print(f"Minimum Gap Percentage: {min_gap:.2f}%")
        print(f"All Makespan Values: {makespans}")
        print(f"All Gap Percentages: {[f'{g:.2f}%' for g in gap_percentages]}")
    
    # Calculate and print averages
    if min_gaps:
        avg_min_gap = sum(min_gaps) / len(min_gaps)
        avg_min_makespan = sum(min_makespans) / len(min_makespans)
        print("\n=== Summary Statistics ===")
        print(f"Average Minimum Gap Percentage: {avg_min_gap:.2f}%")
        print(f"Average Minimum Makespan: {avg_min_makespan:.2f}")
    
    return min_gaps, min_makespans

def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    ta_csv_path = os.path.join(project_root, "results_baselines_ta", "maple-multiple", "validation_summary.csv")
    dmu_csv_path = os.path.join(project_root, "results_baselines_dmu", "maple-multiple", "validation_summary.csv")
    dmu_convergence_path = os.path.join(project_root, "results_baselines_dmu", "maple-multiple", "convergence_makespans_summary.csv")
    
    # Load DMU upper bounds
    dmu_upper_bounds = load_dmu_upper_bounds(dmu_convergence_path)
    
    # Process TA datasets
    if os.path.exists(ta_csv_path):
        print("\nProcessing TA datasets...")
        ta_min_gaps, ta_min_makespans = analyze_validation_summary(ta_csv_path, TA_UPPER_BOUNDS, "TA")
        if ta_min_gaps:
            print("\n=== TA Overall Statistics ===")
            print(f"Average Minimum Gap Percentage: {sum(ta_min_gaps) / len(ta_min_gaps):.2f}%")
            print(f"Average Minimum Makespan: {sum(ta_min_makespans) / len(ta_min_makespans):.2f}")
    
    # Process DMU datasets
    if os.path.exists(dmu_csv_path):
        print("\nProcessing DMU datasets...")
        dmu_min_gaps, dmu_min_makespans = analyze_validation_summary(dmu_csv_path, dmu_upper_bounds, "DMU")
        if dmu_min_gaps:
            print("\n=== DMU Overall Statistics ===")
            print(f"Average Minimum Gap Percentage: {sum(dmu_min_gaps) / len(dmu_min_gaps):.2f}%")
            print(f"Average Minimum Makespan: {sum(dmu_min_makespans) / len(dmu_min_makespans):.2f}")
    
    # Print combined statistics if both datasets were processed
    if os.path.exists(ta_csv_path) and os.path.exists(dmu_csv_path):
        all_min_gaps = ta_min_gaps + dmu_min_gaps
        all_min_makespans = ta_min_makespans + dmu_min_makespans
        print("\n=== Combined Statistics (TA + DMU) ===")
        print(f"Average Minimum Gap Percentage: {sum(all_min_gaps) / len(all_min_gaps):.2f}%")
        print(f"Average Minimum Makespan: {sum(all_min_makespans) / len(all_min_makespans):.2f}")

if __name__ == "__main__":
    main() 