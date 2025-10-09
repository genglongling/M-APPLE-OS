#!/usr/bin/env python3
"""
Extract MAPLE makespan values from all iterations and create a comprehensive table
"""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple

def extract_maple_iterations(txt_file_path: str) -> List[int]:
    """Extract all makespan values from MAPLE iterations"""
    if not os.path.exists(txt_file_path):
        return []
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all "Best static makespan" occurrences
        pattern = r'Best static makespan for \w+: (\d+)'
        matches = re.findall(pattern, content)
        
        if matches:
            return [int(match) for match in matches]
        else:
            return []
            
    except Exception as e:
        print(f"Error reading {txt_file_path}: {e}")
        return []

def get_all_maple_files(results_dir: str = "results_maple") -> List[str]:
    """Get all MAPLE result files"""
    if not os.path.exists(results_dir):
        return []
    
    maple_files = []
    for filename in os.listdir(results_dir):
        if filename.endswith('_maple.txt'):
            maple_files.append(os.path.join(results_dir, filename))
    
    return sorted(maple_files)

def extract_dataset_name(filename: str) -> str:
    """Extract dataset name from filename"""
    basename = os.path.basename(filename)
    # Remove '_maple.txt' suffix
    dataset_name = basename.replace('_maple.txt', '')
    return dataset_name

def create_iterations_table():
    """Create a comprehensive table of MAPLE iterations"""
    
    # Get all MAPLE files
    maple_files = get_all_maple_files()
    
    if not maple_files:
        print("No MAPLE files found in results_maple directory")
        return
    
    print(f"Found {len(maple_files)} MAPLE files")
    
    # Extract data from all files
    all_data = []
    
    for file_path in maple_files:
        dataset_name = extract_dataset_name(file_path)
        iterations = extract_maple_iterations(file_path)
        
        if iterations:
            # Create a row for each dataset
            row = {'Dataset': dataset_name}
            
            # Add each iteration as a column
            for i, makespan in enumerate(iterations, 1):
                row[f'Iteration_{i}'] = makespan
            
            # Add summary statistics
            row['Total_Iterations'] = len(iterations)
            row['First_Makespan'] = iterations[0] if iterations else None
            row['Last_Makespan'] = iterations[-1] if iterations else None
            row['Best_Makespan'] = min(iterations) if iterations else None
            row['Worst_Makespan'] = max(iterations) if iterations else None
            row['Improvement'] = iterations[0] - min(iterations) if iterations else 0
            row['Deterioration'] = max(iterations) - iterations[0] if iterations else 0
            
            all_data.append(row)
        else:
            print(f"No iterations found in {file_path}")
    
    if not all_data:
        print("No data found in any MAPLE files")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Sort by dataset name
    df = df.sort_values('Dataset')
    
    # Display the table
    print("\n" + "="*100)
    print("MAPLE ITERATIONS ANALYSIS")
    print("="*100)
    
    # Show basic info
    print(f"\nTotal datasets: {len(df)}")
    print(f"Average iterations per dataset: {df['Total_Iterations'].mean():.1f}")
    print(f"Max iterations: {df['Total_Iterations'].max()}")
    print(f"Min iterations: {df['Total_Iterations'].min()}")
    
    # Show improvement/deterioration stats
    print(f"\nDatasets with improvement: {(df['Improvement'] > 0).sum()}")
    print(f"Datasets with deterioration: {(df['Deterioration'] > 0).sum()}")
    print(f"Average improvement: {df['Improvement'].mean():.1f}")
    print(f"Average deterioration: {df['Deterioration'].mean():.1f}")
    
    # Show the main table
    print("\n" + "="*150)
    print("DETAILED ITERATIONS TABLE")
    print("="*150)
    
    # Display with all iterations
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print(df.to_string(index=False))
    
    # Show summary statistics
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    summary_stats = df[['Dataset', 'Total_Iterations', 'First_Makespan', 'Last_Makespan', 
                       'Best_Makespan', 'Worst_Makespan', 'Improvement', 'Deterioration']].copy()
    
    print(summary_stats.to_string(index=False))
    
    # Save to CSV
    output_file = "maple_iterations_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Create a detailed iterations file
    detailed_file = "maple_detailed_iterations.txt"
    with open(detailed_file, 'w') as f:
        f.write("MAPLE DETAILED ITERATIONS ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        for _, row in df.iterrows():
            f.write(f"Dataset: {row['Dataset']}\n")
            f.write(f"Total Iterations: {row['Total_Iterations']}\n")
            f.write(f"First Makespan: {row['First_Makespan']}\n")
            f.write(f"Last Makespan: {row['Last_Makespan']}\n")
            f.write(f"Best Makespan: {row['Best_Makespan']}\n")
            f.write(f"Worst Makespan: {row['Worst_Makespan']}\n")
            f.write(f"Improvement: {row['Improvement']}\n")
            f.write(f"Deterioration: {row['Deterioration']}\n")
            
            # Show all iterations
            f.write("All Iterations: ")
            iterations = []
            for col in df.columns:
                if col.startswith('Iteration_'):
                    iter_num = int(col.split('_')[1])
                    if not pd.isna(row[col]):
                        iterations.append((iter_num, int(row[col])))
            
            iterations.sort()
            for iter_num, makespan in iterations:
                f.write(f"Iter{iter_num}:{makespan} ")
            f.write("\n\n" + "-"*50 + "\n\n")
    
    print(f"Detailed analysis saved to: {detailed_file}")

if __name__ == "__main__":
    create_iterations_table()
