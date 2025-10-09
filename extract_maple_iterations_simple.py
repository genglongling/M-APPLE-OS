#!/usr/bin/env python3
"""
Extract MAPLE makespan values from all iterations and create a comprehensive table
"""

import os
import re
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

def format_number(num, width=8):
    """Format number with fixed width"""
    if num is None:
        return " " * width
    return f"{num:>{width}}"

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
    max_iterations = 0
    
    for file_path in maple_files:
        dataset_name = extract_dataset_name(file_path)
        iterations = extract_maple_iterations(file_path)
        
        if iterations:
            max_iterations = max(max_iterations, len(iterations))
            
            # Create a row for each dataset
            row = {
                'dataset': dataset_name,
                'iterations': iterations,
                'total_iterations': len(iterations),
                'first_makespan': iterations[0] if iterations else None,
                'last_makespan': iterations[-1] if iterations else None,
                'best_makespan': min(iterations) if iterations else None,
                'worst_makespan': max(iterations) if iterations else None,
                'improvement': iterations[0] - min(iterations) if iterations else 0,
                'deterioration': max(iterations) - iterations[0] if iterations else 0
            }
            all_data.append(row)
        else:
            print(f"No iterations found in {file_path}")
    
    if not all_data:
        print("No data found in any MAPLE files")
        return
    
    # Sort by dataset name
    all_data.sort(key=lambda x: x['dataset'])
    
    # Display the table
    print("\n" + "="*120)
    print("MAPLE ITERATIONS ANALYSIS")
    print("="*120)
    
    # Show basic info
    print(f"\nTotal datasets: {len(all_data)}")
    avg_iterations = sum(row['total_iterations'] for row in all_data) / len(all_data)
    print(f"Average iterations per dataset: {avg_iterations:.1f}")
    print(f"Max iterations: {max_iterations}")
    print(f"Min iterations: {min(row['total_iterations'] for row in all_data)}")
    
    # Show improvement/deterioration stats
    improved = sum(1 for row in all_data if row['improvement'] > 0)
    deteriorated = sum(1 for row in all_data if row['deterioration'] > 0)
    avg_improvement = sum(row['improvement'] for row in all_data) / len(all_data)
    avg_deterioration = sum(row['deterioration'] for row in all_data) / len(all_data)
    
    print(f"\nDatasets with improvement: {improved}")
    print(f"Datasets with deterioration: {deteriorated}")
    print(f"Average improvement: {avg_improvement:.1f}")
    print(f"Average deterioration: {avg_deterioration:.1f}")
    
    # Show the main table
    print("\n" + "="*150)
    print("DETAILED ITERATIONS TABLE")
    print("="*150)
    
    # Header
    header = "Dataset".ljust(20)
    for i in range(1, max_iterations + 1):
        header += f"Iter{i}".rjust(8)
    header += " | First".rjust(8) + " | Last".rjust(8) + " | Best".rjust(8) + " | Worst".rjust(8) + " | Impr".rjust(8) + " | Detr".rjust(8)
    
    print(header)
    print("-" * len(header))
    
    # Data rows
    for row in all_data:
        line = row['dataset'].ljust(20)
        
        # Add iteration values
        for i in range(1, max_iterations + 1):
            if i <= len(row['iterations']):
                line += format_number(row['iterations'][i-1])
            else:
                line += " " * 8
        
        # Add summary columns
        line += " | " + format_number(row['first_makespan'])
        line += " | " + format_number(row['last_makespan'])
        line += " | " + format_number(row['best_makespan'])
        line += " | " + format_number(row['worst_makespan'])
        line += " | " + format_number(row['improvement'])
        line += " | " + format_number(row['deterioration'])
        
        print(line)
    
    # Save to CSV
    output_file = "maple_iterations_analysis.csv"
    with open(output_file, 'w') as f:
        # Write header
        f.write("Dataset,Total_Iterations,First_Makespan,Last_Makespan,Best_Makespan,Worst_Makespan,Improvement,Deterioration")
        for i in range(1, max_iterations + 1):
            f.write(f",Iteration_{i}")
        f.write("\n")
        
        # Write data
        for row in all_data:
            f.write(f"{row['dataset']},{row['total_iterations']},{row['first_makespan']},{row['last_makespan']},{row['best_makespan']},{row['worst_makespan']},{row['improvement']},{row['deterioration']}")
            for i in range(1, max_iterations + 1):
                if i <= len(row['iterations']):
                    f.write(f",{row['iterations'][i-1]}")
                else:
                    f.write(",")
            f.write("\n")
    
    print(f"\nResults saved to: {output_file}")
    
    # Create a detailed iterations file
    detailed_file = "maple_detailed_iterations.txt"
    with open(detailed_file, 'w') as f:
        f.write("MAPLE DETAILED ITERATIONS ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        for row in all_data:
            f.write(f"Dataset: {row['dataset']}\n")
            f.write(f"Total Iterations: {row['total_iterations']}\n")
            f.write(f"First Makespan: {row['first_makespan']}\n")
            f.write(f"Last Makespan: {row['last_makespan']}\n")
            f.write(f"Best Makespan: {row['best_makespan']}\n")
            f.write(f"Worst Makespan: {row['worst_makespan']}\n")
            f.write(f"Improvement: {row['improvement']}\n")
            f.write(f"Deterioration: {row['deterioration']}\n")
            
            # Show all iterations
            f.write("All Iterations: ")
            for i, makespan in enumerate(row['iterations'], 1):
                f.write(f"Iter{i}:{makespan} ")
            f.write("\n\n" + "-"*50 + "\n\n")
    
    print(f"Detailed analysis saved to: {detailed_file}")

if __name__ == "__main__":
    create_iterations_table()
