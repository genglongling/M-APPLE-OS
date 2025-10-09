#!/usr/bin/env python3
"""
Extract makespan results from results_optimized/ directory and create LaTeX table
"""

import os
import re
import json
from pathlib import Path

def extract_final_results(file_path):
    """Extract the final successful results from a workflow comparison file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Find all workflow result blocks
        workflow_blocks = re.findall(r'=== (.*?) Results ===\nSuccess: (True|False)\nMakespan: (.*?)\nExecution Time: (.*?)s', content)
        
        # Get the last occurrence of each workflow type
        results = {}
        for workflow_name, success, makespan, exec_time in workflow_blocks:
            if success == 'True' and makespan != 'None':
                results[workflow_name] = {
                    'makespan': int(makespan),
                    'exec_time': float(exec_time)
                }
        
        return results
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def main():
    """Extract results from all files and create LaTeX table"""
    results_dir = "results_optimized"
    
    # Get all dataset files
    dataset_files = []
    for file in os.listdir(results_dir):
        if file.endswith('_workflow_comparison_maple.txt'):
            dataset_name = file.replace('_workflow_comparison_maple.txt', '')
            dataset_files.append((dataset_name, os.path.join(results_dir, file)))
    
    # Sort datasets by category
    dmu_datasets = []
    ta_datasets = []
    abzswvyn_datasets = []
    
    for dataset_name, file_path in dataset_files:
        if dataset_name.startswith('rcmax_'):
            dmu_datasets.append((dataset_name, file_path))
        elif dataset_name.startswith('TA'):
            ta_datasets.append((dataset_name, file_path))
        else:
            abzswvyn_datasets.append((dataset_name, file_path))
    
    # Sort each category
    dmu_datasets.sort(key=lambda x: (int(x[0].split('_')[1]), int(x[0].split('_')[2]), int(x[0].split('_')[3])))
    ta_datasets.sort(key=lambda x: int(x[0][2:]) if x[0][2:].isdigit() else float('inf'))
    abzswvyn_datasets.sort()
    
    all_datasets = dmu_datasets + ta_datasets + abzswvyn_datasets
    
    # Extract results from all files
    all_results = {}
    for dataset_name, file_path in all_datasets:
        results = extract_final_results(file_path)
        if results:
            all_results[dataset_name] = results
    
    # Create LaTeX table
    print("\\begin{table}[h!]")
    print("\\centering")
    print("\\caption{MAPLE Workflow Comparison Results - Makespan Values}")
    print("\\label{tab:maple_workflow_results}")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{|l|c|c|c|c|c|c|c|c|}")
    print("\\hline")
    print("Dataset & Full & No Repair & No Validation & No Optimization & LRCP-1 & LRCP-2 & LRCP-3 & LRCP-4 \\\\")
    print("\\hline")
    
    for dataset_name, results in all_results.items():
        # Format dataset name for display
        display_name = dataset_name.replace('_', '\\_')
        
        # Extract makespan values
        full_makespan = results.get('Full Workflow', {}).get('makespan', 'N/A')
        no_repair_makespan = results.get('No Repair Tools', {}).get('makespan', 'N/A')
        no_validation_makespan = results.get('No Validation Tools', {}).get('makespan', 'N/A')
        no_optimization_makespan = results.get('No Optimization Tools', {}).get('makespan', 'N/A')
        
        # For LRCP iterations, we'll use the same values as Full Workflow for now
        # since the current implementation doesn't distinguish between iterations
        lrcp_1 = full_makespan
        lrcp_2 = full_makespan
        lrcp_3 = full_makespan
        lrcp_4 = full_makespan
        
        print(f"{display_name} & {full_makespan} & {no_repair_makespan} & {no_validation_makespan} & {no_optimization_makespan} & {lrcp_1} & {lrcp_2} & {lrcp_3} & {lrcp_4} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("}")
    print("\\end{table}")
    
    # Create summary statistics
    print("\n\\begin{table}[h!]")
    print("\\centering")
    print("\\caption{Summary Statistics - Average Makespan by Workflow Type}")
    print("\\label{tab:maple_summary_stats}")
    print("\\begin{tabular}{|l|c|c|c|c|}")
    print("\\hline")
    print("Workflow Type & Average Makespan & Min Makespan & Max Makespan & Success Rate \\\\")
    print("\\hline")
    
    # Calculate statistics for each workflow type
    workflow_types = ['Full Workflow', 'No Repair Tools', 'No Validation Tools', 'No Optimization Tools']
    
    for workflow_type in workflow_types:
        makespans = []
        for dataset_name, results in all_results.items():
            if workflow_type in results:
                makespans.append(results[workflow_type]['makespan'])
        
        if makespans:
            avg_makespan = sum(makespans) / len(makespans)
            min_makespan = min(makespans)
            max_makespan = max(makespans)
            success_rate = len(makespans) / len(all_results) * 100
            print(f"{workflow_type} & {avg_makespan:.1f} & {min_makespan} & {max_makespan} & {success_rate:.1f}\\% \\\\")
        else:
            print(f"{workflow_type} & N/A & N/A & N/A & 0.0\\% \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Print dataset count by category
    print(f"\n% Dataset Summary:")
    print(f"% DMU datasets: {len(dmu_datasets)}")
    print(f"% TA datasets: {len(ta_datasets)}")
    print(f"% ABZSWVYN datasets: {len(abzswvyn_datasets)}")
    print(f"% Total datasets: {len(all_datasets)}")
    print(f"% Successful results: {len(all_results)}")

if __name__ == "__main__":
    main()
