#!/usr/bin/env python3
"""
Generate line plots showing error rate vs repair iteration (1-5) for each dataset category.
Each instance is shown as a separate line.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def get_dataset_category(dataset_name):
    """Get dataset category from dataset name"""
    if dataset_name.startswith('rcmax'):
        return "RCMAX"
    elif dataset_name.startswith('ta'):
        return "TA"
    elif dataset_name.startswith('abz'):
        return "ABZ"
    elif dataset_name.startswith('swv'):
        return "SWV"
    elif dataset_name.startswith('yn'):
        return "YN"
    else:
        return "Unknown"

def extract_error_count_from_log(dataset_name, iteration, results_dir):
    """Extract actual error count from terminal output log"""
    log_file = os.path.join(results_dir, f"{dataset_name}_terminal_output.log")
    
    if not os.path.exists(log_file):
        return 0
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for error count patterns in the log
        # Pattern: "Repair iteration X reduced errors from Y to Z"
        error_patterns = [
            rf"Repair iteration {iteration} reduced errors from (\d+) to (\d+)",
            rf"⚠️ Repair iteration {iteration} reduced errors from (\d+) to (\d+)",
            rf"iteration {iteration}.*?reduced errors from (\d+) to (\d+)",
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the "to" value (final error count after repair)
                return int(matches[0][1])
        
        # If no specific iteration found, look for any error count
        general_patterns = [
            r"reduced errors from (\d+) to (\d+)",
            r"⚠️.*?reduced errors from (\d+) to (\d+)",
        ]
        
        for pattern in general_patterns:
            matches = re.findall(pattern, content)
            if matches:
                # Return the last "to" value
                return int(matches[-1][1])
        
        # Look for missing jobs count
        missing_jobs_pattern = r"Missing jobs in schedule: \[(.*?)\]"
        missing_matches = re.findall(missing_jobs_pattern, content)
        if missing_matches:
            # Count the number of missing jobs
            missing_jobs_str = missing_matches[0]
            missing_jobs = [job.strip().strip("'\"") for job in missing_jobs_str.split(',')]
            return len([job for job in missing_jobs if job])
        
        return 0
        
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return 0

def calculate_error_rate(schedule, dataset_name, iteration, results_dir):
    """Calculate error rate from schedule and terminal log"""
    if not schedule or len(schedule) == 0:
        return 100.0  # 100% error if no schedule
    
    # Try to extract actual error count from terminal log
    actual_errors = extract_error_count_from_log(dataset_name, iteration, results_dir)
    
    if actual_errors > 0:
        # Use actual error count from log
        total_operations = len(schedule)
        if total_operations == 0:
            return 100.0
        error_rate = (actual_errors / total_operations) * 100
        return min(error_rate, 100.0)
    
    # Fallback: count basic validation errors from schedule
    error_count = 0
    total_operations = len(schedule)
    
    # Check for basic validation errors
    for operation in schedule:
        # Check for invalid time entries
        if operation.get('start', 0) >= operation.get('end', 0):
            error_count += 1
        # Check for negative times
        if operation.get('start', 0) < 0 or operation.get('end', 0) < 0:
            error_count += 1
        # Check for zero or negative duration
        if operation.get('duration', 0) <= 0:
            error_count += 1
    
    # Calculate error rate as percentage
    if total_operations == 0:
        return 100.0
    
    error_rate = (error_count / total_operations) * 100
    return min(error_rate, 100.0)  # Cap at 100%

def extract_repair_iteration_data(results_dir):
    """Extract repair iteration data from all workflow comparison files"""
    data_by_category = defaultdict(lambda: defaultdict(list))
    
    # Find all workflow comparison files
    for filename in os.listdir(results_dir):
        if filename.endswith('_workflow_comparison.json'):
            dataset_name = filename.replace('_workflow_comparison.json', '')
            category = get_dataset_category(dataset_name)
            
            if category == "Unknown":
                continue
                
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract repair iterations from full workflow
                if 'workflow_results' in data and 'full' in data['workflow_results']:
                    full_workflow = data['workflow_results']['full']
                    if 'repair_iterations' in full_workflow:
                        repair_iterations = full_workflow['repair_iterations']
                        
                        # Extract error rate for each iteration
                        iteration_data = []
                        for i, iteration in enumerate(repair_iterations, 1):
                            if i <= 5:  # Only first 5 iterations
                                schedule = iteration.get('schedule', [])
                                error_rate = calculate_error_rate(schedule, dataset_name, i, results_dir)
                                iteration_data.append((i, error_rate))
                        
                        if len(iteration_data) > 0:
                            data_by_category[category][dataset_name] = iteration_data
                            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    return data_by_category

def create_plots(data_by_category, output_dir="plots"):
    """Create line plots for each dataset category"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    
    for category, instances in data_by_category.items():
        if not instances:
            continue
            
        print(f"Creating plot for {category} category with {len(instances)} instances")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each instance as a line
        colors = plt.cm.tab10(np.linspace(0, 1, len(instances)))
        
        for i, (instance_name, iteration_data) in enumerate(instances.items()):
            if len(iteration_data) == 0:
                continue
                
            iterations = [item[0] for item in iteration_data]
            error_rates = [item[1] for item in iteration_data]
            
            # Plot line for this instance
            ax.plot(iterations, error_rates, 
                   marker='o', linewidth=2, markersize=6,
                   color=colors[i], alpha=0.7,
                   label=instance_name)
        
        # Customize plot
        ax.set_xlabel('Repair Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Error Rate vs Repair Iteration - {category} Datasets', 
                    fontsize=14, fontweight='bold')
        
        # Set axis limits
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 105)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend (only if not too many instances)
        if len(instances) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # For many instances, add a note about the number
            ax.text(0.02, 0.98, f'{len(instances)} instances', 
                   transform=ax.transAxes, fontsize=10, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set x-axis ticks
        ax.set_xticks([1, 2, 3, 4, 5])
        
        # Tight layout
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_dir, f'error_rate_vs_iteration_{category.lower()}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        
        # Also save as PDF for better quality
        pdf_file = os.path.join(output_dir, f'error_rate_vs_iteration_{category.lower()}.pdf')
        plt.savefig(pdf_file, bbox_inches='tight')
        print(f"Saved plot: {pdf_file}")
        
        plt.close()

def create_summary_plot(data_by_category, output_dir="plots"):
    """Create a summary plot showing all categories"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    categories = list(data_by_category.keys())
    
    for i, category in enumerate(categories):
        if i >= len(axes):
            break
            
        ax = axes[i]
        instances = data_by_category[category]
        
        if not instances:
            ax.text(0.5, 0.5, f'No data for {category}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{category} Datasets')
            continue
        
        # Plot each instance as a line
        colors = plt.cm.tab10(np.linspace(0, 1, len(instances)))
        
        for j, (instance_name, iteration_data) in enumerate(instances.items()):
            if len(iteration_data) == 0:
                continue
                
            iterations = [item[0] for item in iteration_data]
            error_rates = [item[1] for item in iteration_data]
            
            # Plot line for this instance
            ax.plot(iterations, error_rates, 
                   marker='o', linewidth=1.5, markersize=4,
                   color=colors[j], alpha=0.7)
        
        # Customize subplot
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title(f'{category} Datasets ({len(instances)} instances)')
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 4, 5])
    
    # Hide unused subplots
    for i in range(len(categories), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Error Rate vs Repair Iteration - All Dataset Categories', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary plot
    summary_file = os.path.join(output_dir, 'error_rate_vs_iteration_all_categories.png')
    plt.savefig(summary_file, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot: {summary_file}")
    
    plt.close()

def main():
    """Main function to generate repair iteration plots"""
    print("🔍 Generating Repair Iteration Error Rate Plots")
    print("=" * 60)
    
    # Define all result directories
    result_dirs = [
        "results_optimized(gpt-4o)",
        "results_optimized(claude-4)", 
        "results_optimized(deepseek-v3)",
        "results_optimized(gemini-2.5)"
    ]
    
    # Extract data from all directories
    all_data_by_category = defaultdict(lambda: defaultdict(list))
    
    for results_dir in result_dirs:
        if not os.path.exists(results_dir):
            print(f"⚠️ Results directory not found: {results_dir}")
            continue
            
        print(f"📊 Extracting data from {results_dir}...")
        data_by_category = extract_repair_iteration_data(results_dir)
        
        # Merge data from this directory
        for category, instances in data_by_category.items():
            for instance_name, iteration_data in instances.items():
                # Add source directory to instance name to avoid conflicts
                source_name = results_dir.replace("results_optimized(", "").replace(")", "")
                unique_instance_name = f"{instance_name}_{source_name}"
                all_data_by_category[category][unique_instance_name] = iteration_data
    
    # Print summary
    total_instances = sum(len(instances) for instances in all_data_by_category.values())
    print(f"📈 Found {total_instances} instances across {len(all_data_by_category)} categories")
    
    for category, instances in all_data_by_category.items():
        print(f"  {category}: {len(instances)} instances")
    
    # Create individual plots for each category
    print("\n📊 Creating individual plots...")
    create_plots(all_data_by_category)
    
    # Create summary plot
    print("\n📊 Creating summary plot...")
    create_summary_plot(all_data_by_category)
    
    print("\n✅ All plots generated successfully!")
    print("📁 Check the 'plots' directory for the generated figures")

if __name__ == "__main__":
    main()
