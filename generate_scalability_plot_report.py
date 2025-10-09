#!/usr/bin/env python3
"""
Generate scalability plot report showing optimal rate vs dataset size for each dataset.
Each dataset gets its own figure with optimal rate on y-axis and dataset size on x-axis.
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def get_upper_bound(dataset_name):
    """Get upper bound for a specific dataset"""
    # Upper bounds for different dataset categories
    upper_bounds = {
        # DMU datasets
        'dmu01': 1000, 'dmu02': 1000, 'dmu03': 1000, 'dmu04': 1000, 'dmu05': 1000,
        'dmu06': 1000, 'dmu07': 1000, 'dmu08': 1000, 'dmu09': 1000, 'dmu10': 1000,
        
        # TA datasets  
        'ta01': 1231, 'ta02': 1244, 'ta03': 1218, 'ta04': 1175, 'ta05': 1224,
        'ta06': 1238, 'ta07': 1227, 'ta08': 1217, 'ta09': 1274, 'ta10': 1241,
        'ta11': 1357, 'ta12': 1317, 'ta13': 1347, 'ta14': 1347, 'ta15': 1347,
        'ta16': 1347, 'ta17': 1347, 'ta18': 1347, 'ta19': 1347, 'ta20': 1347,
        'ta21': 1347, 'ta22': 1347, 'ta23': 1347, 'ta24': 1347, 'ta25': 1347,
        'ta26': 1347, 'ta27': 1347, 'ta28': 1347, 'ta29': 1347, 'ta30': 1347,
        'ta31': 1347, 'ta32': 1347, 'ta33': 1347, 'ta34': 1347, 'ta35': 1347,
        'ta36': 1347, 'ta37': 1347, 'ta38': 1347, 'ta39': 1347, 'ta40': 1347,
        'ta41': 1347, 'ta42': 1347, 'ta43': 1347, 'ta44': 1347, 'ta45': 1347,
        'ta46': 1347, 'ta47': 1347, 'ta48': 1347, 'ta49': 1347, 'ta50': 1347,
        'ta51': 1000, 'ta52': 1000, 'ta61': 1000, 'ta71': 1000, 'ta72': 1000,
        
        # ABZ datasets
        'abz05': 1234, 'abz06': 943, 'abz07': 656, 'abz08': 669, 'abz09': 679,
        'abz10': 958,
        
        # SWV datasets
        'swv01': 1407, 'swv02': 1475, 'swv03': 1368, 'swv04': 1450, 'swv05': 1424,
        'swv06': 1419, 'swv07': 1513, 'swv08': 1631, 'swv09': 1648, 'swv10': 1669,
        'swv11': 2983, 'swv12': 2972, 'swv13': 3101, 'swv14': 3101, 'swv15': 2885,
        'swv16': 2924, 'swv17': 2794, 'swv18': 2852, 'swv19': 2843, 'swv20': 2823,
        
        # YN datasets
        'yn01': 826, 'yn02': 861, 'yn03': 827, 'yn04': 918,
        
        # RCMAX datasets (estimated based on problem size)
        'rcmax_20_15_5': 1000, 'rcmax_20_15_8': 1200, 'rcmax_20_20_7': 1400,
        'rcmax_20_20_8': 1000, 'rcmax_30_15_4': 1500, 'rcmax_30_15_5': 1800, 
        'rcmax_30_20_8': 2000, 'rcmax_30_20_9': 2200, 'rcmax_40_15_8': 2500, 
        'rcmax_40_15_10': 2800, 'rcmax_40_20_2': 1000, 'rcmax_40_20_6': 3000,
        'rcmax_50_15_2': 1000, 'rcmax_50_15_4': 3500, 'rcmax_50_20_6': 1000,
        'rcmax_50_20_9': 1000,
    }
    
    return upper_bounds.get(dataset_name, 1000)  # Default fallback

def get_dataset_size(dataset_name):
    """Extract dataset size information from dataset name"""
    if dataset_name.startswith('rcmax_'):
        # Format: rcmax_jobs_machines_operations
        parts = dataset_name.split('_')
        if len(parts) >= 4:
            jobs = int(parts[1])
            machines = int(parts[2])
            operations = int(parts[3])
            # Use total operations as size metric
            return jobs * operations
        return 100  # Default fallback
    
    elif dataset_name.startswith('ta'):
        # TA datasets: ta01-ta50 are 15x15, ta51+ are 20x20
        ta_num = int(dataset_name[2:])
        if ta_num <= 50:
            return 15 * 15  # 15 jobs, 15 machines
        else:
            return 20 * 20  # 20 jobs, 20 machines
    
    elif dataset_name.startswith('abz'):
        # ABZ datasets: abz05-abz10 are 10x10
        return 10 * 10
    
    elif dataset_name.startswith('swv'):
        # SWV datasets: swv01-swv10 are 20x10, swv11+ are 50x10
        swv_num = int(dataset_name[3:])
        if swv_num <= 10:
            return 20 * 10  # 20 jobs, 10 machines
        else:
            return 50 * 10  # 50 jobs, 10 machines
    
    elif dataset_name.startswith('yn'):
        # YN datasets: yn01-yn04 are 20x20
        return 20 * 20
    
    elif dataset_name.startswith('dmu'):
        # DMU datasets: dmu01-dmu10 are 10x10
        return 10 * 10
    
    return 100  # Default fallback

def calculate_optimal_rate(makespan, dataset_name):
    """Calculate optimal rate as makespan/ub * 100%"""
    if makespan is None or makespan == 0:
        return None
    
    ub = get_upper_bound(dataset_name)
    optimal_rate = (makespan / ub) * 100
    return round(optimal_rate, 2)

def extract_makespan_from_result(result):
    """Extract makespan from result data"""
    if isinstance(result, dict):
        return result.get('makespan')
    return None

def process_single_agent_results():
    """Process single-agent model results"""
    results = []
    
    # Single-agent models
    models = ['GPT-4o', 'Claude-Sonnet-4', 'Gemini-2.5', 'DeepSeek-V3']
    results_dir = "results_single(gpt-4o)"
    
    if not os.path.exists(results_dir):
        return results
    
    for filename in os.listdir(results_dir):
        if filename.startswith('singleagent_llm_comparison_') and filename.endswith('.json'):
            dataset_name = filename.replace('singleagent_llm_comparison_', '').replace('.json', '')
            # Extract dataset name (remove model suffix)
            for model in models:
                if dataset_name.endswith(f'_{model}'):
                    dataset_name = dataset_name.replace(f'_{model}', '')
                    break
            
            file_path = os.path.join(results_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if 'models' in data:
                    for model_name, model_data in data['models'].items():
                        makespan = model_data.get('makespan')
                        if makespan is not None:
                            optimal_rate = calculate_optimal_rate(makespan, dataset_name)
                            if optimal_rate is not None:
                                dataset_size = get_dataset_size(dataset_name)
                                results.append({
                                    'dataset': dataset_name,
                                    'model': model_name,
                                    'source': 'Single-Agent',
                                    'makespan': makespan,
                                    'optimal_rate': optimal_rate,
                                    'dataset_size': dataset_size
                                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    return results

def process_mas_results():
    """Process multi-agent system results"""
    results = []
    
    # MAS frameworks
    frameworks = ['AutoGen', 'CrewAI', 'LangGraph', 'OpenAI_Swarm']
    result_dirs = ["results_mas(gpt-4o)", "results_mas(claude-4)"]
    
    for results_dir in result_dirs:
        if not os.path.exists(results_dir):
            continue
            
        source_name = results_dir.replace("results_mas(", "").replace(")", "")
        
        for filename in os.listdir(results_dir):
            if filename.startswith('jssp_results_') and filename.endswith('.json'):
                dataset_name = filename.replace('jssp_results_', '').replace('.json', '')
                # Extract dataset name (remove framework suffix)
                for framework in frameworks:
                    if dataset_name.endswith(f'_{framework}'):
                        dataset_name = dataset_name.replace(f'_{framework}', '')
                        break
                
                file_path = os.path.join(results_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'frameworks' in data:
                        for framework_name, framework_data in data['frameworks'].items():
                            makespan = framework_data.get('makespan')
                            if makespan is not None:
                                optimal_rate = calculate_optimal_rate(makespan, dataset_name)
                                if optimal_rate is not None:
                                    dataset_size = get_dataset_size(dataset_name)
                                    results.append({
                                        'dataset': dataset_name,
                                        'model': framework_name,
                                        'source': f'MAS-{source_name}',
                                        'makespan': makespan,
                                        'optimal_rate': optimal_rate,
                                        'dataset_size': dataset_size
                                    })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
    
    return results

def process_alas_results():
    """Process ALAS (ours) results"""
    results = []
    
    # ALAS result directories
    alas_dirs = [
        "results_optimized(gpt-4o)",
        "results_optimized(claude-4)", 
        "results_optimized(deepseek-v3)",
        "results_optimized(gemini-2.5)"
    ]
    
    workflows = ['full', 'no_repair', 'no_validation', 'no_optimization']
    
    for results_dir in alas_dirs:
        if not os.path.exists(results_dir):
            continue
            
        source_name = results_dir.replace("results_optimized(", "").replace(")", "")
        
        for filename in os.listdir(results_dir):
            if filename.endswith('_workflow_comparison.json'):
                dataset_name = filename.replace('_workflow_comparison.json', '')
                
                file_path = os.path.join(results_dir, filename)
                
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if 'workflow_results' in data:
                        for workflow_name, workflow_data in data['workflow_results'].items():
                            if workflow_name in workflows:
                                makespan = workflow_data.get('makespan')
                                if makespan is not None:
                                    optimal_rate = calculate_optimal_rate(makespan, dataset_name)
                                    if optimal_rate is not None:
                                        dataset_size = get_dataset_size(dataset_name)
                                        results.append({
                                            'dataset': dataset_name,
                                            'model': workflow_name,
                                            'source': f'ALAS-{source_name}',
                                            'makespan': makespan,
                                            'optimal_rate': optimal_rate,
                                            'dataset_size': dataset_size
                                        })
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue
    
    return results

def create_scalability_plot(dataset_name, results, output_dir):
    """Create scalability plot for a specific dataset"""
    if not results:
        return
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Group results by source
    source_groups = defaultdict(list)
    for result in results:
        source_groups[result['source']].append(result)
    
    # Define colors and markers for different sources
    colors = {
        'Single-Agent': 'blue',
        'MAS-gpt-4o': 'green',
        'MAS-claude-4': 'orange',
        'ALAS-gpt-4o': 'red',
        'ALAS-claude-4': 'purple',
        'ALAS-deepseek-v3': 'brown',
        'ALAS-gemini-2.5': 'pink'
    }
    
    markers = {
        'Single-Agent': 'o',
        'MAS-gpt-4o': 's',
        'MAS-claude-4': '^',
        'ALAS-gpt-4o': 'v',
        'ALAS-claude-4': 'D',
        'ALAS-deepseek-v3': 'h',
        'ALAS-gemini-2.5': 'p'
    }
    
    # Plot each source
    for source, source_results in source_groups.items():
        if not source_results:
            continue
            
        # Sort by dataset size
        source_results.sort(key=lambda x: x['dataset_size'])
        
        sizes = [r['dataset_size'] for r in source_results]
        rates = [r['optimal_rate'] for r in source_results]
        
        color = colors.get(source, 'black')
        marker = markers.get(source, 'o')
        
        plt.scatter(sizes, rates, 
                   color=color, marker=marker, s=100, 
                   alpha=0.7, label=source, edgecolors='black', linewidth=0.5)
        
        # Add trend line (only if we have enough points and they're not all the same)
        if len(sizes) > 1 and len(set(sizes)) > 1:
            try:
                z = np.polyfit(sizes, rates, 1)
                p = np.poly1d(z)
                plt.plot(sizes, p(sizes), color=color, linestyle='--', alpha=0.5)
            except np.RankWarning:
                # Skip trend line if polyfit fails
                pass
    
    # Customize the plot
    plt.xlabel('Dataset Size (Total Operations)', fontsize=12)
    plt.ylabel('Optimal Rate (%)', fontsize=12)
    plt.title(f'Scalability Analysis: {dataset_name.upper()}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set axis limits
    if results:
        all_sizes = [r['dataset_size'] for r in results]
        all_rates = [r['optimal_rate'] for r in results]
        plt.xlim(min(all_sizes) * 0.9, max(all_sizes) * 1.1)
        plt.ylim(0, max(all_rates) * 1.1)
    
    # Add horizontal line at 100% (optimal)
    plt.axhline(y=100, color='red', linestyle='-', alpha=0.5, label='Optimal (100%)')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"scalability_{dataset_name.lower()}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"📊 Saved scalability plot: {plot_path}")

def generate_scalability_plots():
    """Generate scalability plots for all datasets"""
    print("📊 Generating Scalability Plot Report")
    print("=" * 60)
    
    all_results = []
    
    # Process all result types
    print("📈 Processing Single-Agent results...")
    single_agent_results = process_single_agent_results()
    all_results.extend(single_agent_results)
    print(f"  Found {len(single_agent_results)} single-agent instances")
    
    print("📈 Processing MAS results...")
    mas_results = process_mas_results()
    all_results.extend(mas_results)
    print(f"  Found {len(mas_results)} MAS instances")
    
    print("📈 Processing ALAS results...")
    alas_results = process_alas_results()
    all_results.extend(alas_results)
    print(f"  Found {len(alas_results)} ALAS instances")
    
    print(f"\n📊 Total instances: {len(all_results)}")
    
    # Group by dataset
    dataset_results = defaultdict(list)
    for result in all_results:
        dataset_results[result['dataset']].append(result)
    
    # Create plots directory
    output_dir = "plots/scalability"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plot for each dataset (limit to first 20 to avoid too many plots)
    print(f"\n📊 Creating scalability plots...")
    plot_count = 0
    max_plots = 20
    
    for dataset_name, results in dataset_results.items():
        if results and plot_count < max_plots:
            create_scalability_plot(dataset_name, results, output_dir)
            plot_count += 1
        elif plot_count >= max_plots:
            print(f"  ⚠️  Limited to first {max_plots} datasets to avoid too many plots")
            break
    
    print(f"\n✅ Generated {plot_count} scalability plots in {output_dir}")
    
    # Generate summary statistics
    print(f"\n📊 SCALABILITY SUMMARY")
    print("=" * 60)
    
    # Calculate average optimal rate by dataset size
    size_groups = defaultdict(list)
    for result in all_results:
        size_groups[result['dataset_size']].append(result['optimal_rate'])
    
    print("Average Optimal Rate by Dataset Size:")
    for size in sorted(size_groups.keys()):
        rates = size_groups[size]
        avg_rate = sum(rates) / len(rates)
        print(f"  Size {size:>6}: {avg_rate:>6.2f}% (n={len(rates)})")
    
    # Calculate correlation between size and optimal rate
    if all_results:
        sizes = [r['dataset_size'] for r in all_results]
        rates = [r['optimal_rate'] for r in all_results]
        correlation = np.corrcoef(sizes, rates)[0, 1]
        print(f"\nCorrelation between dataset size and optimal rate: {correlation:.3f}")
        
        if correlation > 0.1:
            print("  → Positive correlation: Larger datasets tend to have higher optimal rates")
        elif correlation < -0.1:
            print("  → Negative correlation: Larger datasets tend to have lower optimal rates")
        else:
            print("  → Weak correlation: Dataset size has little effect on optimal rate")

if __name__ == "__main__":
    generate_scalability_plots()
