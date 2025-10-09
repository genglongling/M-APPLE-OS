#!/usr/bin/env python3
"""
Generate scalability summary report showing optimal rate vs dataset size.
This version focuses on generating a text report without matplotlib issues.
"""

import json
import os
import re
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

def generate_scalability_summary():
    """Generate scalability summary report"""
    print("📊 Generating Scalability Summary Report")
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
    
    # Group by dataset size
    size_groups = defaultdict(list)
    for result in all_results:
        size_groups[result['dataset_size']].append(result)
    
    # Generate summary report
    print(f"\n📋 SCALABILITY SUMMARY BY DATASET SIZE")
    print("=" * 80)
    
    # Sort by dataset size
    sorted_sizes = sorted(size_groups.keys())
    
    for size in sorted_sizes:
        results = size_groups[size]
        if not results:
            continue
            
        # Calculate statistics
        optimal_rates = [r['optimal_rate'] for r in results]
        avg_rate = sum(optimal_rates) / len(optimal_rates)
        min_rate = min(optimal_rates)
        max_rate = max(optimal_rates)
        
        # Count by source
        source_counts = defaultdict(int)
        for result in results:
            source_counts[result['source']] += 1
        
        print(f"\n🔍 Dataset Size: {size} operations")
        print("-" * 50)
        print(f"  Instances: {len(results)}")
        print(f"  Average Optimal Rate: {avg_rate:.2f}%")
        print(f"  Best Rate: {min_rate:.2f}%")
        print(f"  Worst Rate: {max_rate:.2f}%")
        print(f"  Sources: {dict(source_counts)}")
        
        # Show best and worst performers
        results.sort(key=lambda x: x['optimal_rate'])
        print(f"  Best: {results[0]['source']} {results[0]['model']} ({results[0]['optimal_rate']:.2f}%)")
        print(f"  Worst: {results[-1]['source']} {results[-1]['model']} ({results[-1]['optimal_rate']:.2f}%)")
    
    # Generate correlation analysis
    print(f"\n📊 CORRELATION ANALYSIS")
    print("=" * 60)
    
    if all_results:
        sizes = [r['dataset_size'] for r in all_results]
        rates = [r['optimal_rate'] for r in all_results]
        
        # Calculate correlation coefficient
        n = len(sizes)
        if n > 1:
            mean_size = sum(sizes) / n
            mean_rate = sum(rates) / n
            
            numerator = sum((sizes[i] - mean_size) * (rates[i] - mean_rate) for i in range(n))
            denominator = (sum((sizes[i] - mean_size) ** 2 for i in range(n)) * 
                          sum((rates[i] - mean_rate) ** 2 for i in range(n))) ** 0.5
            
            if denominator != 0:
                correlation = numerator / denominator
                print(f"Correlation between dataset size and optimal rate: {correlation:.3f}")
                
                if correlation > 0.1:
                    print("  → Positive correlation: Larger datasets tend to have higher optimal rates")
                elif correlation < -0.1:
                    print("  → Negative correlation: Larger datasets tend to have lower optimal rates")
                else:
                    print("  → Weak correlation: Dataset size has little effect on optimal rate")
            else:
                print("  → Cannot calculate correlation (no variance in data)")
    
    # Generate source comparison
    print(f"\n📊 SOURCE COMPARISON BY DATASET SIZE")
    print("=" * 60)
    
    source_groups = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        source_groups[result['source']][result['dataset_size']].append(result['optimal_rate'])
    
    for source in sorted(source_groups.keys()):
        print(f"\n{source}:")
        for size in sorted(source_groups[source].keys()):
            rates = source_groups[source][size]
            avg_rate = sum(rates) / len(rates)
            print(f"  Size {size:>6}: {avg_rate:>6.2f}% (n={len(rates)})")
    
    print(f"\n✅ Scalability summary report generated successfully!")

if __name__ == "__main__":
    generate_scalability_summary()

