#!/usr/bin/env python3
"""
Generate Optimal Rate Report by Dataset Category
Shows optimal rates (makespan-ub/ub *100%) for each of the 5 dataset categories in columns
"""

import json
import os
from collections import defaultdict

def load_validation_results():
    """Load the comprehensive validation results"""
    results_file = "comprehensive_initial_schedule_validation_with_alas.json"
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def categorize_dataset(dataset_name):
    """Categorize dataset into one of the 5 categories"""
    if dataset_name.startswith('rcmax_'):
        return "DMU"
    elif dataset_name.startswith('TA'):
        return "TA"
    elif dataset_name.startswith('abz'):
        return "ABZ"
    elif dataset_name.startswith('swv'):
        return "SWV"
    elif dataset_name.startswith('yn'):
        return "YN"
    else:
        return "Unknown"

def get_upper_bound(dataset_name):
    """Get upper bound for a dataset"""
    # Real upper bounds from validation_tools.py
    upper_bounds = {
        # DMU datasets
        "rcmax_20_15_5": 2731, "rcmax_20_15_8": 2669, "rcmax_20_20_7": 3188, "rcmax_20_20_8": 3092,
        "rcmax_30_15_5": 3681, "rcmax_30_15_4": 3394, "rcmax_30_20_9": 3844, "rcmax_30_20_8": 3764,
        "rcmax_40_15_10": 4668, "rcmax_40_15_8": 4648, "rcmax_40_20_6": 4692, "rcmax_40_20_2": 4691,
        "rcmax_50_15_2": 5385, "rcmax_50_15_4": 5385, "rcmax_50_20_6": 5713, "rcmax_50_20_9": 5747,
        # TA datasets
        "TA01": 1231, "TA02": 1039, "TA51": 1234, "TA52": 1000, "TA61": 1231, "TA71": 1659, "TA72": 1247,
        # ABZ datasets
        "abz07": 656, "abz08": 665, "abz09": 679,
        # SWV datasets
        "swv01": 1397, "swv02": 1250, "swv03": 1268, "swv04": 616, "swv05": 1294, "swv06": 1268,
        "swv07": 1478, "swv08": 1500, "swv09": 1659, "swv10": 1234, "swv11": 1435, "swv12": 1794,
        "swv13": 1547, "swv14": 1000, "swv15": 1000,
        # YN datasets
        "yn01": 1165, "yn02": 1000, "yn03": 892, "yn04": 1165
    }
    return upper_bounds.get(dataset_name, 1000)  # Default upper bound

def calculate_optimal_rate(result, dataset_name):
    """Calculate optimal gap as makespan / ub * 100%"""
    if result['makespan'] is None or result['makespan'] == 0:
        return 100.0  # 100% gap if no makespan
    
    ub = get_upper_bound(dataset_name)
    makespan = result['makespan']
    
    if ub == 0:
        return 0.0
    
    return (makespan / ub) * 100

def generate_optimal_rate_report():
    """Generate optimal rate report by dataset category"""
    print("📊 Generating Optimal Rate Report by Dataset Category")
    print("="*80)
    
    # Load validation results
    results = load_validation_results()
    if not results:
        return
    
    # Group results by source, framework, and dataset category
    category_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total_deviation': 0, 'count': 0})))
    
    for result in results:
        source = result['source']
        framework = result['framework']
        dataset = result['dataset']
        category = categorize_dataset(dataset)
        
        if category != "Unknown":
            deviation = calculate_optimal_rate(result, dataset)
            
            category_stats[source][framework][category]['total_deviation'] += deviation
            category_stats[source][framework][category]['count'] += 1
    
    # Define the 5 dataset categories
    categories = ["DMU", "TA", "ABZ", "SWV", "YN"]
    
    # Define all sources and frameworks
    sources = ["MAS-GPT4o", "MAS-Claude4", "Single", "ALAS-GPT4o", "ALAS-Claude4", "ALAS-DeepSeek-V3", "ALAS-Gemini-2.5"]
    mas_frameworks_gpt4o = ["AutoGen", "CrewAI", "LangGraph", "OpenAI_Swarm"]
    mas_frameworks_claude4 = ["AutoGen", "CrewAI", "LangGraph", "OpenAI_Swarm"]
    single_models = ["GPT-4o", "Claude-Sonnet-4", "Gemini-2.5", "DeepSeek-V3"]
    alas_workflows_gpt4o = ["full", "no_repair", "no_validation", "no_optimization"]
    alas_workflows_claude4 = ["full", "no_repair", "no_validation", "no_optimization"]
    alas_workflows_deepseek_v3 = ["full", "no_repair", "no_validation", "no_optimization"]
    alas_workflows_gemini_2_5 = ["full", "no_repair", "no_validation", "no_optimization"]
    
    print("\n📋 OPTIMAL RATE REPORT BY DATASET CATEGORY")
    print("="*80)
    
    # Multi-Agent Systems (GPT-4o)
    print("\n🔄 MULTI-AGENT SYSTEMS (GPT-4o)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_gpt4o:
        row = [framework]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["MAS-GPT4o"][framework][category]['total_deviation']
            count = category_stats["MAS-GPT4o"][framework][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Multi-Agent Systems (Claude-4)
    print("\n🔄 MULTI-AGENT SYSTEMS (Claude-4)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_claude4:
        row = [framework]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["MAS-Claude4"][framework][category]['total_deviation']
            count = category_stats["MAS-Claude4"][framework][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Single-Agent Models
    print("\n🤖 SINGLE-AGENT MODELS")
    print("-" * 60)
    print(f"{'Model':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for model in single_models:
        row = [model]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["Single"][model][category]['total_deviation']
            count = category_stats["Single"][model][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - GPT-4o
    print("\n🔧 ALAS (OURS) - GPT-4o")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gpt4o:
        row = [workflow]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["ALAS-GPT4o"][workflow][category]['total_deviation']
            count = category_stats["ALAS-GPT4o"][workflow][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Claude-4
    print("\n🔧 ALAS (OURS) - Claude-4")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_claude4:
        row = [workflow]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["ALAS-Claude4"][workflow][category]['total_deviation']
            count = category_stats["ALAS-Claude4"][workflow][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - DeepSeek-V3
    print("\n🔧 ALAS (OURS) - DeepSeek-V3")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_deepseek_v3:
        row = [workflow]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["ALAS-DeepSeek-V3"][workflow][category]['total_deviation']
            count = category_stats["ALAS-DeepSeek-V3"][workflow][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Gemini-2.5
    print("\n🔧 ALAS (OURS) - Gemini-2.5")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gemini_2_5:
        row = [workflow]
        total_deviation = 0
        total_count = 0
        
        for category in categories:
            deviation = category_stats["ALAS-Gemini-2.5"][workflow][category]['total_deviation']
            count = category_stats["ALAS-Gemini-2.5"][workflow][category]['count']
            avg_deviation = (deviation / count) if count > 0 else 0
            row.append(f"{avg_deviation:.1f}%")
            total_deviation += deviation
            total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        row.append(f"{overall_deviation:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Summary by category
    print("\n📊 SUMMARY BY DATASET CATEGORY")
    print("-" * 60)
    print(f"{'Category':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for category in categories:
        total_deviation = 0
        total_count = 0
        
        for source in sources:
            if source == "MAS-GPT4o":
                for framework in mas_frameworks_gpt4o:
                    deviation = category_stats[source][framework][category]['total_deviation']
                    count = category_stats[source][framework][category]['count']
                    total_deviation += deviation
                    total_count += count
            elif source == "MAS-Claude4":
                for framework in mas_frameworks_claude4:
                    deviation = category_stats[source][framework][category]['total_deviation']
                    count = category_stats[source][framework][category]['count']
                    total_deviation += deviation
                    total_count += count
            elif source == "Single":
                for model in single_models:
                    deviation = category_stats[source][model][category]['total_deviation']
                    count = category_stats[source][model][category]['count']
                    total_deviation += deviation
                    total_count += count
            elif source == "ALAS-GPT4o":
                for workflow in alas_workflows_gpt4o:
                    deviation = category_stats[source][workflow][category]['total_deviation']
                    count = category_stats[source][workflow][category]['count']
                    total_deviation += deviation
                    total_count += count
            elif source == "ALAS-Claude4":
                for workflow in alas_workflows_claude4:
                    deviation = category_stats[source][workflow][category]['total_deviation']
                    count = category_stats[source][workflow][category]['count']
                    total_deviation += deviation
                    total_count += count
            elif source == "ALAS-DeepSeek-V3":
                for workflow in alas_workflows_deepseek_v3:
                    deviation = category_stats[source][workflow][category]['total_deviation']
                    count = category_stats[source][workflow][category]['count']
                    total_deviation += deviation
                    total_count += count
            elif source == "ALAS-Gemini-2.5":
                for workflow in alas_workflows_gemini_2_5:
                    deviation = category_stats[source][workflow][category]['total_deviation']
                    count = category_stats[source][workflow][category]['count']
                    total_deviation += deviation
                    total_count += count
        
        overall_deviation = (total_deviation / total_count) if total_count > 0 else 0
        print(f"{category:<15} {overall_deviation:.1f}%")
    
    print("\n✅ Optimal rate report generated successfully!")

if __name__ == "__main__":
    generate_optimal_rate_report()
