#!/usr/bin/env python3
"""
Generate Error Rate Report by Dataset Category
Shows error rates (error steps/full steps) for each of the 5 dataset categories in columns
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

def calculate_error_rate(result):
    """Calculate error rate as percentage of schedule entries that have errors"""
    if result['schedule_entries'] == 0:
        return 100.0  # 100% error if no schedule entries
    
    error_count = len(result['validation_errors'])
    total_entries = result['schedule_entries']
    
    if total_entries == 0:
        return 100.0
    
    # Calculate error rate as percentage of entries with errors
    # Each validation error represents a problem with a schedule entry
    # Cap at 100% to ensure logical consistency
    error_rate = min((error_count / total_entries) * 100, 100.0)
    return error_rate

def generate_error_rate_report():
    """Generate error rate report by dataset category"""
    print("📊 Generating Error Rate Report by Dataset Category")
    print("="*80)
    
    # Load validation results
    results = load_validation_results()
    if not results:
        return
    
    # Group results by source, framework, and dataset category
    category_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'error_steps': 0, 'total_steps': 0, 'count': 0})))
    
    for result in results:
        source = result['source']
        framework = result['framework']
        dataset = result['dataset']
        category = categorize_dataset(dataset)
        
        if category != "Unknown":
            error_count = len(result['validation_errors'])
            total_steps = result['schedule_entries']
            
            category_stats[source][framework][category]['error_steps'] += error_count
            category_stats[source][framework][category]['total_steps'] += total_steps
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
    
    print("\n📋 ERROR RATE REPORT BY DATASET CATEGORY")
    print("="*80)
    
    # Multi-Agent Systems (GPT-4o)
    print("\n🔄 MULTI-AGENT SYSTEMS (GPT-4o)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_gpt4o:
        row = [framework]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["MAS-GPT4o"][framework][category]['error_steps']
            steps = category_stats["MAS-GPT4o"][framework][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Multi-Agent Systems (Claude-4)
    print("\n🔄 MULTI-AGENT SYSTEMS (Claude-4)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_claude4:
        row = [framework]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["MAS-Claude4"][framework][category]['error_steps']
            steps = category_stats["MAS-Claude4"][framework][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Single-Agent Models
    print("\n🤖 SINGLE-AGENT MODELS")
    print("-" * 60)
    print(f"{'Model':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for model in single_models:
        row = [model]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["Single"][model][category]['error_steps']
            steps = category_stats["Single"][model][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - GPT-4o
    print("\n🔧 ALAS (OURS) - GPT-4o")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gpt4o:
        row = [workflow]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["ALAS-GPT4o"][workflow][category]['error_steps']
            steps = category_stats["ALAS-GPT4o"][workflow][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Claude-4
    print("\n🔧 ALAS (OURS) - Claude-4")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_claude4:
        row = [workflow]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["ALAS-Claude4"][workflow][category]['error_steps']
            steps = category_stats["ALAS-Claude4"][workflow][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - DeepSeek-V3
    print("\n🔧 ALAS (OURS) - DeepSeek-V3")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_deepseek_v3:
        row = [workflow]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["ALAS-DeepSeek-V3"][workflow][category]['error_steps']
            steps = category_stats["ALAS-DeepSeek-V3"][workflow][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Gemini-2.5
    print("\n🔧 ALAS (OURS) - Gemini-2.5")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gemini_2_5:
        row = [workflow]
        total_error_steps = 0
        total_steps = 0
        
        for category in categories:
            error_steps = category_stats["ALAS-Gemini-2.5"][workflow][category]['error_steps']
            steps = category_stats["ALAS-Gemini-2.5"][workflow][category]['total_steps']
            rate = min((error_steps / steps * 100) if steps > 0 else 0, 100.0)
            row.append(f"{rate:.1f}%")
            total_error_steps += error_steps
            total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Summary by category
    print("\n📊 SUMMARY BY DATASET CATEGORY")
    print("-" * 60)
    print(f"{'Category':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for category in categories:
        total_error_steps = 0
        total_steps = 0
        
        for source in sources:
            if source == "MAS-GPT4o":
                for framework in mas_frameworks_gpt4o:
                    error_steps = category_stats[source][framework][category]['error_steps']
                    steps = category_stats[source][framework][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
            elif source == "MAS-Claude4":
                for framework in mas_frameworks_claude4:
                    error_steps = category_stats[source][framework][category]['error_steps']
                    steps = category_stats[source][framework][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
            elif source == "Single":
                for model in single_models:
                    error_steps = category_stats[source][model][category]['error_steps']
                    steps = category_stats[source][model][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
            elif source == "ALAS-GPT4o":
                for workflow in alas_workflows_gpt4o:
                    error_steps = category_stats[source][workflow][category]['error_steps']
                    steps = category_stats[source][workflow][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
            elif source == "ALAS-Claude4":
                for workflow in alas_workflows_claude4:
                    error_steps = category_stats[source][workflow][category]['error_steps']
                    steps = category_stats[source][workflow][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
            elif source == "ALAS-DeepSeek-V3":
                for workflow in alas_workflows_deepseek_v3:
                    error_steps = category_stats[source][workflow][category]['error_steps']
                    steps = category_stats[source][workflow][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
            elif source == "ALAS-Gemini-2.5":
                for workflow in alas_workflows_gemini_2_5:
                    error_steps = category_stats[source][workflow][category]['error_steps']
                    steps = category_stats[source][workflow][category]['total_steps']
                    total_error_steps += error_steps
                    total_steps += steps
        
        overall_rate = min((total_error_steps / total_steps * 100) if total_steps > 0 else 0, 100.0)
        print(f"{category:<15} {overall_rate:.1f}%")
    
    print("\n✅ Error rate report generated successfully!")

if __name__ == "__main__":
    generate_error_rate_report()
