#!/usr/bin/env python3
"""
Generate Makespan Report by Dataset Category
Shows makespan values for each dataset instance across different frameworks and models
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

def generate_makespan_report():
    """Generate makespan report by dataset category"""
    print("📊 Generating Makespan Report by Dataset Category")
    print("="*80)
    
    # Load validation results
    results = load_validation_results()
    if not results:
        return
    
    # Group results by dataset and source/framework
    dataset_results = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        source = result['source']
        framework = result['framework']
        dataset = result['dataset']
        makespan = result['makespan']
        category = categorize_dataset(dataset)
        
        if category != "Unknown" and makespan is not None:
            key = f"{source}-{framework}"
            dataset_results[dataset][key].append(makespan)
    
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
    
    print("\n📋 MAKESPAN REPORT BY DATASET CATEGORY")
    print("="*80)
    
    # Group datasets by category
    category_datasets = defaultdict(list)
    for dataset in dataset_results.keys():
        category = categorize_dataset(dataset)
        if category != "Unknown":
            category_datasets[category].append(dataset)
    
    # Print results for each category
    for category in categories:
        if category not in category_datasets:
            continue
            
        print(f"\n📊 {category} DATASETS")
        print("-" * 80)
        
        datasets = sorted(category_datasets[category])
        for dataset in datasets:
            print(f"\n🔹 {dataset.upper()}")
            print("-" * 40)
            
            # Collect all makespan values for this dataset
            all_makespans = []
            for key, makespans in dataset_results[dataset].items():
                if makespans:
                    avg_makespan = sum(makespans) / len(makespans)
                    all_makespans.append((key, avg_makespan, makespans))
            
            # Sort by makespan (ascending - lower is better)
            all_makespans.sort(key=lambda x: x[1])
            
            # Print results
            print(f"{'Framework/Model':<25} {'Avg Makespan':<15} {'Min':<8} {'Max':<8} {'Count':<6}")
            print("-" * 70)
            
            for key, avg_makespan, makespans in all_makespans:
                min_makespan = min(makespans)
                max_makespan = max(makespans)
                count = len(makespans)
                
                print(f"{key:<25} {avg_makespan:<15.1f} {min_makespan:<8.1f} {max_makespan:<8.1f} {count:<6}")
    
    # Summary statistics by category
    print("\n📊 SUMMARY BY CATEGORY")
    print("="*80)
    
    for category in categories:
        if category not in category_datasets:
            continue
            
        print(f"\n🔹 {category} CATEGORY SUMMARY")
        print("-" * 50)
        
        category_makespans = defaultdict(list)
        
        for dataset in category_datasets[category]:
            for key, makespans in dataset_results[dataset].items():
                if makespans:
                    category_makespans[key].extend(makespans)
        
        if category_makespans:
            print(f"{'Framework/Model':<25} {'Avg Makespan':<15} {'Min':<8} {'Max':<8} {'Total Count':<12}")
            print("-" * 80)
            
            # Sort by average makespan
            sorted_results = []
            for key, makespans in category_makespans.items():
                avg_makespan = sum(makespans) / len(makespans)
                min_makespan = min(makespans)
                max_makespan = max(makespans)
                sorted_results.append((key, avg_makespan, min_makespan, max_makespan, len(makespans)))
            
            sorted_results.sort(key=lambda x: x[1])
            
            for key, avg_makespan, min_makespan, max_makespan, count in sorted_results:
                print(f"{key:<25} {avg_makespan:<15.1f} {min_makespan:<8.1f} {max_makespan:<8.1f} {count:<12}")
    
    print("\n✅ Makespan report generated successfully!")

if __name__ == "__main__":
    generate_makespan_report()
