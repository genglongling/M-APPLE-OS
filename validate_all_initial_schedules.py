#!/usr/bin/env python3
"""
Comprehensive Validation of All Initial Schedules
Validates schedules from results_mas(gpt-4o) and results_single(gpt-4o) directories
"""

import os
import json
import sys
import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add src to path for imports
sys.path.append('src')

# Import validation tools
from utils.validation_tools import ValidationTools

# Full dataset suite
test_datasets = [
    # DMU datasets (16 datasets)
    "rcmax_20_15_5", "rcmax_20_15_8", "rcmax_20_20_7", "rcmax_20_20_8",
    "rcmax_30_15_5", "rcmax_30_15_4", "rcmax_30_20_9", "rcmax_30_20_8",
    "rcmax_40_15_10", "rcmax_40_15_8", "rcmax_40_20_6", "rcmax_40_20_2",
    "rcmax_50_15_2", "rcmax_50_15_4", "rcmax_50_20_6", "rcmax_50_20_9",
    # TA datasets (7 datasets)
    "TA01", "TA02", "TA51", "TA52", "TA61", "TA71", "TA72",
    # ABZSWVYN datasets (18 datasets)
    "abz07", "abz08", "abz09",
    "swv01", "swv02", "swv03", "swv04", "swv05", "swv06", "swv07", "swv08", "swv09", "swv10",
    "swv11", "swv12", "swv13", "swv14", "swv15",
    "yn01", "yn02", "yn03", "yn04"
]

# Multi-agent frameworks
mas_frameworks = ["AutoGen", "CrewAI", "LangGraph", "OpenAI_Swarm"]

# Single-agent models (corrected file names)
single_models = ["gpt-4o", "Claude-Sonnet-4", "gemini-2.5", "deepseek-v3"]

def extract_schedule_from_mas_json(file_path: str) -> Tuple[List[Dict], int, str]:
    """Extract schedule from MAS JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Look for schedule in different possible locations
        schedule = []
        makespan = None
        framework = "Unknown"
        
        # Extract framework name from file path
        if 'AutoGen' in file_path:
            framework = "AutoGen"
        elif 'CrewAI' in file_path:
            framework = "CrewAI"
        elif 'LangGraph' in file_path:
            framework = "LangGraph"
        elif 'OpenAI_Swarm' in file_path:
            framework = "OpenAI_Swarm"
        
        # Try different JSON structures for MAS files
        if 'frameworks' in data and framework in data['frameworks']:
            # New MAS format: data.frameworks[framework].structured_schedule
            framework_data = data['frameworks'][framework]
            if 'structured_schedule' in framework_data:
                schedule = framework_data['structured_schedule']
            if 'makespan' in framework_data:
                makespan = framework_data['makespan']
        elif 'schedule' in data:
            schedule = data['schedule']
        elif 'schedules' in data:
            schedule = data['schedules']
        elif 'results' in data and 'schedule' in data['results']:
            schedule = data['results']['schedule']
        elif 'output' in data and 'schedule' in data['output']:
            schedule = data['output']['schedule']
        
        # Extract makespan if not found in framework data
        if makespan is None:
            if 'makespan' in data:
                makespan = data['makespan']
            elif 'results' in data and 'makespan' in data['results']:
                makespan = data['results']['makespan']
            elif 'output' in data and 'makespan' in data['output']:
                makespan = data['output']['makespan']
        
        return schedule, makespan, framework
        
    except Exception as e:
        print(f"❌ Error reading MAS JSON file {file_path}: {str(e)}")
        return [], None, "Unknown"

def extract_schedule_from_single_json(file_path: str) -> Tuple[List[Dict], int, str]:
    """Extract schedule from single-agent JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Look for schedule in models structure
        schedule = []
        makespan = None
        model = "Unknown"
        
        if 'models' in data:
            for model_name in single_models:
                if model_name in data['models']:
                    model_data = data['models'][model_name]
                    if 'structured_schedule' in model_data:
                        schedule = model_data['structured_schedule']
                        makespan = model_data.get('makespan', None)
                        model = model_name
                        break
        
        return schedule, makespan, model
        
    except Exception as e:
        print(f"❌ Error reading single-agent JSON file {file_path}: {str(e)}")
        return [], None, "Unknown"

def validate_schedule(schedule: List[Dict], dataset: str, source: str, framework: str) -> Dict:
    """Validate a single schedule"""
    if not schedule:
        return {
            'dataset': dataset,
            'source': source,
            'framework': framework,
            'success': False,
            'error': 'No schedule found',
            'validation_errors': [],
            'makespan': None,
            'schedule_entries': 0
        }
    
    # Calculate makespan
    calculated_makespan = max(entry.get('end', 0) for entry in schedule) if schedule else 0
    
    # Run validation
    validation_result = ValidationTools.validate_schedule(schedule)
    
    return {
        'dataset': dataset,
        'source': source,
        'framework': framework,
        'success': validation_result['valid'],
        'validation_errors': validation_result['errors'],
        'makespan': calculated_makespan,
        'schedule_entries': len(schedule),
        'error_count': len(validation_result['errors'])
    }

def validate_mas_schedules():
    """Validate all MAS schedules"""
    print("🔍 Validating Multi-Agent System Schedules")
    print("="*60)
    
    results = []
    total_files = 0
    successful_validations = 0
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for framework in mas_frameworks:
            json_file = f"results_mas(gpt-4o)/jssp_results_{dataset}_{framework}.json"
            
            if os.path.exists(json_file):
                total_files += 1
                print(f"  🔍 {framework}...")
                
                schedule, makespan, extracted_framework = extract_schedule_from_mas_json(json_file)
                result = validate_schedule(schedule, dataset, "MAS", framework)
                results.append(result)
                
                if result['success']:
                    successful_validations += 1
                    print(f"    ✅ VALID (makespan: {result['makespan']})")
                else:
                    error_count = result.get('error_count', 0)
                    print(f"    ❌ INVALID ({error_count} errors, makespan: {result['makespan']})")
            else:
                print(f"  ⚠️ {framework}: File not found")
    
    success_rate = (successful_validations / total_files) * 100 if total_files > 0 else 0
    
    print(f"\n📈 MAS Validation Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successful validations: {successful_validations}")
    print(f"  Success rate: {success_rate:.2f}%")
    
    return results, success_rate

def validate_single_schedules():
    """Validate all single-agent schedules"""
    print("\n🔍 Validating Single-Agent Schedules")
    print("="*60)
    
    results = []
    total_files = 0
    successful_validations = 0
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for model in single_models:
            json_file = f"results_single(gpt-4o)/singleagent_llm_comparison_{dataset}_{model}.json"
            
            if os.path.exists(json_file):
                total_files += 1
                print(f"  🔍 {model}...")
                
                schedule, makespan, extracted_model = extract_schedule_from_single_json(json_file)
                result = validate_schedule(schedule, dataset, "Single", model)
                results.append(result)
                
                if result['success']:
                    successful_validations += 1
                    print(f"    ✅ VALID (makespan: {result['makespan']})")
                else:
                    error_count = result.get('error_count', 0)
                    print(f"    ❌ INVALID ({error_count} errors, makespan: {result['makespan']})")
            else:
                print(f"  ⚠️ {model}: File not found")
    
    success_rate = (successful_validations / total_files) * 100 if total_files > 0 else 0
    
    print(f"\n📈 Single-Agent Validation Summary:")
    print(f"  Total files processed: {total_files}")
    print(f"  Successful validations: {successful_validations}")
    print(f"  Success rate: {success_rate:.2f}%")
    
    return results, success_rate

def main():
    """Main function to validate all initial schedules"""
    print("🔍 Comprehensive Initial Schedule Validation Analysis")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total datasets: {len(test_datasets)}")
    print(f"📊 MAS frameworks: {len(mas_frameworks)}")
    print(f"📊 Single-agent models: {len(single_models)}")
    
    # Validate MAS schedules
    mas_results, mas_success_rate = validate_mas_schedules()
    
    # Validate single-agent schedules
    single_results, single_success_rate = validate_single_schedules()
    
    # Combine all results
    all_results = mas_results + single_results
    
    # Calculate overall statistics
    total_validations = len(all_results)
    total_successful = sum(1 for r in all_results if r['success'])
    overall_success_rate = (total_successful / total_validations) * 100 if total_validations > 0 else 0
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE INITIAL SCHEDULE VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Total validations: {total_validations}")
    print(f"✅ Successful validations: {total_successful}")
    print(f"❌ Failed validations: {total_validations - total_successful}")
    print(f"📈 Overall success rate: {overall_success_rate:.2f}%")
    print(f"\n📊 Breakdown by source:")
    print(f"  🔄 Multi-Agent Systems: {mas_success_rate:.2f}%")
    print(f"  🤖 Single-Agent Models: {single_success_rate:.2f}%")
    
    # Show detailed results by framework/model
    print(f"\n📋 DETAILED RESULTS BY FRAMEWORK/MODEL:")
    print(f"{'Source':<8} {'Framework/Model':<20} {'Valid':<6} {'Total':<6} {'Rate':<8}")
    print(f"{'-'*50}")
    
    # Group results by framework/model
    framework_stats = {}
    for result in all_results:
        key = f"{result['source']}_{result['framework']}"
        if key not in framework_stats:
            framework_stats[key] = {'valid': 0, 'total': 0}
        framework_stats[key]['total'] += 1
        if result['success']:
            framework_stats[key]['valid'] += 1
    
    for key, stats in framework_stats.items():
        source, framework = key.split('_', 1)
        rate = (stats['valid'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{source:<8} {framework:<20} {stats['valid']:<6} {stats['total']:<6} {rate:<8.2f}%")
    
    # Save comprehensive results
    output_file = "comprehensive_initial_schedule_validation.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_validations': total_validations,
            'successful_validations': total_successful,
            'overall_success_rate': overall_success_rate,
            'mas_success_rate': mas_success_rate,
            'single_success_rate': single_success_rate,
            'framework_stats': framework_stats,
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    print(f"🎯 Overall initial schedule success rate: {overall_success_rate:.2f}%")

if __name__ == "__main__":
    main()
