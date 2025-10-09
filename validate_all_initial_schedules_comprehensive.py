#!/usr/bin/env python3
"""
Comprehensive Validation of All Initial Schedules
Validates schedules from results_mas(gpt-4o), results_single(gpt-4o), and results_optimized(claude-4) directories
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

# Multi-agent frameworks (GPT-4o and Claude-4)
mas_frameworks_gpt4o = ["AutoGen", "CrewAI", "LangGraph", "OpenAI_Swarm"]
mas_frameworks_claude4 = ["AutoGen", "CrewAI", "LangGraph", "OpenAI_Swarm"]

# Single-agent models (corrected file names and JSON model names)
single_models = ["GPT-4o", "Claude-Sonnet-4", "Gemini-2.5", "DeepSeek-V3"]

# ALAS (ours) workflows (GPT-4o, Claude-4, DeepSeek-V3, and Gemini-2.5)
alas_workflows_gpt4o = ["full", "no_repair", "no_validation", "no_optimization"]
alas_workflows_claude4 = ["full", "no_repair", "no_validation", "no_optimization"]
alas_workflows_deepseek_v3 = ["full", "no_repair", "no_validation", "no_optimization"]
alas_workflows_gemini_2_5 = ["full", "no_repair", "no_validation", "no_optimization"]

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

def extract_schedule_from_alas_json(file_path: str, workflow: str) -> Tuple[List[Dict], int, str]:
    """Extract schedule from ALAS JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        schedule = []
        makespan = None
        
        if 'workflow_results' in data and workflow in data['workflow_results']:
            workflow_data = data['workflow_results'][workflow]
            if 'schedule' in workflow_data:
                schedule = workflow_data['schedule']
            if 'makespan' in workflow_data:
                makespan = workflow_data['makespan']
            # Note: We include both successful and failed schedules for analysis
        
        return schedule, makespan, f"ALAS-{workflow}"
        
    except Exception as e:
        print(f"❌ Error reading ALAS JSON file {file_path}: {str(e)}")
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
        'error': None if validation_result['valid'] else 'Validation failed',
        'validation_errors': validation_result['errors'],
        'makespan': calculated_makespan,
        'schedule_entries': len(schedule)
    }

def main():
    print("🔍 Comprehensive Initial Schedule Validation Analysis")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total datasets: {len(test_datasets)}")
    print(f"📊 MAS frameworks (GPT-4o): {len(mas_frameworks_gpt4o)}")
    print(f"📊 MAS frameworks (Claude-4): {len(mas_frameworks_claude4)}")
    print(f"📊 Single-agent models: {len(single_models)}")
    print(f"📊 ALAS workflows (GPT-4o): {len(alas_workflows_gpt4o)}")
    print(f"📊 ALAS workflows (Claude-4): {len(alas_workflows_claude4)}")
    print(f"📊 ALAS workflows (DeepSeek-V3): {len(alas_workflows_deepseek_v3)}")
    print(f"📊 ALAS workflows (Gemini-2.5): {len(alas_workflows_gemini_2_5)}")
    
    all_results = []
    
    # --- Validate Multi-Agent System Schedules (GPT-4o) ---
    print("\n🔍 Validating Multi-Agent System Schedules (GPT-4o)")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for framework in mas_frameworks_gpt4o:
            print(f"  🔍 {framework}...")
            # Use converted AutoGen files if available, otherwise use original
            if framework == "AutoGen":
                json_file_path = f"results_mas(gpt-4o)_converted/jssp_results_{dataset}_{framework}.json"
                if not os.path.exists(json_file_path):
                    json_file_path = f"results_mas(gpt-4o)/jssp_results_{dataset}_{framework}.json"
            else:
                json_file_path = f"results_mas(gpt-4o)/jssp_results_{dataset}_{framework}.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'MAS-GPT4o',
                    'framework': framework,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_framework = extract_schedule_from_mas_json(json_file_path)
                result = validate_schedule(schedule, dataset, 'MAS-GPT4o', framework)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'MAS-GPT4o',
                    'framework': framework,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Validate Multi-Agent System Schedules (Claude-4) ---
    print("\n🔍 Validating Multi-Agent System Schedules (Claude-4)")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for framework in mas_frameworks_claude4:
            print(f"  🔍 {framework}...")
            # Use converted AutoGen files if available, otherwise use original
            if framework == "AutoGen":
                json_file_path = f"results_mas(claude-4)_converted/jssp_results_{dataset}_{framework}.json"
                if not os.path.exists(json_file_path):
                    json_file_path = f"results_mas(claude-4)/jssp_results_{dataset}_{framework}.json"
            else:
                json_file_path = f"results_mas(claude-4)/jssp_results_{dataset}_{framework}.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'MAS-Claude4',
                    'framework': framework,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_framework = extract_schedule_from_mas_json(json_file_path)
                result = validate_schedule(schedule, dataset, 'MAS-Claude4', framework)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'MAS-Claude4',
                    'framework': framework,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Validate Single-Agent Schedules ---
    print("\n🔍 Validating Single-Agent Schedules")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for model in single_models:
            print(f"  🔍 {model}...")
            json_file_path = f"results_single(gpt-4o)/singleagent_llm_comparison_{dataset}_{model}.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'Single',
                    'framework': model,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_model = extract_schedule_from_single_json(json_file_path)
                result = validate_schedule(schedule, dataset, 'Single', model)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'Single',
                    'framework': model,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Validate ALAS (Ours) Schedules (GPT-4o) ---
    print("\n🔍 Validating ALAS (Ours) Schedules (GPT-4o)")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for workflow in alas_workflows_gpt4o:
            print(f"  🔍 ALAS-GPT4o-{workflow}...")
            json_file_path = f"results_optimized(gpt-4o)/{dataset}_workflow_comparison.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-GPT4o',
                    'framework': workflow,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_workflow = extract_schedule_from_alas_json(json_file_path, workflow)
                result = validate_schedule(schedule, dataset, 'ALAS-GPT4o', workflow)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-GPT4o',
                    'framework': workflow,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Validate ALAS (Ours) Schedules (Claude-4) ---
    print("\n🔍 Validating ALAS (Ours) Schedules (Claude-4)")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for workflow in alas_workflows_claude4:
            print(f"  🔍 ALAS-Claude4-{workflow}...")
            json_file_path = f"results_optimized(claude-4)/{dataset}_workflow_comparison.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-Claude4',
                    'framework': workflow,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_workflow = extract_schedule_from_alas_json(json_file_path, workflow)
                result = validate_schedule(schedule, dataset, 'ALAS-Claude4', workflow)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-Claude4',
                    'framework': workflow,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Validate ALAS (Ours) Schedules (DeepSeek-V3) ---
    print("\n🔍 Validating ALAS (Ours) Schedules (DeepSeek-V3)")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for workflow in alas_workflows_deepseek_v3:
            print(f"  🔍 ALAS-DeepSeek-V3-{workflow}...")
            json_file_path = f"results_optimized(deepseek-v3)/{dataset}_workflow_comparison.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-DeepSeek-V3',
                    'framework': workflow,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_workflow = extract_schedule_from_alas_json(json_file_path, workflow)
                result = validate_schedule(schedule, dataset, 'ALAS-DeepSeek-V3', workflow)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-DeepSeek-V3',
                    'framework': workflow,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Validate ALAS (Ours) Schedules (Gemini-2.5) ---
    print("\n🔍 Validating ALAS (Ours) Schedules (Gemini-2.5)")
    print("="*60)
    
    for dataset in test_datasets:
        print(f"\n📊 Processing {dataset}...")
        
        for workflow in alas_workflows_gemini_2_5:
            print(f"  🔍 ALAS-Gemini-2.5-{workflow}...")
            json_file_path = f"results_optimized(gemini-2.5)/{dataset}_workflow_comparison.json"
            
            if not os.path.exists(json_file_path):
                print(f"    ⚠️ JSON file not found: {json_file_path}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-Gemini-2.5',
                    'framework': workflow,
                    'success': False,
                    'error': 'File not found',
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
                continue
            
            try:
                schedule, makespan, extracted_workflow = extract_schedule_from_alas_json(json_file_path, workflow)
                result = validate_schedule(schedule, dataset, 'ALAS-Gemini-2.5', workflow)
                all_results.append(result)
                
                status = "✅ VALID" if result['success'] else "❌ INVALID"
                error_count = len(result['validation_errors'])
                print(f"    {status} ({error_count} errors, makespan: {result['makespan']})")
                
            except Exception as e:
                print(f"    ❌ Error processing {json_file_path}: {str(e)}")
                all_results.append({
                    'dataset': dataset,
                    'source': 'ALAS-Gemini-2.5',
                    'framework': workflow,
                    'success': False,
                    'error': str(e),
                    'validation_errors': [],
                    'makespan': None,
                    'schedule_entries': 0
                })
    
    # --- Summary Statistics ---
    print(f"\n{'='*80}")
    print("COMPREHENSIVE INITIAL SCHEDULE VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    # Overall statistics
    total_validations = len(all_results)
    successful_validations = sum(1 for r in all_results if r['success'])
    failed_validations = total_validations - successful_validations
    overall_success_rate = (successful_validations / total_validations) * 100 if total_validations > 0 else 0
    
    print(f"📊 Total validations: {total_validations}")
    print(f"✅ Successful validations: {successful_validations}")
    print(f"❌ Failed validations: {failed_validations}")
    print(f"📈 Overall success rate: {overall_success_rate:.2f}%")
    
    # Breakdown by source
    print("\n📊 Breakdown by source:")
    for source in ['MAS-GPT4o', 'MAS-Claude4', 'Single', 'ALAS-GPT4o', 'ALAS-Claude4', 'ALAS-DeepSeek-V3', 'ALAS-Gemini-2.5']:
        source_results = [r for r in all_results if r['source'] == source]
        if source_results:
            source_successful = sum(1 for r in source_results if r['success'])
            source_total = len(source_results)
            source_rate = (source_successful / source_total) * 100 if source_total > 0 else 0
            print(f"  {source}: {source_rate:.2f}% ({source_successful}/{source_total})")
    
    # Detailed breakdown by framework/model
    print("\n📋 DETAILED RESULTS BY FRAMEWORK/MODEL:")
    print(f"{'Source':<12} {'Framework/Model':<20} {'Valid':<5} {'Total':<5} {'Rate':<8}")
    print("-" * 75)
    
    # MAS frameworks (GPT-4o)
    for framework in mas_frameworks_gpt4o:
        valid_count = sum(1 for r in all_results if r['source'] == 'MAS-GPT4o' and r['framework'] == framework and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'MAS-GPT4o':<12} {framework:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # MAS frameworks (Claude-4)
    for framework in mas_frameworks_claude4:
        valid_count = sum(1 for r in all_results if r['source'] == 'MAS-Claude4' and r['framework'] == framework and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'MAS-Claude4':<12} {framework:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # Single-agent models
    for model in single_models:
        valid_count = sum(1 for r in all_results if r['source'] == 'Single' and r['framework'] == model and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'Single':<12} {model:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # ALAS workflows (GPT-4o)
    for workflow in alas_workflows_gpt4o:
        valid_count = sum(1 for r in all_results if r['source'] == 'ALAS-GPT4o' and r['framework'] == workflow and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'ALAS-GPT4o':<12} {workflow:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # ALAS workflows (Claude-4)
    for workflow in alas_workflows_claude4:
        valid_count = sum(1 for r in all_results if r['source'] == 'ALAS-Claude4' and r['framework'] == workflow and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'ALAS-Claude4':<12} {workflow:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # ALAS workflows (DeepSeek-V3)
    for workflow in alas_workflows_deepseek_v3:
        valid_count = sum(1 for r in all_results if r['source'] == 'ALAS-DeepSeek-V3' and r['framework'] == workflow and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'ALAS-DeepSeek-V3':<12} {workflow:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # ALAS workflows (Gemini-2.5)
    for workflow in alas_workflows_gemini_2_5:
        valid_count = sum(1 for r in all_results if r['source'] == 'ALAS-Gemini-2.5' and r['framework'] == workflow and r['success'])
        total_count = len(test_datasets)
        rate = (valid_count / total_count) * 100 if total_count > 0 else 0
        print(f"{'ALAS-Gemini-2.5':<12} {workflow:<20} {valid_count:<5} {total_count:<5} {rate:<8.2f}%")
    
    # Save results
    output_file = "comprehensive_initial_schedule_validation_with_alas.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾 Comprehensive results saved to: {output_file}")
    print(f"🎯 Overall initial schedule success rate: {overall_success_rate:.2f}%")

if __name__ == "__main__":
    main()
