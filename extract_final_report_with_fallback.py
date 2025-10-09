#!/usr/bin/env python3
"""
Enhanced extraction script with fallback logic:
1. First try to extract from JSON files (final makespan)
2. If no value found, extract from TXT terminal output files
3. Handle template responses appropriately
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional

def extract_makespan_from_response(response_text: str) -> Optional[int]:
    """Step 1: Extract reported makespan from response text"""
    if not response_text:
        return None
    
    # Look for makespan patterns in the actual response text
    makespan_patterns = [
        r'FINAL MAKESPAN:\s*(\d+)',
        r'Final MAKESPAN:\s*(\d+)',
        r'FINAL Makespan:\s*(\d+)',
        r'Final Makespan:\s*(\d+)',
        r'FINAL makespan:\s*(\d+)',
        r'Final makespan:\s*(\d+)',
        r'makespan:\s*(\d+)',
        r'Makespan:\s*(\d+)',
        r'MAKESPAN:\s*(\d+)',
        r'Total time:\s*(\d+)',
        r'Total Time:\s*(\d+)',
        r'TOTAL TIME:\s*(\d+)',
        r'Completion time:\s*(\d+)',
        r'Completion Time:\s*(\d+)',
        r'COMPLETION TIME:\s*(\d+)',
        r'Total makespan:\s*(\d+)',
        r'Total Makespan:\s*(\d+)',
        r'TOTAL MAKESPAN:\s*(\d+)',
        r'Completion:\s*(\d+)',
        r'COMPLETION:\s*(\d+)'
    ]
    
    found_values = []
    for pattern in makespan_patterns:
        matches = re.findall(pattern, response_text, re.IGNORECASE)
        found_values.extend([int(match) for match in matches])
    
    # Filter out obvious template values (too small to be real makespan)
    real_values = [v for v in found_values if v > 100]  # Only values > 100 are likely real
    
    if real_values:
        return max(real_values)  # Return the largest (most likely to be makespan)
    
    return None



def extract_from_txt_file(txt_file_path: str) -> Tuple[Optional[int], str]:
    """Extract makespan from TXT terminal output file"""
    if not os.path.exists(txt_file_path):
        return None, "file_not_found"
    
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Special handling for MAPLE files
        if "MAPLE" in txt_file_path:
            # Look for "Best static makespan" pattern - take the FIRST occurrence (not the last)
            maple_matches = re.findall(r'Best static makespan for \w+: (\d+)', content)
            if maple_matches:
                # Take the first makespan value (likely the correct one)
                return int(maple_matches[0]), "maple_best_makespan"
        
        # For other frameworks, look for FINAL MAKESPAN in actual responses first
        final_makespan = extract_makespan_from_response(content)
        if final_makespan and final_makespan > 100:  # Only reasonable values
            return final_makespan, "terminal_final_makespan"
        
        # Look for structured schedule and extract max end time (but avoid timestamps)
        # Only look for "End Time:" in structured schedule context, not in timestamps
        schedule_lines = re.findall(r'End Time:\s*(\d+)', content)
        if schedule_lines:
            # Filter out timestamp-like values (years like 2025, 2024, etc.)
            valid_end_times = [int(line) for line in schedule_lines if int(line) < 2000 and int(line) > 100]
            if valid_end_times:
                max_end_time = max(valid_end_times)
                return max_end_time, "terminal_max_end_time"
        
        # Skip the summary makespan if it's clearly a template (like 25)
        makespan_match = re.search(r'Makespan:\s*(\d+)', content)
        if makespan_match:
            makespan_value = int(makespan_match.group(1))
            if makespan_value > 100:  # Only reasonable values
                return makespan_value, "terminal_makespan"
        
        return None, "no_makespan_found"
        
    except Exception as e:
        return None, f"error: {str(e)}"

def extract_from_json_file(json_file_path: str) -> Tuple[Optional[int], str]:
    """Extract makespan from JSON file"""
    if not os.path.exists(json_file_path):
        return None, "file_not_found"
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract response text
        response_text = data.get('response_text', '')
        structured_schedule = data.get('structured_schedule', '')
        
        # Try to get makespan from response
        makespan, source = get_final_makespan(response_text, structured_schedule)
        if makespan:
            return makespan, f"json_{source}"
        
        return None, "json_no_makespan"
        
    except Exception as e:
        return None, f"json_error: {str(e)}"

def extract_with_fallback(dataset: str, framework: str, results_dir: str = "results", results_maple_dir: str = "results_maple") -> Tuple[Optional[int], str]:
    """Extract makespan directly from TXT files only"""
    
    # Determine the correct directory and file
    if framework == "MAPLE":
        base_dir = results_maple_dir
        txt_file = os.path.join(base_dir, f"{dataset}_MAPLE.txt")
    else:
        base_dir = results_dir
        txt_file = os.path.join(base_dir, f"jssp_results_{dataset}_{framework}_terminal_output.txt")
    
    # Extract directly from TXT file
    makespan, source = extract_from_txt_file(txt_file)
    if makespan is not None:
        return makespan, f"txt_{source}"
    
    return None, "no_data_found"

def check_schedule_validity(response_text: str, structured_schedule: str = None) -> Tuple[bool, str]:
    """Check if the schedule is valid"""
    if not response_text and not structured_schedule:
        return False, "No data found"
    
    # Check for template responses
    template_indicators = [
        "[calculated_makespan]",
        "[integer value]",
        "Example format:",
        "Please solve this",
        "Let's say we have the following scenario"
    ]
    
    for indicator in template_indicators:
        if indicator in response_text:
            return False, "Template response"
    
    # Check if we have actual schedule data
    if structured_schedule and len(structured_schedule.strip()) > 50:
        return True, "Valid schedule found"
    
    if response_text and ("FINAL MAKESPAN:" in response_text or "STRUCTURED SCHEDULE:" in response_text):
        return True, "Valid response format"
    
    return False, "Invalid or incomplete response"

def main():
    """Main extraction function with fallback logic"""
    
    # Dataset mappings
    dmu_mapping = {
        'dmu03': 'rcmax_20_15_5',
        'dmu04': 'rcmax_20_15_8', 
        'dmu08': 'rcmax_20_20_7',
        'dmu09': 'rcmax_20_20_8',
        'dmu13': 'rcmax_30_15_5',
        'dmu14': 'rcmax_30_15_4',
        'dmu18': 'rcmax_30_20_9',
        'dmu19': 'rcmax_30_20_8',
        'dmu23': 'rcmax_40_15_10',
        'dmu24': 'rcmax_40_15_8',
        'dmu28': 'rcmax_40_20_6',
        'dmu29': 'rcmax_40_20_2',
        'dmu33': 'rcmax_50_15_2',
        'dmu34': 'rcmax_50_15_4',
        'dmu38': 'rcmax_50_20_6',
        'dmu39': 'rcmax_50_20_9'
    }
    
    # Test datasets
    dmu_datasets = ['dmu03', 'dmu04', 'dmu08', 'dmu09', 'dmu13', 'dmu14', 'dmu18', 'dmu19', 'dmu23', 'dmu24', 'dmu28', 'dmu29', 'dmu33', 'dmu34', 'dmu38', 'dmu39']
    ta_datasets = ['ta01', 'ta02', 'ta51', 'ta52', 'ta61', 'ta71', 'ta72']
    abz_swv_yn_datasets = ['abz07', 'abz08', 'abz09', 'swv01', 'swv02', 'swv03', 'swv04', 'swv05', 'swv06', 'swv07', 'swv08', 'swv09', 'swv10', 'swv11', 'swv12', 'swv13', 'swv14', 'swv15', 'yn01', 'yn02', 'yn03', 'yn04']
    
    frameworks = ['MAPLE', 'AutoGen', 'CrewAI', 'LangGraph', 'OpenAI_Swarm']
    
    print("Enhanced Extraction with Fallback Logic")
    print("=" * 50)
    
    # Process DMU datasets
    print("\nDMU Datasets - Makespan Performance:")
    print("| Dataset | MAPLE | AutoGen | CrewAI | LangGraph | OpenAI Swarm |")
    print("|---------|-------|---------|--------|-----------|--------------|")
    
    for dataset in dmu_datasets:
        actual_dataset = dmu_mapping.get(dataset, dataset)
        row = f"| {dataset.upper()} |"
        
        for framework in frameworks:
            makespan, source = extract_with_fallback(actual_dataset, framework)
            if makespan is not None:
                row += f" {makespan} |"
            else:
                row += " N/A |"
        
        print(row)
    
    # Process TA datasets  
    print("\nTA Datasets - Makespan Performance:")
    print("| Dataset | MAPLE | AutoGen | CrewAI | LangGraph | OpenAI Swarm |")
    print("|---------|-------|---------|--------|-----------|--------------|")
    
    for dataset in ta_datasets:
        row = f"| {dataset.upper()} |"
        
        for framework in frameworks:
            makespan, source = extract_with_fallback(dataset, framework)
            if makespan is not None:
                row += f" {makespan} |"
            else:
                row += " N/A |"
        
        print(row)
    
    # Process ABZ/SWV/YN datasets
    print("\nABZ/SWV/YN Datasets - Makespan Performance:")
    print("| Dataset | MAPLE | AutoGen | CrewAI | LangGraph | OpenAI Swarm |")
    print("|---------|-------|---------|--------|-----------|--------------|")
    
    for dataset in abz_swv_yn_datasets:
        row = f"| {dataset.upper()} |"
        
        for framework in frameworks:
            makespan, source = extract_with_fallback(dataset, framework)
            if makespan is not None:
                row += f" {makespan} |"
            else:
                row += " N/A |"
        
        print(row)
    
    # Validity tables
    print("\nDMU Datasets - Validity Performance:")
    print("| Dataset | MAPLE | AutoGen | CrewAI | LangGraph | OpenAI Swarm |")
    print("|---------|-------|---------|--------|-----------|--------------|")
    
    for dataset in dmu_datasets:
        actual_dataset = dmu_mapping.get(dataset, dataset)
        row = f"| {dataset.upper()} |"
        
        for framework in frameworks:
            makespan, source = extract_with_fallback(actual_dataset, framework)
            if makespan is not None:
                row += " Yes |"
            else:
                if "template" in source.lower():
                    row += " No (Template response) |"
                else:
                    row += " No (No data found) |"
        
        print(row)
    
    print("\nTA Datasets - Validity Performance:")
    print("| Dataset | MAPLE | AutoGen | CrewAI | LangGraph | OpenAI Swarm |")
    print("|---------|-------|---------|--------|-----------|--------------|")
    
    for dataset in ta_datasets:
        row = f"| {dataset.upper()} |"
        
        for framework in frameworks:
            makespan, source = extract_with_fallback(dataset, framework)
            if makespan is not None:
                row += " Yes |"
            else:
                if "template" in source.lower():
                    row += " No (Template response) |"
                else:
                    row += " No (No data found) |"
        
        print(row)
    
    print("\nABZ/SWV/YN Datasets - Validity Performance:")
    print("| Dataset | MAPLE | AutoGen | CrewAI | LangGraph | OpenAI Swarm |")
    print("|---------|-------|---------|--------|-----------|--------------|")
    
    for dataset in abz_swv_yn_datasets:
        row = f"| {dataset.upper()} |"
        
        for framework in frameworks:
            makespan, source = extract_with_fallback(dataset, framework)
            if makespan is not None:
                row += " Yes |"
            else:
                if "template" in source.lower():
                    row += " No (Template response) |"
                else:
                    row += " No (No data found) |"
        
        print(row)

if __name__ == "__main__":
    main()
