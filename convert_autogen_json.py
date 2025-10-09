#!/usr/bin/env python3
"""
Convert AutoGen JSON files to correct format
Fixes malformed job names and template text in AutoGen results
"""

import json
import os
import re
from typing import Dict, List, Any

def is_malformed_job(job_value: str) -> bool:
    """Check if a job value contains template text or is malformed"""
    if not isinstance(job_value, str):
        return False
    
    malformed_patterns = [
        r'\[job_name\]',
        r'Example format',
        r'FINAL MAKESPAN',
        r'STRUCTURED SCHEDULE',
        r'Step: \[step_number\]',
        r'Machine: \[machine_name\]',
        r'Start Time: \[start_time\]',
        r'End Time: \[end_time\]',
        r'Duration: \[duration\]'
    ]
    
    for pattern in malformed_patterns:
        if re.search(pattern, job_value, re.IGNORECASE):
            return True
    
    return False

def extract_job_name_from_malformed(job_value: str) -> str:
    """Extract actual job name from malformed template text"""
    # Look for patterns like "Job: Job1" or "Job1" in the text
    job_match = re.search(r'Job:\s*(Job\d+)', job_value)
    if job_match:
        return job_match.group(1)
    
    # Look for standalone Job patterns
    job_match = re.search(r'\b(Job\d+)\b', job_value)
    if job_match:
        return job_match.group(1)
    
    # If no job found, return a default
    return "Job1"

def convert_autogen_schedule(schedule: List[Dict]) -> List[Dict]:
    """Convert malformed AutoGen schedule to correct format"""
    converted_schedule = []
    
    for entry in schedule:
        if not isinstance(entry, dict):
            continue
            
        # Check if job field is malformed
        job_value = entry.get('job', '')
        
        if is_malformed_job(job_value):
            # Extract actual job name from template text
            actual_job = extract_job_name_from_malformed(job_value)
            
            # Create corrected entry
            corrected_entry = {
                'job': actual_job,
                'step': entry.get('step', 1),
                'machine': entry.get('machine', 'Machine0'),
                'start': entry.get('start', 0),
                'end': entry.get('end', 0),
                'duration': entry.get('duration', 0)
            }
            converted_schedule.append(corrected_entry)
        else:
            # Entry is already correct
            converted_schedule.append(entry)
    
    return converted_schedule

def convert_autogen_json(input_file: str, output_file: str) -> bool:
    """Convert a single AutoGen JSON file to correct format"""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Check if this is an AutoGen file
        if 'frameworks' not in data or 'AutoGen' not in data['frameworks']:
            print(f"⚠️ Not an AutoGen file: {input_file}")
            return False
        
        # Get the AutoGen data
        autogen_data = data['frameworks']['AutoGen']
        
        # Convert the schedule
        if 'structured_schedule' in autogen_data:
            original_schedule = autogen_data['structured_schedule']
            converted_schedule = convert_autogen_schedule(original_schedule)
            
            # Update the data
            autogen_data['structured_schedule'] = converted_schedule
            
            print(f"✅ Converted {len(original_schedule)} entries to {len(converted_schedule)} valid entries")
        
        # Save the converted file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Saved converted file: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {input_file}: {str(e)}")
        return False

def convert_all_autogen_files(input_dir: str, output_dir: str):
    """Convert all AutoGen JSON files in a directory"""
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all AutoGen JSON files
    autogen_files = []
    for filename in os.listdir(input_dir):
        if filename.endswith('_AutoGen.json'):
            autogen_files.append(filename)
    
    print(f"🔍 Found {len(autogen_files)} AutoGen files to convert")
    
    converted_count = 0
    for filename in autogen_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n📄 Converting {filename}...")
        if convert_autogen_json(input_path, output_path):
            converted_count += 1
    
    print(f"\n✅ Successfully converted {converted_count}/{len(autogen_files)} AutoGen files")

def main():
    """Main function to convert AutoGen files"""
    print("🔄 AutoGen JSON Converter")
    print("="*50)
    
    # Convert GPT-4o AutoGen files
    print("\n📁 Converting GPT-4o AutoGen files...")
    convert_all_autogen_files(
        "results_mas(gpt-4o)", 
        "results_mas(gpt-4o)_converted"
    )
    
    # Convert Claude-4 AutoGen files
    print("\n📁 Converting Claude-4 AutoGen files...")
    convert_all_autogen_files(
        "results_mas(claude-4)", 
        "results_mas(claude-4)_converted"
    )
    
    print("\n✅ AutoGen conversion completed!")

if __name__ == "__main__":
    main()
