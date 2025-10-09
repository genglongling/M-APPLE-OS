#!/usr/bin/env python3
"""
Validate Claude-4 Generated Schedules
Recalculate success rate by running Claude-4 schedules through validation tools
"""

import os
import json
import sys
from datetime import datetime

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

def validate_claude4_schedule(dataset_name):
    """Validate a single Claude-4 generated schedule"""
    print(f"\n{'='*60}")
    print(f"Validating Claude-4 schedule for {dataset_name}")
    print(f"{'='*60}")
    
    try:
        # Load Claude-4 generated JSON file
        json_file_path = f"results_single(gpt-4o)/singleagent_llm_comparison_{dataset_name}_Claude-Sonnet-4.json"
        
        if not os.path.exists(json_file_path):
            print(f"❌ JSON file not found: {json_file_path}")
            return {
                'dataset': dataset_name,
                'success': False,
                'error': 'JSON file not found',
                'validation_errors': [],
                'makespan': None
            }
        
        # Load the JSON data
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract Claude-4 schedule
        if 'models' not in data or 'Claude-Sonnet-4' not in data['models']:
            print(f"❌ No Claude-Sonnet-4 data found in {json_file_path}")
            return {
                'dataset': dataset_name,
                'success': False,
                'error': 'No Claude-Sonnet-4 data found',
                'validation_errors': [],
                'makespan': None
            }
        
        claude_data = data['models']['Claude-Sonnet-4']
        
        if 'structured_schedule' not in claude_data:
            print(f"❌ No structured_schedule found in Claude-4 data")
            return {
                'dataset': dataset_name,
                'success': False,
                'error': 'No structured_schedule found',
                'validation_errors': [],
                'makespan': None
            }
        
        schedule = claude_data['structured_schedule']
        makespan = claude_data.get('makespan', None)
        
        print(f"📊 Loaded {len(schedule)} schedule entries")
        print(f"📊 Claude-4 reported makespan: {makespan}")
        
        # Calculate actual makespan
        actual_makespan = max(entry.get('end', 0) for entry in schedule) if schedule else 0
        print(f"📊 Calculated makespan: {actual_makespan}")
        
        # Run validation
        print("🔍 Running validation...")
        validation_result = ValidationTools.validate_schedule(schedule)
        
        if validation_result['valid']:
            print("✅ Validation PASSED - Schedule is valid")
            return {
                'dataset': dataset_name,
                'success': True,
                'validation_errors': [],
                'makespan': actual_makespan,
                'claude_makespan': makespan,
                'schedule_entries': len(schedule)
            }
        else:
            print(f"❌ Validation FAILED - {len(validation_result['errors'])} errors found")
            print("🔍 Validation errors:")
            for i, error in enumerate(validation_result['errors'][:10]):  # Show first 10 errors
                print(f"  {i+1}. {error}")
            if len(validation_result['errors']) > 10:
                print(f"  ... and {len(validation_result['errors']) - 10} more errors")
            
            return {
                'dataset': dataset_name,
                'success': False,
                'validation_errors': validation_result['errors'],
                'makespan': actual_makespan,
                'claude_makespan': makespan,
                'schedule_entries': len(schedule),
                'error_count': len(validation_result['errors'])
            }
            
    except Exception as e:
        print(f"❌ Error validating {dataset_name}: {str(e)}")
        return {
            'dataset': dataset_name,
            'success': False,
            'error': str(e),
            'validation_errors': [],
            'makespan': None
        }

def main():
    """Main function to validate all Claude-4 schedules"""
    print("🔍 Claude-4 Schedule Validation Analysis")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Total datasets: {len(test_datasets)}")
    
    results = []
    successful_validations = 0
    failed_validations = 0
    
    for i, dataset in enumerate(test_datasets):
        print(f"\n--- Processing dataset {i+1}/{len(test_datasets)}: {dataset} ---")
        
        result = validate_claude4_schedule(dataset)
        results.append(result)
        
        if result['success']:
            successful_validations += 1
            print(f"✅ {dataset}: VALID (makespan: {result['makespan']})")
        else:
            failed_validations += 1
            error_count = result.get('error_count', 0)
            print(f"❌ {dataset}: INVALID ({error_count} errors, makespan: {result['makespan']})")
    
    # Calculate success rate
    total_datasets = len(test_datasets)
    success_rate = (successful_validations / total_datasets) * 100 if total_datasets > 0 else 0
    
    # Print summary
    print(f"\n{'='*80}")
    print("CLAUDE-4 SCHEDULE VALIDATION SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Total datasets processed: {total_datasets}")
    print(f"✅ Successful validations: {successful_validations}")
    print(f"❌ Failed validations: {failed_validations}")
    print(f"📈 Success rate: {success_rate:.2f}%")
    
    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    print(f"{'Dataset':<20} {'Status':<8} {'Makespan':<10} {'Errors':<8} {'Entries':<8}")
    print(f"{'-'*60}")
    
    for result in results:
        status = "✅ VALID" if result['success'] else "❌ INVALID"
        makespan = result['makespan'] if result['makespan'] is not None else "N/A"
        error_count = result.get('error_count', 0)
        entries = result.get('schedule_entries', 0)
        
        print(f"{result['dataset']:<20} {status:<8} {makespan:<10} {error_count:<8} {entries:<8}")
    
    # Save results to JSON
    output_file = "claude4_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_datasets': total_datasets,
            'successful_validations': successful_validations,
            'failed_validations': failed_validations,
            'success_rate': success_rate,
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎯 Claude-4 initial schedule success rate: {success_rate:.2f}%")

if __name__ == "__main__":
    main()
