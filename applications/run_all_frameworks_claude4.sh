#!/bin/bash

# Set Anthropic API Key for Claude-4
export ANTHROPIC_API_KEY=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

# Create results directory
mkdir -p "results_mas(claude-4)"

# DMU datasets (16 datasets)
dmu_datasets=(
    "rcmax_20_15_5" "rcmax_20_15_8" "rcmax_20_20_7" "rcmax_20_20_8"
    "rcmax_30_15_5" "rcmax_30_15_4" "rcmax_30_20_9" "rcmax_30_20_8"
    "rcmax_40_15_10" "rcmax_40_15_8" "rcmax_40_20_6" "rcmax_40_20_2"
    "rcmax_50_15_2" "rcmax_50_15_4" "rcmax_50_20_6" "rcmax_50_20_9"
)

# TA datasets (7 datasets)
ta_datasets=(
    "TA01" "TA02" "TA51" "TA52" "TA61" "TA71" "TA72"
)

# ABZSWVYN datasets (18 datasets)
abzswvyn_datasets=(
    "abz07" "abz08" "abz09"
    "swv01" "swv02" "swv03" "swv04" "swv05" "swv06" "swv07" "swv08" "swv09" "swv10"
    "swv11" "swv12" "swv13" "swv14" "swv15"
    "yn01" "yn02" "yn03" "yn04"
)

# Frameworks to test (all using Claude-4)
frameworks=("CrewAI" "AutoGen" "OpenAI_Swarm" "LangGraph")

# Calculate total tests
total_datasets=$((${#dmu_datasets[@]} + ${#ta_datasets[@]} + ${#abzswvyn_datasets[@]}))
total_frameworks=${#frameworks[@]}
total_tests=$((total_datasets * total_frameworks))

echo "Starting comprehensive JSSP framework testing with Claude-4..."
echo "Total datasets: $total_datasets"
echo "Total frameworks: $total_frameworks"
echo "Total tests: $total_tests"
echo "Estimated time: ~$((total_tests * 2)) minutes (2 min per test)"
echo "=========================================="

# Initialize counters
test_count=0
success_count=0
fail_count=0

# Function to run a single test
run_test() {
    local dataset=$1
    local framework=$2
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local log_file="results_mas(claude-4)/${dataset}_${framework}_${timestamp}.log"
    
    # Update progress
    test_count=$((test_count + 1))
    echo "[$test_count/$total_tests] Testing $framework on $dataset with Claude-4..."
    echo "Logging to: $log_file"
    
    # Run the test and capture all output
    python3.10 applications/run_jssp_framework_comparison.py \
        --dataset "$dataset" \
        --api-key "$ANTHROPIC_API_KEY" \
        --frameworks "$framework" \
        --output "results_mas(claude-4)/jssp_results_${dataset}_${framework}.json" \
        2>&1 | tee "$log_file"
    
    # Check if the test was successful
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        echo "✅ Completed $framework on $dataset successfully"
        success_count=$((success_count + 1))
    else
        echo "❌ Failed $framework on $dataset (exit code: $exit_code)"
        fail_count=$((fail_count + 1))
    fi
    
    # Show progress
    echo "Progress: $test_count/$total_tests tests completed"
    echo "Success: $success_count, Failed: $fail_count"
    echo "----------------------------------------"
}

# Test DMU datasets
echo "Testing DMU datasets..."
for dataset in "${dmu_datasets[@]}"; do
    for framework in "${frameworks[@]}"; do
        run_test "$dataset" "$framework"
    done
done

# Test TA datasets
echo "Testing TA datasets..."
for dataset in "${ta_datasets[@]}"; do
    for framework in "${frameworks[@]}"; do
        run_test "$dataset" "$framework"
    done
done

# Test ABZSWVYN datasets
echo "Testing ABZSWVYN datasets..."
for dataset in "${abzswvyn_datasets[@]}"; do
    for framework in "${frameworks[@]}"; do
        run_test "$dataset" "$framework"
    done
done

echo "=========================================="
echo "🎉 ALL TESTS COMPLETED!"
echo "=========================================="
echo "📊 FINAL SUMMARY:"
echo "Total tests run: $test_count"
echo "✅ Successful: $success_count"
echo "❌ Failed: $fail_count"
if [ $test_count -gt 0 ]; then
    success_rate=$((success_count * 100 / test_count))
    echo "📈 Success rate: $success_rate%"
fi
echo ""
echo "📁 Results saved in ./results_mas(claude-4)/ directory"
echo "📄 Log files: {dataset}_{framework}_{timestamp}.log"
echo "📄 JSON results: jssp_results_{dataset}_{framework}.json"
echo "=========================================="

# Generate detailed summary report
echo "Generating detailed summary report..."
python3.10 -c "
import os
import json
import glob
from datetime import datetime

results_dir = 'results_mas(claude-4)'
if os.path.exists(results_dir):
    json_files = glob.glob(f'{results_dir}/jssp_results_*.json')
    log_files = glob.glob(f'{results_dir}/*.log')
    
    print(f'📊 DETAILED SUMMARY REPORT')
    print(f'============================')
    print(f'Generated at: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
    print(f'JSON result files: {len(json_files)}')
    print(f'Log files: {len(log_files)}')
    print(f'')
    
    # Analyze results by framework
    framework_stats = {}
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if 'frameworks' in data:
                    for framework, result in data['frameworks'].items():
                        if framework not in framework_stats:
                            framework_stats[framework] = {'success': 0, 'failed': 0}
                        
                        if result.get('success', False):
                            framework_stats[framework]['success'] += 1
                        else:
                            framework_stats[framework]['failed'] += 1
        except Exception as e:
            print(f'Error reading {json_file}: {e}')
    
    if framework_stats:
        print('📈 Results by Framework:')
        for framework, stats in framework_stats.items():
            total = stats['success'] + stats['failed']
            rate = (stats['success'] / total * 100) if total > 0 else 0
            print(f'  {framework}: {stats[\"success\"]}/{total} ({rate:.1f}%)')
    
    print(f'')
    print(f'📁 All results saved in: {results_dir}/')
else:
    print('❌ No results directory found')
"
