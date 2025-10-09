#!/bin/bash

# Set OpenAI API Key
export OPENAI_API_KEY=API_KEY_TO_REPLACE

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

# Frameworks to test
frameworks=("CrewAI" "AutoGen" "OpenAI_Swarm" "LangGraph")

echo "Starting comprehensive JSSP framework testing..."
echo "Total datasets: $((${#dmu_datasets[@]} + ${#ta_datasets[@]} + ${#abzswvyn_datasets[@]}))"
echo "Total frameworks: ${#frameworks[@]}"
echo "Total tests: $((${#dmu_datasets[@]} + ${#ta_datasets[@]} + ${#abzswvyn_datasets[@]}) * ${#frameworks[@]})"
echo "=========================================="

# Function to run a single test
run_test() {
    local dataset=$1
    local framework=$2
    echo "Testing $framework on $dataset..."
    python3.10 applications/run_jssp_framework_comparison.py --dataset "$dataset" --api-key "$OPENAI_API_KEY" --frameworks "$framework"
    echo "Completed $framework on $dataset"
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
echo "All tests completed!"
echo "Results saved in ./results/ directory with format: jssp_results_{dataset}_{framework}.*"
