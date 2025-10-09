#!/bin/bash

# Set API Keys for all LLM providers
export OPENAI_API_KEY=API_KEY_TO_REPLACE
export ANTHROPIC_API_KEY=API_KEY_TO_REPLACE
export GOOGLE_API_KEY=API_KEY_TO_REPLACE
export DEEPSEEK_API_KEY=API_KEY_TO_REPLACE

# DMU datasets (16 datasets)
# "rcmax_20_15_5"
dmu_datasets=(
     "rcmax_20_15_8" "rcmax_20_20_7" "rcmax_20_20_8"
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

# All LLM models to test together
all_models=("Claude-Sonnet-4") # "GPT-4o" "DeepSeek-V3" "Claude-Sonnet-4" "Gemini-2.5"

echo "Starting comprehensive Single-Agent LLM JSSP testing (Efficient Mode)..."
echo "Total datasets: $((${#dmu_datasets[@]} + ${#ta_datasets[@]} + ${#abzswvyn_datasets[@]}))"
echo "Total models: ${#all_models[@]}"
echo "Total tests: $((${#dmu_datasets[@]} + ${#ta_datasets[@]} + ${#abzswvyn_datasets[@]}))"
echo "=========================================="

# Function to run all models for a single dataset
run_dataset_test() {
    local dataset=$1
    echo "Testing all LLM models on $dataset..."
    python3.10 applications/run_jssp_framework_comparison_singleagent.py --dataset "$dataset" --models "${all_models[@]}"
    echo "Completed all models on $dataset"
    echo "----------------------------------------"
}

# Test DMU datasets
echo "Testing DMU datasets..."
for dataset in "${dmu_datasets[@]}"; do
    run_dataset_test "$dataset"
done

# Test TA datasets
echo "Testing TA datasets..."
for dataset in "${ta_datasets[@]}"; do
    run_dataset_test "$dataset"
done

# Test ABZSWVYN datasets
echo "Testing ABZSWVYN datasets..."
for dataset in "${abzswvyn_datasets[@]}"; do
    run_dataset_test "$dataset"
done

echo "=========================================="
echo "All single-agent LLM tests completed!"
echo "Results saved in ./results/ directory with format: singleagent_llm_comparison.json"
