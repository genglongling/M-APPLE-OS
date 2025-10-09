#!/usr/bin/env python3

# DMU datasets (16 datasets)
dmu_datasets = [
    "rcmax_20_15_5", "rcmax_20_15_8", "rcmax_20_20_7", "rcmax_20_20_8",
    "rcmax_30_15_5", "rcmax_30_15_4", "rcmax_30_20_9", "rcmax_30_20_8",
    "rcmax_40_15_10", "rcmax_40_15_8", "rcmax_40_20_6", "rcmax_40_20_2",
    "rcmax_50_15_2", "rcmax_50_15_4", "rcmax_50_20_6", "rcmax_50_20_9"
]

# TA datasets (7 datasets)
ta_datasets = [
    "TA01", "TA02", "TA51", "TA52", "TA61", "TA71", "TA72"
]

# ABZSWVYN datasets (18 datasets)
abzswvyn_datasets = [
    "abz07", "abz08", "abz09",
    "swv01", "swv02", "swv03", "swv04", "swv05", "swv06", "swv07", "swv08", "swv09", "swv10",
    "swv11", "swv12", "swv13", "swv14", "swv15",
    "yn01", "yn02", "yn03", "yn04"
]

# Frameworks to test
frameworks = ["CrewAI", "AutoGen", "OpenAI_Swarm", "LangGraph"]

# API Key
api_key = "API_KEY_TO_REPLACE"

# Combine all datasets
all_datasets = dmu_datasets + ta_datasets + abzswvyn_datasets

print(f"# Total datasets: {len(all_datasets)}")
print(f"# Total frameworks: {len(frameworks)}")
print(f"# Total tests: {len(all_datasets) * len(frameworks)}")
print()

# Generate all commands
command_count = 0
for dataset in all_datasets:
    for framework in frameworks:
        command_count += 1
        print(f"# Test {command_count}: {framework} on {dataset}")
        print(f"python3.10 applications/jssp_framework_comparison.py --dataset {dataset} --api-key {api_key} --frameworks {framework}")
        print()

print(f"# Generated {command_count} test commands")
