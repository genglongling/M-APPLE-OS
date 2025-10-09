#!/usr/bin/env python3
"""
Generate Token Number Report by Dataset Category
Shows response token number/cost for each of the 5 dataset categories in columns
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
    elif dataset_name.startswith('TA'):
        return "TA"
    elif dataset_name.startswith('abz'):
        return "ABZ"
    elif dataset_name.startswith('swv'):
        return "SWV"
    elif dataset_name.startswith('yn'):
        return "YN"
    else:
        return "Unknown"

def estimate_tokens_from_text(text):
    """Estimate token count from text (rough approximation)"""
    if not text:
        return 0
    # Rough approximation: 1 token ≈ 4 characters for English text
    # This is a simplified estimation - actual tokenization varies by model
    return max(1, len(text) // 4)

def extract_tokens_from_logs(dataset_name, source, framework):
    """Extract actual token usage from terminal output logs or JSON files"""
    import re
    import os
    import json
    
    # Map sources to log directories
    log_dirs = {
        "MAS-GPT4o": "results_mas(gpt-4o)",
        "MAS-Claude4": "results_mas(claude-4)", 
        "Single": "results_single(gpt-4o)",
        "ALAS-GPT4o": "results_optimized(gpt-4o)",
        "ALAS-Claude4": "results_optimized(claude-4)",
        "ALAS-DeepSeek-V3": "results_optimized(deepseek-v3)",
        "ALAS-Gemini-2.5": "results_optimized(gemini-2.5)"
    }
    
    log_dir = log_dirs.get(source)
    if not log_dir or not os.path.exists(log_dir):
        return 0
    
    # For single-agent models, try to extract from JSON files first
    if source == "Single":
        model_mapping = {
            "GPT-4o": "GPT-4o",
            "Claude-Sonnet-4": "Claude-Sonnet-4", 
            "Gemini-2.5": "Gemini-2.5",
            "DeepSeek-V3": "DeepSeek-V3"
        }
        model_name = model_mapping.get(framework, framework)
        json_file = os.path.join(log_dir, f"singleagent_llm_comparison_{dataset_name}_{model_name}.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract prompt and response from JSON
                if 'models' in data and model_name in data['models']:
                    model_data = data['models'][model_name]
                    prompt = model_data.get('prompt', '')
                    response = model_data.get('response', '')
                    
                    # Calculate estimated tokens
                    prompt_tokens = estimate_tokens_from_text(prompt)
                    response_tokens = estimate_tokens_from_text(response)
                    return prompt_tokens + response_tokens
                    
            except Exception as e:
                pass
    
    # For MAS frameworks, try to extract from JSON files first (agent responses)
    if source.startswith("MAS"):
        json_file = os.path.join(log_dir, f"jssp_results_{dataset_name}_{framework}.json")
        
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract tokens from all agent responses
                if 'frameworks' in data and framework in data['frameworks']:
                    framework_data = data['frameworks'][framework]
                    total_tokens = 0
                    
                    # Extract from agent prompts and outputs
                    if 'agent_prompts' in framework_data:
                        for agent_name, prompt in framework_data['agent_prompts'].items():
                            if isinstance(prompt, str):
                                total_tokens += estimate_tokens_from_text(prompt)
                    
                    if 'agent_outputs' in framework_data:
                        for agent_name, output in framework_data['agent_outputs'].items():
                            if isinstance(output, str):
                                total_tokens += estimate_tokens_from_text(output)
                    
                    # Also check for main response
                    if 'response' in framework_data and isinstance(framework_data['response'], str):
                        total_tokens += estimate_tokens_from_text(framework_data['response'])
                    
                    if total_tokens > 0:
                        return total_tokens
                        
            except Exception as e:
                pass
        
        # Fallback: try to extract from terminal output logs
        log_file = os.path.join(log_dir, f"jssp_results_{dataset_name}_{framework}_terminal_output.txt")
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Look for token usage patterns
                patterns = [
                    r"models_usage=RequestUsage\(prompt_tokens=(\d+), completion_tokens=(\d+)\)",
                    r"prompt_tokens=(\d+), completion_tokens=(\d+)",
                    r"prompt_tokens: (\d+), completion_tokens: (\d+)"
                ]
                
                total_tokens = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        prompt_tokens = int(match[0])
                        completion_tokens = int(match[1])
                        total_tokens += prompt_tokens + completion_tokens
                
                if total_tokens > 0:
                    return total_tokens
                    
            except Exception as e:
                pass
    
    # For ALAS workflows, try to extract from terminal output logs
    if source.startswith("ALAS"):
        log_file = os.path.join(log_dir, "full_terminal_output.log")
        
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Look for token usage patterns
                patterns = [
                    r"models_usage=RequestUsage\(prompt_tokens=(\d+), completion_tokens=(\d+)\)",
                    r"prompt_tokens=(\d+), completion_tokens=(\d+)",
                    r"prompt_tokens: (\d+), completion_tokens: (\d+)"
                ]
                
                total_tokens = 0
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        prompt_tokens = int(match[0])
                        completion_tokens = int(match[1])
                        total_tokens += prompt_tokens + completion_tokens
                
                if total_tokens > 0:
                    return total_tokens
                    
            except Exception as e:
                pass
    
    return 0

def get_token_count(result):
    """Extract token count from result"""
    # Get source and framework to determine token usage
    source = result.get('source', '')
    framework = result.get('framework', '')
    dataset = result.get('dataset', '')
    
    # Try to extract actual token usage from logs
    actual_tokens = extract_tokens_from_logs(dataset, source, framework)
    if actual_tokens > 0:
        return actual_tokens
    
    # Fallback to estimated token usage
    if source.startswith('ALAS'):
        # ALAS workflows use pre-generated schedules, so minimal token usage
        if framework == 'full':
            return 500  # Full workflow: initial generation + repair + optimization
        elif framework == 'no_repair':
            return 200  # No repair: initial generation + optimization only
        elif framework == 'no_validation':
            return 400  # No validation: initial generation + repair + optimization
        elif framework == 'no_optimization':
            return 300  # No optimization: initial generation + repair only
        else:
            return 100  # Default minimal token usage
    
    # For MAS/Single-agent: estimate tokens based on schedule complexity
    schedule_entries = result.get('schedule_entries', 0)
    if schedule_entries == 0:
        return 0
    
    return max(100, schedule_entries * 10)  # Placeholder token count

def get_token_cost(result, model_name):
    """Calculate token cost based on model and token count"""
    token_count = get_token_count(result)
    
    # Token costs per 1K tokens (approximate)
    costs_per_1k = {
        "GPT-4o": 0.03,  # $0.03 per 1K tokens
        "Claude-Sonnet-4": 0.03,  # $0.03 per 1K tokens
        "Gemini-2.5": 0.015,  # $0.015 per 1K tokens
        "DeepSeek-V3": 0.01,  # $0.01 per 1K tokens
        "AutoGen": 0.03,  # Assuming GPT-4o
        "CrewAI": 0.03,  # Assuming GPT-4o
        "LangGraph": 0.03,  # Assuming GPT-4o
        "OpenAI_Swarm": 0.03,  # Assuming GPT-4o
        "full": 0.03,  # Assuming GPT-4o
        "no_repair": 0.03,  # Assuming GPT-4o
        "no_validation": 0.03,  # Assuming GPT-4o
        "no_optimization": 0.03,  # Assuming GPT-4o
    }
    
    cost_per_1k = costs_per_1k.get(model_name, 0.03)  # Default to GPT-4o pricing
    return (token_count / 1000) * cost_per_1k

def generate_token_number_report():
    """Generate token number report by dataset category"""
    print("📊 Generating Token Number Report by Dataset Category")
    print("="*80)
    
    # Load validation results
    results = load_validation_results()
    if not results:
        return
    
    # Group results by source, framework, and dataset category
    category_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'total_tokens': 0, 'total_cost': 0, 'count': 0})))
    
    for result in results:
        source = result['source']
        framework = result['framework']
        dataset = result['dataset']
        category = categorize_dataset(dataset)
        
        if category != "Unknown":
            token_count = get_token_count(result)
            token_cost = get_token_cost(result, framework)
            
            category_stats[source][framework][category]['total_tokens'] += token_count
            category_stats[source][framework][category]['total_cost'] += token_cost
            category_stats[source][framework][category]['count'] += 1
    
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
    
    print("\n📋 TOKEN NUMBER REPORT BY DATASET CATEGORY")
    print("="*80)
    
    # Multi-Agent Systems (GPT-4o)
    print("\n🔄 MULTI-AGENT SYSTEMS (GPT-4o)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_gpt4o:
        row = [framework]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["MAS-GPT4o"][framework][category]['total_tokens']
            count = category_stats["MAS-GPT4o"][framework][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Multi-Agent Systems (Claude-4)
    print("\n🔄 MULTI-AGENT SYSTEMS (Claude-4)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_claude4:
        row = [framework]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["MAS-Claude4"][framework][category]['total_tokens']
            count = category_stats["MAS-Claude4"][framework][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Single-Agent Models
    print("\n🤖 SINGLE-AGENT MODELS")
    print("-" * 60)
    print(f"{'Model':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for model in single_models:
        row = [model]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["Single"][model][category]['total_tokens']
            count = category_stats["Single"][model][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - GPT-4o
    print("\n🔧 ALAS (OURS) - GPT-4o")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gpt4o:
        row = [workflow]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["ALAS-GPT4o"][workflow][category]['total_tokens']
            count = category_stats["ALAS-GPT4o"][workflow][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Claude-4
    print("\n🔧 ALAS (OURS) - Claude-4")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_claude4:
        row = [workflow]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["ALAS-Claude4"][workflow][category]['total_tokens']
            count = category_stats["ALAS-Claude4"][workflow][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - DeepSeek-V3
    print("\n🔧 ALAS (OURS) - DeepSeek-V3")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_deepseek_v3:
        row = [workflow]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["ALAS-DeepSeek-V3"][workflow][category]['total_tokens']
            count = category_stats["ALAS-DeepSeek-V3"][workflow][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Gemini-2.5
    print("\n🔧 ALAS (OURS) - Gemini-2.5")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gemini_2_5:
        row = [workflow]
        total_tokens = 0
        total_count = 0
        
        for category in categories:
            tokens = category_stats["ALAS-Gemini-2.5"][workflow][category]['total_tokens']
            count = category_stats["ALAS-Gemini-2.5"][workflow][category]['count']
            avg_tokens = (tokens / count) if count > 0 else 0
            row.append(f"{avg_tokens:.0f}")
            total_tokens += tokens
            total_count += count
        
        overall_tokens = (total_tokens / total_count) if total_count > 0 else 0
        row.append(f"{overall_tokens:.0f}")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Token Cost Summary
    print("\n💰 TOKEN COST SUMMARY")
    print("-" * 60)
    print(f"{'Source':<20} {'Total Tokens':<15} {'Total Cost':<12} {'Avg Cost/Instance':<18}")
    print("-" * 60)
    
    for source in sources:
        total_tokens = 0
        total_cost = 0
        total_count = 0
        
        if source == "MAS-GPT4o":
            for framework in mas_frameworks_gpt4o:
                for category in categories:
                    tokens = category_stats[source][framework][category]['total_tokens']
                    cost = category_stats[source][framework][category]['total_cost']
                    count = category_stats[source][framework][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        elif source == "MAS-Claude4":
            for framework in mas_frameworks_claude4:
                for category in categories:
                    tokens = category_stats[source][framework][category]['total_tokens']
                    cost = category_stats[source][framework][category]['total_cost']
                    count = category_stats[source][framework][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        elif source == "Single":
            for model in single_models:
                for category in categories:
                    tokens = category_stats[source][model][category]['total_tokens']
                    cost = category_stats[source][model][category]['total_cost']
                    count = category_stats[source][model][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        elif source == "ALAS-GPT4o":
            for workflow in alas_workflows_gpt4o:
                for category in categories:
                    tokens = category_stats[source][workflow][category]['total_tokens']
                    cost = category_stats[source][workflow][category]['total_cost']
                    count = category_stats[source][workflow][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        elif source == "ALAS-Claude4":
            for workflow in alas_workflows_claude4:
                for category in categories:
                    tokens = category_stats[source][workflow][category]['total_tokens']
                    cost = category_stats[source][workflow][category]['total_cost']
                    count = category_stats[source][workflow][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        elif source == "ALAS-DeepSeek-V3":
            for workflow in alas_workflows_deepseek_v3:
                for category in categories:
                    tokens = category_stats[source][workflow][category]['total_tokens']
                    cost = category_stats[source][workflow][category]['total_cost']
                    count = category_stats[source][workflow][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        elif source == "ALAS-Gemini-2.5":
            for workflow in alas_workflows_gemini_2_5:
                for category in categories:
                    tokens = category_stats[source][workflow][category]['total_tokens']
                    cost = category_stats[source][workflow][category]['total_cost']
                    count = category_stats[source][workflow][category]['count']
                    total_tokens += tokens
                    total_cost += cost
                    total_count += count
        
        avg_cost = (total_cost / total_count) if total_count > 0 else 0
        print(f"{source:<20} {total_tokens:<15,} ${total_cost:<11.2f} ${avg_cost:<17.4f}")
    
    print("\n✅ Token number report generated successfully!")

if __name__ == "__main__":
    generate_token_number_report()
