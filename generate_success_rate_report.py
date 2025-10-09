#!/usr/bin/env python3
"""
Generate Success Rate Report by Dataset Category
Shows success rates for each of the 5 dataset categories in columns
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

def generate_success_rate_report():
    """Generate success rate report by dataset category"""
    print("📊 Generating Success Rate Report by Dataset Category")
    print("="*80)
    
    # Load validation results
    results = load_validation_results()
    if not results:
        return
    
    # Group results by source, framework, and dataset category
    category_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'valid': 0, 'total': 0})))
    
    for result in results:
        source = result['source']
        framework = result['framework']
        dataset = result['dataset']
        category = categorize_dataset(dataset)
        success = result['success']
        
        if category != "Unknown":
            if success:
                category_stats[source][framework][category]['valid'] += 1
            category_stats[source][framework][category]['total'] += 1
    
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
    
    print("\n📋 SUCCESS RATE REPORT BY DATASET CATEGORY")
    print("="*80)
    
    # Multi-Agent Systems (GPT-4o)
    print("\n🔄 MULTI-AGENT SYSTEMS (GPT-4o)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_gpt4o:
        row = [framework]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["MAS-GPT4o"][framework][category]['valid']
            total = category_stats["MAS-GPT4o"][framework][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Multi-Agent Systems (Claude-4)
    print("\n🔄 MULTI-AGENT SYSTEMS (Claude-4)")
    print("-" * 60)
    print(f"{'Framework':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for framework in mas_frameworks_claude4:
        row = [framework]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["MAS-Claude4"][framework][category]['valid']
            total = category_stats["MAS-Claude4"][framework][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Single-Agent Models
    print("\n🤖 SINGLE-AGENT MODELS")
    print("-" * 60)
    print(f"{'Model':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for model in single_models:
        row = [model]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["Single"][model][category]['valid']
            total = category_stats["Single"][model][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - GPT-4o
    print("\n🔧 ALAS (OURS) - GPT-4o")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gpt4o:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-GPT4o"][workflow][category]['valid']
            total = category_stats["ALAS-GPT4o"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Claude-4
    print("\n🔧 ALAS (OURS) - Claude-4")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_claude4:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-Claude4"][workflow][category]['valid']
            total = category_stats["ALAS-Claude4"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - DeepSeek-V3
    print("\n🔧 ALAS (OURS) - DeepSeek-V3")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_deepseek_v3:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-DeepSeek-V3"][workflow][category]['valid']
            total = category_stats["ALAS-DeepSeek-V3"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # ALAS (Ours) - Gemini-2.5
    print("\n🔧 ALAS (OURS) - Gemini-2.5")
    print("-" * 60)
    print(f"{'Workflow':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for workflow in alas_workflows_gemini_2_5:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-Gemini-2.5"][workflow][category]['valid']
            total = category_stats["ALAS-Gemini-2.5"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}%")
        
        print(f"{row[0]:<15} {row[1]:<8} {row[2]:<8} {row[3]:<8} {row[4]:<8} {row[5]:<8} {row[6]:<8}")
    
    # Summary by category
    print("\n📊 SUMMARY BY DATASET CATEGORY")
    print("-" * 60)
    print(f"{'Category':<15} {'DMU':<8} {'TA':<8} {'ABZ':<8} {'SWV':<8} {'YN':<8} {'Overall':<8}")
    print("-" * 60)
    
    for category in categories:
        total_valid = 0
        total_count = 0
        
        for source in sources:
            if source == "MAS-GPT4o":
                for framework in mas_frameworks_gpt4o:
                    valid = category_stats[source][framework][category]['valid']
                    total = category_stats[source][framework][category]['total']
                    total_valid += valid
                    total_count += total
            elif source == "MAS-Claude4":
                for framework in mas_frameworks_claude4:
                    valid = category_stats[source][framework][category]['valid']
                    total = category_stats[source][framework][category]['total']
                    total_valid += valid
                    total_count += total
            elif source == "Single":
                for model in single_models:
                    valid = category_stats[source][model][category]['valid']
                    total = category_stats[source][model][category]['total']
                    total_valid += valid
                    total_count += total
            elif source == "ALAS-GPT4o":
                for workflow in alas_workflows_gpt4o:
                    valid = category_stats[source][workflow][category]['valid']
                    total = category_stats[source][workflow][category]['total']
                    total_valid += valid
                    total_count += total
            elif source == "ALAS-Claude4":
                for workflow in alas_workflows_claude4:
                    valid = category_stats[source][workflow][category]['valid']
                    total = category_stats[source][workflow][category]['total']
                    total_valid += valid
                    total_count += total
            elif source == "ALAS-DeepSeek-V3":
                for workflow in alas_workflows_deepseek_v3:
                    valid = category_stats[source][workflow][category]['valid']
                    total = category_stats[source][workflow][category]['total']
                    total_valid += valid
                    total_count += total
            elif source == "ALAS-Gemini-2.5":
                for workflow in alas_workflows_gemini_2_5:
                    valid = category_stats[source][workflow][category]['valid']
                    total = category_stats[source][workflow][category]['total']
                    total_valid += valid
                    total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        print(f"{category:<15} {overall_rate:.1f}%")
    
    # Generate LaTeX table
    print("\n📄 Generating LaTeX table...")
    generate_latex_table(category_stats, categories, mas_frameworks_gpt4o, mas_frameworks_claude4, single_models, alas_workflows_gpt4o, alas_workflows_claude4, alas_workflows_deepseek_v3, alas_workflows_gemini_2_5)
    
    print("\n✅ Success rate report generated successfully!")

def generate_latex_table(category_stats, categories, mas_frameworks_gpt4o, mas_frameworks_claude4, single_models, alas_workflows_gpt4o, alas_workflows_claude4, alas_workflows_deepseek_v3, alas_workflows_gemini_2_5):
    """Generate LaTeX table for the success rate report"""
    
    latex_content = [
        "\\documentclass{article}",
        "\\usepackage{amsmath}",
        "\\usepackage{amsfonts}",
        "\\usepackage{amssymb}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\usepackage{longtable}",
        "\\usepackage{xcolor}",
        "\\usepackage[a4paper, margin=1in]{geometry}",
        "\\begin{document}",
        "\\section*{Success Rate Report by Dataset Category}",
        "",
        "\\subsection*{Multi-Agent Systems (MAS)}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for Multi-Agent Systems by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Framework} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ]
    
    # Add MAS-GPT4o results
    for framework in mas_frameworks_gpt4o:
        row = [framework]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["MAS-GPT4o"][framework][category]['valid']
            total = category_stats["MAS-GPT4o"][framework][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\subsection*{Multi-Agent Systems (Claude-4)}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for Multi-Agent Systems (Claude-4) by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Framework} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ])
    
    # Add MAS-Claude4 results
    for framework in mas_frameworks_claude4:
        row = [framework]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["MAS-Claude4"][framework][category]['valid']
            total = category_stats["MAS-Claude4"][framework][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\subsection*{Single-Agent Models}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for Single-Agent Models by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Model} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ])
    
    # Add Single-Agent results
    for model in single_models:
        row = [model]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["Single"][model][category]['valid']
            total = category_stats["Single"][model][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\subsection*{ALAS (Ours) - GPT-4o}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for ALAS Workflows (GPT-4o) by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Workflow} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ])
    
    # Add ALAS-GPT4o results
    for workflow in alas_workflows_gpt4o:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-GPT4o"][workflow][category]['valid']
            total = category_stats["ALAS-GPT4o"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\subsection*{ALAS (Ours) - Claude-4}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for ALAS Workflows (Claude-4) by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Workflow} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ])
    
    # Add ALAS-Claude4 results
    for workflow in alas_workflows_claude4:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-Claude4"][workflow][category]['valid']
            total = category_stats["ALAS-Claude4"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    # Add ALAS-DeepSeek-V3 results
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\subsection*{ALAS (Ours) - DeepSeek-V3}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for ALAS Workflows (DeepSeek-V3) by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Workflow} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ])
    
    for workflow in alas_workflows_deepseek_v3:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-DeepSeek-V3"][workflow][category]['valid']
            total = category_stats["ALAS-DeepSeek-V3"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    # Add ALAS-Gemini-2.5 results
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "",
        "\\subsection*{ALAS (Ours) - Gemini-2.5}",
        "\\begin{longtable}{|l|c|c|c|c|c|c|}",
        "\\caption{Success Rates for ALAS Workflows (Gemini-2.5) by Dataset Category} \\\\",
        "\\toprule",
        "\\textbf{Workflow} & \\textbf{DMU} & \\textbf{TA} & \\textbf{ABZ} & \\textbf{SWV} & \\textbf{YN} & \\textbf{Overall} \\\\",
        "\\midrule",
        "\\endhead"
    ])
    
    for workflow in alas_workflows_gemini_2_5:
        row = [workflow]
        total_valid = 0
        total_count = 0
        
        for category in categories:
            valid = category_stats["ALAS-Gemini-2.5"][workflow][category]['valid']
            total = category_stats["ALAS-Gemini-2.5"][workflow][category]['total']
            rate = (valid / total * 100) if total > 0 else 0
            row.append(f"{rate:.1f}\\%")
            total_valid += valid
            total_count += total
        
        overall_rate = (total_valid / total_count * 100) if total_count > 0 else 0
        row.append(f"{overall_rate:.1f}\\%")
        
        latex_content.append(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} & {row[5]} & {row[6]} \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{longtable}",
        "\\end{document}"
    ])
    
    # Save LaTeX file
    latex_file = "success_rate_report_by_category.tex"
    with open(latex_file, 'w') as f:
        f.write('\n'.join(latex_content))
    
    print(f"📄 LaTeX table saved to: {latex_file}")

if __name__ == "__main__":
    generate_success_rate_report()
