#!/usr/bin/env python3
"""
Extract makespan data from single-agent LLM comparison results
"""

import json
import os
import glob
from collections import defaultdict

def extract_makespan_data(results_dir):
    """Extract makespan data from all result files"""
    
    # Initialize data structure
    results = defaultdict(lambda: defaultdict(dict))
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(results_dir, "singleagent_llm_comparison_*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            dataset = data.get('dataset', 'unknown')
            models = data.get('models', {})
            
            for model_name, model_data in models.items():
                success = model_data.get('success', False)
                makespan = model_data.get('makespan', None)
                
                if success and makespan is not None:
                    results[dataset][model_name] = makespan
                else:
                    results[dataset][model_name] = "N/A"
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results

def generate_latex_table(results):
    """Generate LaTeX table from results"""
    
    # Get all unique models
    all_models = set()
    for dataset_data in results.values():
        all_models.update(dataset_data.keys())
    all_models = sorted(list(all_models))
    
    # Get all datasets sorted by category
    datasets = sorted(results.keys())
    
    # Group datasets by category
    abz_datasets = [d for d in datasets if d.startswith('abz')]
    ta_datasets = [d for d in datasets if d.startswith('TA')]
    swv_datasets = [d for d in datasets if d.startswith('swv')]
    rcmax_datasets = [d for d in datasets if d.startswith('rcmax')]
    yn_datasets = [d for d in datasets if d.startswith('yn')]
    
    latex_content = []
    latex_content.append("\\documentclass{article}")
    latex_content.append("\\usepackage[utf8]{inputenc}")
    latex_content.append("\\usepackage{booktabs}")
    latex_content.append("\\usepackage{multirow}")
    latex_content.append("\\usepackage{array}")
    latex_content.append("\\usepackage{geometry}")
    latex_content.append("\\usepackage{longtable}")
    latex_content.append("\\geometry{margin=0.5in}")
    latex_content.append("")
    latex_content.append("\\title{Comprehensive Single-Agent LLM Makespan Results}")
    latex_content.append("\\author{Experimental Results}")
    latex_content.append("\\date{\\today}")
    latex_content.append("")
    latex_content.append("\\begin{document}")
    latex_content.append("")
    latex_content.append("\\maketitle")
    latex_content.append("")
    latex_content.append("\\section{Complete Results Summary}")
    latex_content.append("")
    latex_content.append("This document presents comprehensive makespan results from single-agent LLM comparison experiments on Job Shop Scheduling Problem (JSSP) datasets.")
    latex_content.append("")
    
    # Create tables for each category
    categories = [
        ("ABZ Datasets", abz_datasets),
        ("TA Datasets", ta_datasets), 
        ("SWV Datasets", swv_datasets),
        ("RCMAX Datasets", rcmax_datasets),
        ("YN Datasets", yn_datasets)
    ]
    
    for category_name, category_datasets in categories:
        if not category_datasets:
            continue
            
        latex_content.append(f"\\subsection{{{category_name}}}")
        latex_content.append("")
        latex_content.append("\\begin{longtable}{@{}l" + "c" * len(all_models) + "@{}}")
        latex_content.append("\\toprule")
        
        # Header
        header = "\\textbf{Dataset} & " + " & ".join([f"\\textbf{{{model}}}" for model in all_models]) + " \\\\"
        latex_content.append(header)
        latex_content.append("\\midrule")
        latex_content.append("\\endhead")
        
        # Data rows
        for dataset in category_datasets:
            if dataset in results:
                row_data = [f"\\textbf{{{dataset}}}"]
                for model in all_models:
                    makespan = results[dataset].get(model, "N/A")
                    if makespan == "N/A":
                        row_data.append("N/A")
                    else:
                        row_data.append(str(makespan))
                latex_content.append(" & ".join(row_data) + " \\\\")
        
        latex_content.append("\\bottomrule")
        latex_content.append("\\end{longtable}")
        latex_content.append("")
    
    # Summary statistics
    latex_content.append("\\section{Summary Statistics}")
    latex_content.append("")
    
    # Calculate success rates
    model_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    
    for dataset_data in results.values():
        for model in all_models:
            model_stats[model]['total'] += 1
            if dataset_data.get(model, "N/A") != "N/A":
                model_stats[model]['success'] += 1
    
    latex_content.append("\\begin{table}[h!]")
    latex_content.append("\\centering")
    latex_content.append("\\caption{Model Success Rates}")
    latex_content.append("\\begin{tabular}{@{}lccc@{}}")
    latex_content.append("\\toprule")
    latex_content.append("\\textbf{Model} & \\textbf{Successful} & \\textbf{Total} & \\textbf{Success Rate} \\\\")
    latex_content.append("\\midrule")
    
    for model in all_models:
        success = model_stats[model]['success']
        total = model_stats[model]['total']
        rate = (success / total * 100) if total > 0 else 0
        latex_content.append(f"{model} & {success} & {total} & {rate:.1f}\\% \\\\")
    
    latex_content.append("\\bottomrule")
    latex_content.append("\\end{tabular}")
    latex_content.append("\\end{table}")
    latex_content.append("")
    
    # Best performance analysis
    latex_content.append("\\section{Best Performance Analysis}")
    latex_content.append("")
    
    for dataset in sorted(results.keys()):
        dataset_data = results[dataset]
        valid_results = {model: makespan for model, makespan in dataset_data.items() if makespan != "N/A"}
        
        if valid_results:
            best_model = min(valid_results.keys(), key=lambda x: valid_results[x])
            best_makespan = valid_results[best_model]
            
            latex_content.append(f"\\textbf{{{dataset}}}: {best_model} achieved the best makespan of {best_makespan}")
            latex_content.append("")
    
    latex_content.append("\\end{document}")
    
    return "\\n".join(latex_content)

def main():
    results_dir = "/Users/glin/Documents/GitHub/M-APPLE/M-APPLE-OS/results_single(gpt-4o)"
    
    print("Extracting makespan data...")
    results = extract_makespan_data(results_dir)
    
    print(f"Found {len(results)} datasets")
    for dataset, data in results.items():
        print(f"{dataset}: {len(data)} models")
    
    print("\\nGenerating LaTeX table...")
    latex_content = generate_latex_table(results)
    
    output_file = "/Users/glin/Documents/GitHub/M-APPLE/M-APPLE-OS/comprehensive_single_agent_results.tex"
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {output_file}")
    
    # Also create a simple summary
    print("\\n=== SUMMARY ===")
    for dataset in sorted(results.keys()):
        print(f"\\n{dataset}:")
        for model, makespan in results[dataset].items():
            print(f"  {model}: {makespan}")

if __name__ == "__main__":
    main()
