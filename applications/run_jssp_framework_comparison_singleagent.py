#!/usr/bin/env python3
"""
Single-Agent JSSP LLM Comparison
Compares different LLM models (GPT-4o, DeepSeek-R1, Claude-3.5, Gemini) on Job Shop Scheduling Problems
All models use single-agent approach for consistent comparison
"""

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import dataset loading function with argument parsing prevention
import importlib.util
import sys

# Temporarily replace sys.argv to prevent argument parsing in imported module
original_argv = sys.argv
sys.argv = ['multiagent-jssp1-dmu.py']  # Minimal argv to prevent argument parsing

try:
    jssp_dmu_spec = importlib.util.spec_from_file_location("multiagent_jssp1_dmu", "applications/multiagent-jssp1-dmu.py")
    jssp_dmu_module = importlib.util.module_from_spec(jssp_dmu_spec)
    jssp_dmu_spec.loader.exec_module(jssp_dmu_module)
    load_dmu_dataset = jssp_dmu_module.load_dmu_dataset
finally:
    sys.argv = original_argv  # Restore original argv

class SingleAgentJSSPComparison:
    """
    Single-agent comparison system for JSSP LLM models
    Compares different LLM models using single-agent approach
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize the comparison system with API keys for different models"""
        self.api_keys = api_keys
        
        # Set environment variables for different models
        if 'openai' in api_keys:
            os.environ["OPENAI_API_KEY"] = api_keys['openai']
        if 'anthropic' in api_keys:
            os.environ["ANTHROPIC_API_KEY"] = api_keys['anthropic']
        if 'google' in api_keys:
            os.environ["GOOGLE_API_KEY"] = api_keys['google']
        if 'deepseek' in api_keys:
            os.environ["DEEPSEEK_API_KEY"] = api_keys['deepseek']
        
    def run_gpt4o(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run GPT-4o model on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running GPT-4o on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_keys['openai'])
            
            # Create JSSP query
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Run GPT-4o
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert Job Shop Scheduling Problem solver. Analyze the problem, create schedules, resolve conflicts, and find the minimum makespan solution."},
                    {"role": "user", "content": jssp_query}
                ],
                max_tokens=8192,  # Maximum tokens for GPT-4o
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            makespan = self._extract_makespan_from_response(response_text)
            structured_schedule = self._extract_structured_schedule(response_text)
            
            return {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response_text,
                'prompt': jssp_query,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_deepseek_v3(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run DeepSeek-V3 model on JSSP problem using Anthropic API compatibility"""
        print(f"\n{'='*60}")
        print(f"Running DeepSeek-V3 on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            import anthropic
            import os
            
            # Set environment variables for DeepSeek Anthropic API compatibility
            os.environ["ANTHROPIC_BASE_URL"] = "https://api.deepseek.com/anthropic"
            os.environ["ANTHROPIC_API_KEY"] = self.api_keys['deepseek']
            
            client = anthropic.Anthropic(
                api_key=self.api_keys['deepseek'],
                base_url="https://api.deepseek.com/anthropic"
            )
            
            # Create JSSP query
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Run DeepSeek-V3 using Anthropic API format
            response = client.messages.create(
                model="deepseek-chat",  # DeepSeek will map to appropriate model
                max_tokens=8192,  # Maximum tokens for DeepSeek
                temperature=0.1,
                system="You are an expert Job Shop Scheduling Problem solver. Analyze the problem, create schedules, resolve conflicts, and find the minimum makespan solution.",
                messages=[
                    {"role": "user", "content": jssp_query}
                ]
            )
            
            response_text = response.content[0].text
            makespan = self._extract_makespan_from_response(response_text)
            structured_schedule = self._extract_structured_schedule(response_text)
            
            return {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response_text,
                'prompt': jssp_query,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_claude_sonnet_4(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run Claude Sonnet 4 model on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running Claude Sonnet 4 on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            import anthropic
            
            # Debug: Print the API key being used (first 10 and last 10 characters)
            api_key = self.api_keys['anthropic']
            print(f"🔑 Claude API Key: {api_key[:10]}...{api_key[-10:]}")
            
            client = anthropic.Anthropic(api_key=api_key)
            
            # Create JSSP query
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Run Claude Sonnet 4
            response = client.messages.create(
                model="claude-sonnet-4-20250514",  # Using Claude 4 (Sonnet 4)
                max_tokens=8192,  # Maximum tokens for Claude
                temperature=0.1,
                system="You are an expert Job Shop Scheduling Problem solver. Analyze the problem, create schedules, resolve conflicts, and find the minimum makespan solution.",
                messages=[
                    {"role": "user", "content": jssp_query}
                ]
            )
            
            response_text = response.content[0].text
            makespan = self._extract_makespan_from_response(response_text)
            structured_schedule = self._extract_structured_schedule(response_text)
            
            return {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response_text,
                'prompt': jssp_query,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    def run_gemini_25(self, dataset_name: str, jobs: List[Dict]) -> Dict[str, Any]:
        """Run Gemini 2.5 model on JSSP problem"""
        print(f"\n{'='*60}")
        print(f"Running Gemini 2.5 on {dataset_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_keys['google'])
            model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Using Gemini 2.0 as 2.5 equivalent
            
            # Create JSSP query
            jssp_query = self._create_jssp_query(dataset_name, jobs)
            
            # Run Gemini 2.5
            response = model.generate_content(
                f"You are an expert Job Shop Scheduling Problem solver. Analyze the problem, create schedules, resolve conflicts, and find the minimum makespan solution.\n\n{jssp_query}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,  # Maximum tokens for Gemini
                    temperature=0.1
                )
            )
            
            response_text = response.text
            makespan = self._extract_makespan_from_response(response_text)
            structured_schedule = self._extract_structured_schedule(response_text)
            
            return {
                'success': True,
                'makespan': makespan,
                'structured_schedule': structured_schedule,
                'response': response_text,
                'prompt': jssp_query,
                'execution_time': time.time() - start_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }
    
    
    def _create_jssp_query(self, dataset_name: str, jobs: List[Dict]) -> str:
        """Create a JSSP query for LLM models"""
        query = f"""
        Job Shop Scheduling Problem (JSSP) - Dataset: {dataset_name}
        
        Problem Description:
        - Number of jobs: {len(jobs)}
        - Each job has multiple operations that must be performed in sequence
        - Each operation requires a specific machine and has a duration
        - Each machine can only process one operation at a time
        - Goal: Find the minimum makespan (total completion time)
        
        Job Specifications:
        """
        
        for job in jobs:
            query += f"\n{job['name']}:"
            for i, (machine, duration) in enumerate(job['steps']):
                query += f"\n  Step {i+1}: Machine {machine}, Duration {duration}"
        
        query += """
        
        REQUIRED OUTPUT FORMAT:
        You must provide your solution in the following exact format:
        
        1. FINAL MAKESPAN: [integer value]
        2. STRUCTURED SCHEDULE:
           For each operation, provide:
           - Job: [job_name]
           - Step: [step_number]
           - Machine: [machine_name]
           - Start Time: [start_time]
           - End Time: [end_time]
           - Duration: [duration]
        
        Example format:
        FINAL MAKESPAN: 25
        STRUCTURED SCHEDULE:
        - Job: Job1, Step: 1, Machine: Machine0, Start Time: 0, End Time: 3, Duration: 3
        - Job: Job1, Step: 2, Machine: Machine1, Start Time: 3, End Time: 7, Duration: 4
        - Job: Job2, Step: 1, Machine: Machine1, Start Time: 7, End Time: 10, Duration: 3
        
        Please solve this Job Shop Scheduling Problem and provide:
        1. A valid schedule with start and end times for each operation
        2. The minimum makespan (total completion time)
        3. Ensure all constraints are satisfied:
           - Job precedence: operations within a job must be sequential
           - Machine constraints: no overlapping operations on the same machine
        """
        
        return query
    
    def _extract_makespan_from_response(self, response: str) -> Optional[int]:
        """Extract makespan from LLM response using the required format"""
        import re
        
        # Look for the specific format we requested: "FINAL MAKESPAN: [integer]"
        patterns = [
            r'FINAL MAKESPAN:\s*(\d+)',
            r'FINAL MAKESPAN\s*:\s*(\d+)',
            r'FINAL MAKESPAN\s*=\s*(\d+)',
            # Fallback patterns for other formats
            r'makespan[:\s]*(\d+)',
            r'total time[:\s]*(\d+)',
            r'completion time[:\s]*(\d+)',
            r'finish time[:\s]*(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _extract_structured_schedule(self, response: str) -> List[Dict]:
        """Extract structured schedule from LLM response"""
        import re
        
        schedule = []
        
        # Look for the structured schedule format we requested
        # Pattern: "Job: [name], Step: [num], Machine: [name], Start Time: [num], End Time: [num], Duration: [num]"
        pattern = r'Job:\s*([^,]+),\s*Step:\s*(\d+),\s*Machine:\s*([^,]+),\s*Start Time:\s*(\d+),\s*End Time:\s*(\d+),\s*Duration:\s*(\d+)'
        
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for match in matches:
            job_name, step_num, machine_name, start_time, end_time, duration = match
            schedule.append({
                'job': job_name.strip(),
                'step': int(step_num),
                'machine': machine_name.strip(),
                'start': int(start_time),
                'end': int(end_time),
                'duration': int(duration)
            })
        
        return schedule
    
    def run_comparison(self, dataset_name: str, dataset_path: str, models: List[str] = None) -> Dict[str, Any]:
        """Run comparison across specified LLM models"""
        if models is None:
            models = ['GPT-4o', 'DeepSeek-V3', 'Claude-Sonnet-4', 'Gemini-2.5']
        
        print(f"\n{'='*80}")
        print(f"Single-Agent JSSP LLM Comparison - Dataset: {dataset_name}")
        print(f"{'='*80}")
        
        # Load dataset
        jobs = load_dmu_dataset(dataset_path)
        print(f"Loaded {len(jobs)} jobs from {dataset_name}")
        
        # Model methods
        model_methods = {
            'GPT-4o': self.run_gpt4o,
            'DeepSeek-V3': self.run_deepseek_v3,
            'Claude-Sonnet-4': self.run_claude_sonnet_4,
            'Gemini-2.5': self.run_gemini_25
        }
        
        # Run each model
        comparison_results = {
            'dataset': dataset_name,
            'num_jobs': len(jobs),
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name in models:
            if model_name in model_methods:
                print(f"\n🔄 Testing {model_name}...")
                result = model_methods[model_name](dataset_name, jobs)
                comparison_results['models'][model_name] = result
                
                if result['success']:
                    makespan = result.get('makespan', 'N/A')
                    schedule_count = len(result.get('structured_schedule', []))
                    exec_time = result['execution_time']
                    print(f"✅ {model_name}: Makespan = {makespan}, Schedule_Ops = {schedule_count}, Time = {exec_time:.2f}s")
                else:
                    print(f"❌ {model_name}: Failed - {result.get('error', 'Unknown error')}")
            else:
                print(f"⚠️ Model {model_name} not found")
        
        return comparison_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save comparison results to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_file}")
    
    def save_terminal_output_to_txt(self, results: Dict[str, Any], output_file: str):
        """Save full terminal output to TXT file"""
        txt_file = output_file.replace('.json', '_terminal_output.txt')
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("SINGLE-AGENT LLM JSSP COMPARISON - TERMINAL OUTPUT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Dataset: {results['dataset']}\n")
            f.write(f"Jobs: {results['num_jobs']}\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            for model_name, result in results['models'].items():
                f.write(f"🔍 {model_name} Model Details:\n")
                f.write("-" * 50 + "\n")
                
                if result['success']:
                    f.write(f"✅ Status: Success\n")
                    f.write(f"📊 Makespan: {result.get('makespan', 'N/A')}\n")
                    f.write(f"📋 Schedule Operations: {len(result.get('structured_schedule', []))}\n")
                    f.write(f"⏱️ Execution Time: {result['execution_time']:.2f}s\n\n")
                    
                    # Write full LLM response
                    f.write("🤖 Full LLM Response:\n")
                    f.write("-" * 30 + "\n")
                    f.write(result.get('response', 'No response available') + "\n\n")
                    
                    # Write structured schedule
                    if result.get('structured_schedule'):
                        f.write("📅 Structured Schedule:\n")
                        f.write("-" * 30 + "\n")
                        for entry in result['structured_schedule']:
                            f.write(f"Job: {entry['job']}, Step: {entry['step']}, Machine: {entry['machine']}, Start: {entry['start']}, End: {entry['end']}, Duration: {entry['duration']}\n")
                        f.write("\n")
                else:
                    f.write(f"❌ Status: Failed\n")
                    f.write(f"🚨 Error: {result.get('error', 'Unknown error')}\n")
                    f.write(f"⏱️ Execution Time: {result['execution_time']:.2f}s\n\n")
                
                f.write("=" * 50 + "\n\n")
        
        print(f"📄 Terminal output saved to TXT: {txt_file}")
    
    def save_agent_details_to_txt(self, results: Dict[str, Any], output_file: str):
        """Save agent prompts and outputs to TXT file"""
        txt_file = output_file.replace('.json', '_agent_details.txt')
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("SINGLE-AGENT LLM PROMPTS AND OUTPUTS DETAILS\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, result in results['models'].items():
                f.write(f"🔍 {model_name} Model Details:\n")
                f.write("-" * 50 + "\n")
                
                if result['success']:
                    # Write the JSSP query/prompt that was sent to the LLM
                    f.write(f"\n📝 {model_name} Input Prompt:\n")
                    f.write("-" * 30 + "\n")
                    f.write(result.get('prompt', 'No prompt available') + "\n\n")
                    
                    # Write the full LLM response
                    f.write(f"\n💬 {model_name} Full Output:\n")
                    f.write("-" * 30 + "\n")
                    f.write(result.get('response', 'No response available') + "\n\n")
                    
                    # Write extracted makespan
                    f.write(f"\n📊 Extracted Makespan: {result.get('makespan', 'N/A')}\n")
                    
                    # Write schedule count
                    f.write(f"📋 Schedule Operations Count: {len(result.get('structured_schedule', []))}\n")
                    
                else:
                    f.write(f"❌ {model_name} Failed\n")
                    f.write(f"🚨 Error: {result.get('error', 'Unknown error')}\n")
                
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"📄 Agent details saved to TXT: {txt_file}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print comparison summary"""
        print(f"\n{'='*80}")
        print("SINGLE-AGENT JSSP LLM COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Dataset: {results['dataset']}")
        print(f"Jobs: {results['num_jobs']}")
        print(f"Timestamp: {results['timestamp']}")
        
        print(f"\n{'Model':<15} {'Success':<8} {'Makespan':<10} {'Schedule':<10} {'Time (s)':<10} {'Status'}")
        print("-" * 80)
        
        for model, result in results['models'].items():
            success = "✅" if result['success'] else "❌"
            makespan = str(result.get('makespan', 'N/A')) if result['success'] else 'N/A'
            schedule_count = len(result.get('structured_schedule', [])) if result['success'] else 0
            exec_time = f"{result['execution_time']:.2f}"
            status = "Success" if result['success'] else f"Error: {result.get('error', 'Unknown')[:20]}..."
            
            print(f"{model:<15} {success:<8} {makespan:<10} {schedule_count:<10} {exec_time:<10} {status}")
        
        # Performance ranking
        print(f"\n{'='*80}")
        print("PERFORMANCE RANKING (by Makespan)")
        print(f"{'='*80}")
        
        successful_results = [(name, result) for name, result in results['models'].items() if result['success']]
        successful_results.sort(key=lambda x: x[1].get('makespan') if x[1].get('makespan') is not None else float('inf'))
        
        for i, (model, result) in enumerate(successful_results, 1):
            makespan = result.get('makespan', 'N/A')
            schedule_count = len(result.get('structured_schedule', []))
            print(f"{i}. {model}: Makespan = {makespan}, Schedule_Ops = {schedule_count}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Single-Agent JSSP LLM Comparison')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., rcmax_20_15_5)')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    parser.add_argument('--anthropic-key', type=str, help='Anthropic API key')
    parser.add_argument('--google-key', type=str, help='Google API key')
    parser.add_argument('--deepseek-key', type=str, help='DeepSeek API key')
    parser.add_argument('--models', nargs='+', default=['GPT-4o', 'DeepSeek-V3', 'Claude-Sonnet-4', 'Gemini-2.5'],
                       help='LLM models to compare')
    parser.add_argument('--output', type=str, default='./results/singleagent_llm_comparison_{dataset}_{model}.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Use provided API keys, environment variables, or defaults (in that order)
    api_keys = {
        'openai': args.openai_key or os.environ.get('OPENAI_API_KEY') or 'API_KEY_TO_REPLACE',
        'anthropic': args.anthropic_key or os.environ.get('ANTHROPIC_API_KEY') or 'API_KEY_TO_REPLACE',
        'google': args.google_key or os.environ.get('GOOGLE_API_KEY') or 'API_KEY_TO_REPLACE',
        'deepseek': args.deepseek_key or os.environ.get('DEEPSEEK_API_KEY') or 'API_KEY_TO_REPLACE'
    }
    
    print("🔑 Using provided API keys for LLM comparison")
    print(f"🔑 Anthropic API Key: {api_keys['anthropic'][:10]}...{api_keys['anthropic'][-10:]}")
    print(f"🔍 Environment ANTHROPIC_API_KEY: {os.environ.get('ANTHROPIC_API_KEY', 'Not set')[:10]}...{os.environ.get('ANTHROPIC_API_KEY', 'Not set')[-10:] if os.environ.get('ANTHROPIC_API_KEY') else 'Not set'}")
    
    # Determine dataset path
    dataset_path = None
    for subdir in ['DMU', 'TA', 'abzswvyn']:
        potential_path = os.path.join(project_root, 'applications', subdir, f'{args.dataset}.txt')
        if os.path.exists(potential_path):
            dataset_path = potential_path
            break
    
    if not dataset_path:
        print(f"❌ Dataset {args.dataset} not found in DMU, TA, or ABZSWVYN directories")
        return
    
    # Initialize comparison system
    comparison = SingleAgentJSSPComparison(api_keys)
    
    # Run comparison
    results = comparison.run_comparison(args.dataset, dataset_path, args.models)
    
    # Generate output filename with dataset and model names
    if '{dataset}' in args.output and '{model}' in args.output:
        # For multiple models, create separate files for each
        for model_name in args.models:
            model_results = {
                'dataset': results['dataset'],
                'num_jobs': results['num_jobs'],
                'timestamp': results['timestamp'],
                'models': {model_name: results['models'][model_name]}
            }
            
            output_file = args.output.replace('{dataset}', args.dataset).replace('{model}', model_name)
            
            # Save and display results for this model
            comparison.save_results(model_results, output_file)
            comparison.save_terminal_output_to_txt(model_results, output_file)
            comparison.save_agent_details_to_txt(model_results, output_file)
    else:
        # Use the provided output filename as-is
        comparison.save_results(results, args.output)
        comparison.save_terminal_output_to_txt(results, args.output)
        comparison.save_agent_details_to_txt(results, args.output)
    
    comparison.print_summary(results)

if __name__ == "__main__":
    main()
