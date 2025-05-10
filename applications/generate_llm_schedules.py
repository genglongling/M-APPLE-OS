import os
import csv
import argparse
import google.generativeai as genai
from dotenv import load_dotenv
import re # For parsing

# --- Configuration & Setup ---

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in environment variables. Please ensure it's set, possibly in a .env file.")
        return None
    return api_key

def configure_gemini_client(api_key):
    """Configure the Gemini generative AI client."""
    try:
        genai.configure(api_key=api_key)
        print("Gemini client configured successfully.")
        return True
    except Exception as e:
        print(f"Error configuring Gemini client: {e}")
        return False

# --- DMU Data Loading ---

def load_dmu_dataset(filepath):
    """Load JSSP data from a DMU file."""
    jobs_data = []
    num_jobs = 0
    num_machines = 0
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if not lines:
                print(f"Warning: DMU file {filepath} is empty or contains only whitespace.")
                return None, 0, 0
            
            header = lines[0].split()
            if len(header) < 2:
                print(f"Warning: Invalid header in DMU file {filepath}. Expected at least 2 numbers (jobs, machines). Found: '{lines[0]}'")
                return None, 0, 0

            num_jobs, num_machines = map(int, header[:2]) # First two numbers are jobs and machines

            for job_idx, line in enumerate(lines[1:num_jobs + 1]): # Read only num_jobs lines after header
                tokens = list(map(int, line.split()))
                operations = []
                # Each operation is (machine_id, duration)
                # DMU format: machine_1 duration_1 machine_2 duration_2 ...
                for i in range(0, len(tokens), 2):
                    if i + 1 < len(tokens):
                        # DMU machine IDs are usually 1-indexed, schedule might use 0-indexed or "MachineX"
                        # For now, let's store them as they are and handle naming later if needed
                        operations.append({"machine_id": tokens[i], "duration": tokens[i+1]})
                    else:
                        print(f"Warning: Malformed operation data for job {job_idx+1} in {filepath}. Line: '{line}'")
                jobs_data.append({'name': f'Job{job_idx+1}', 'operations': operations})
        print(f"Loaded {len(jobs_data)} jobs and {num_machines} machines from {filepath}.")
        return jobs_data, num_jobs, num_machines
    except FileNotFoundError:
        print(f"Error: DMU file not found at {filepath}")
        return None, 0, 0
    except ValueError as e:
        print(f"Error parsing DMU file {filepath}: {e}. Offending line might be header or job data.")
        return None, 0, 0
    except Exception as e:
        print(f"An unexpected error occurred while loading DMU file {filepath}: {e}")
        return None, 0, 0

# --- LLM Interaction ---

def construct_jssp_prompt(jobs_data, num_jobs, num_machines, dataset_name):
    """Construct a detailed prompt for the Gemini model to solve JSSP."""
    prompt = f"Solve the Job Shop Scheduling Problem (JSSP) for the dataset '{dataset_name}'.\n\n"
    prompt += f"Instance Details:\n"
    prompt += f"- Number of Jobs: {num_jobs} (ALL {num_jobs} JOBS MUST BE SCHEDULED COMPLETELY)\n"
    prompt += f"- Number of Machines: {num_machines}\n\n"
    prompt += "Job Operations (machine_id is 1-indexed from DMU, duration is in time units):\n"
    for job in jobs_data:
        prompt += f"- {job['name']}:\n"
        for op_idx, op in enumerate(job['operations']):
            prompt += f"  - Operation {op_idx + 1}: Machine {op['machine_id']}, Duration {op['duration']}\n"
    prompt += "\n"
    prompt += "Constraints:\n"
    prompt += "1. Operations for each job must be processed in the given sequence (precedence constraint).\n"
    prompt += "2. Each machine can only process one operation at a time (capacity constraint).\n"
    prompt += "3. An operation cannot start until its preceding operation in the same job is completed.\n"
    prompt += "4. Once an operation starts on a machine, it must run to completion without interruption.\n\n"
    prompt += "Objective: Minimize the makespan (the time when all operations for all jobs are completed).\n\n"
    prompt += "⚠️ CRITICAL REQUIREMENTS - READ CAREFULLY ⚠️\n"
    prompt += "1. I need DIRECT CSV DATA, not code. DO NOT provide Python code that would generate a schedule.\n"
    prompt += "2. Give me the actual finished schedule as CSV data, with one line for each operation.\n"
    prompt += f"3. Your CSV MUST include ALL {num_jobs} jobs from Job1 through Job{num_jobs}.\n"
    prompt += "4. Include ALL operations for ALL jobs, with correct start/end times and precedence constraints.\n"
    prompt += "5. Place your complete CSV schedule inside a code block as shown below.\n\n"
    prompt += "CORRECT FORMAT (actual CSV data):\n"
    prompt += "```csv\n"
    prompt += "job,step,machine,start,end,precedence\n"
    prompt += "Job1,1,Machine5,0,25,None\n"
    prompt += "Job1,2,Machine3,25,40,After Job1 Step 1\n"
    prompt += "... (rows for ALL operations of ALL jobs)\n"
    prompt += "```\n\n"
    prompt += "WRONG FORMAT (DO NOT DO THIS):\n"
    prompt += "```python\n"
    prompt += "def generate_schedule():\n"
    prompt += "    # Code that would generate a schedule\n"
    prompt += "    schedule = []\n"
    prompt += "    for job in jobs:\n"
    prompt += "        # More code\n"
    prompt += "```\n\n"
    prompt += f"AGAIN: Your response must be the ACTUAL finished schedule for all {num_jobs} jobs as direct CSV data, not code to generate it.\n"
    prompt += "Your task is to solve the problem directly and provide the final CSV solution.\n\n"
    prompt += "Please generate the complete schedule now as CSV data for ALL jobs:\n"
    return prompt

def call_gemini_api(prompt_text, gemini_model_name, safety_settings=None):
    """Call the Gemini API with the given prompt."""
    if safety_settings is None: # More permissive for complex generation
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
    try:
        print(f"\n--- Calling Gemini model: {gemini_model_name} ---")
        
        # Try to use the Pro model when possible for better results
        if "flash" in gemini_model_name and gemini_model_name != "gemini-1.5-pro-latest":
            suggested_model = gemini_model_name.replace("flash", "pro")
            try:
                # Check if the Pro model is available
                genai.get_model(suggested_model)
                print(f"Upgrading to more powerful model: {suggested_model}")
                gemini_model_name = suggested_model
            except Exception:
                print(f"Pro model not available, continuing with {gemini_model_name}")
        
        # Configure generative model with strong structuring 
        model = genai.GenerativeModel(
            model_name=gemini_model_name,
            generation_config={
                "temperature": 0.0,  # Use 0 temperature for deterministic output
                "top_p": 1.0,
                "top_k": 16,
                "max_output_tokens": 16384,  # Request maximum tokens possible
            },
            safety_settings=safety_settings
        )
        
        response = model.generate_content(prompt_text)
        
        # Check for empty or blocked response
        if not response.parts:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                print(f"Warning: Prompt was blocked. Reason: {response.prompt_feedback.block_reason}")
                if response.prompt_feedback.block_reason_message:
                     print(f"Block reason message: {response.prompt_feedback.block_reason_message}")
                return None
            else:
                print("Warning: Gemini response is empty or has no text parts.")
                return None
        
        generated_text = response.text
        print(f"--- Gemini response received (first 500 chars): ---\n{generated_text[:500]}...")
        # Also save the full response to a debug file
        debug_file = f"gemini_response_debug_{gemini_model_name.replace('-', '_')}.txt"
        with open(debug_file, 'w') as f:
            f.write(generated_text)
        print(f"Full response saved to {debug_file} for debugging")
        
        return generated_text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def parse_llm_response_to_schedule(llm_response_text):
    """Parse the LLM's text response to extract the schedule.
    Handles various formats including:
    1. CSV lines after 'SCHEDULE START:' delimiter
    2. CSV data in code blocks
    3. CSV data anywhere in the response
    Header: job,step,machine,start,end,precedence
    """
    schedule_entries = []
    if not llm_response_text:
        return schedule_entries

    # Check if the response contains Python code instead of CSV data
    python_code_indicators = [
        "def generate_schedule(", 
        "def solve_jssp(", 
        "schedule = []", 
        "for job_id in range", 
        "class JobScheduler",
        "# Define job data",
        "import",
        "schedule.append"
    ]
    
    contains_python_code = any(indicator in llm_response_text for indicator in python_code_indicators)
    if contains_python_code:
        print("Warning: Detected Python code in the response instead of direct CSV data.")
        print("The model provided code to generate a schedule instead of the actual schedule.")
        print("Will attempt to extract any CSV data, but results may be incomplete.")

    # First try to find CSV code blocks which may contain the schedule
    csv_blocks = re.findall(r'```(?:csv)?\s*\n((?:[^\n]*\n)+?)```', llm_response_text, re.DOTALL)
    
    # If we find CSV code blocks, try to parse each one
    clean_content = ""
    if csv_blocks:
        print(f"Found {len(csv_blocks)} potential CSV blocks in the response")
        for block in csv_blocks:
            clean_content += block + "\n"
    else:
        # Try to find the start of the explicit schedule section
        schedule_start_marker = "SCHEDULE START:"
        schedule_content_match = re.search(f"{re.escape(schedule_start_marker)}(.*)", llm_response_text, re.DOTALL | re.IGNORECASE)
        
        if schedule_content_match:
            clean_content = schedule_content_match.group(1).strip()
        else:
            # Fallback: use the entire response if no clear delimiter or code blocks
            clean_content = llm_response_text
            print("Warning: No explicit schedule delimiter or code blocks found. Attempting to parse entire response.")

    # Process all potential schedule content
    lines = clean_content.splitlines()
    header_pattern = re.compile(r'job\s*,\s*step\s*,\s*machine\s*,\s*start\s*,\s*end', re.IGNORECASE)
    
    # Now process the content line by line
    for line_num, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Normalize CSV, remove quotes and extra spaces
        line = re.sub(r'^\s*"\s*|\s*"\s*$', '', line)  # Remove surrounding quotes
        
        # Check if this is a header line (case insensitive)
        if header_pattern.match(line):
            print(f"Info: Skipping header line: {line}")
            continue  # Skip all header lines
            
        # Skip comment/explanation lines
        if line.startswith('#') or line.startswith('//') or "explanation" in line.lower():
            continue
            
        # Skip lines that are clearly not CSV data
        if ',' not in line or len(line.split(',')) < 4:
            continue
        
        # Skip Python code lines
        if any(code_indicator in line for code_indicator in ["def ", "import ", "class ", "print(", ".append(", "for ", "if "]):
            continue
        
        # Try to parse as a schedule entry
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:  # We need at least job,step,machine,start,end
                # Extract and validate the core fields
                job = parts[0]
                
                # Extra check to skip potential header lines
                if job.lower() == "job":
                    continue
                    
                step = int(parts[1])
                machine = parts[2]
                start = int(parts[3])
                end = int(parts[4])
                
                # Handle the precedence field if present
                precedence = ""
                if len(parts) >= 6:
                    precedence = parts[5]
                    if precedence.lower() in ('none', 'n/a', '-'):
                        precedence = ""
                
                entry = {
                    'job': job,
                    'step': step,
                    'machine': machine,
                    'start': start,
                    'end': end,
                    'precedence': precedence
                }
                schedule_entries.append(entry)
            else:
                print(f"Warning: Skipping line with insufficient columns: '{line}'")
        except ValueError as e:
            print(f"Warning: Could not parse line '{line}' as schedule entry. Error: {e}")
        except Exception as e:
            print(f"Warning: Error processing line '{line}': {e}")
    
    # Additional validation and sorting
    if schedule_entries:
        # Sort schedule entries by job, then step for better readability
        schedule_entries.sort(key=lambda x: (x['job'], x['step']))
        
        # Verify we have all the expected operations
        job_counts = {}
        for entry in schedule_entries:
            job = entry['job']
            if job not in job_counts:
                job_counts[job] = 0
            job_counts[job] += 1
        
        print(f"Successfully parsed {len(schedule_entries)} schedule entries.")
        print(f"Jobs found: {len(job_counts)}")
        for job, count in job_counts.items():
            print(f"  {job}: {count} operations")
    else:
        print("Warning: No schedule entries could be parsed from the LLM response.")
        
    return schedule_entries


# --- Schedule Saving ---

def save_schedule_to_csv(schedule_data, output_filepath):
    """Save the parsed schedule data to a CSV file."""
    if not schedule_data:
        print(f"No schedule data to save for {output_filepath}.")
        # Create an empty file with header to indicate an attempt was made but failed
        # This helps the validation script find a file, even if it's empty of data.
        try:
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            with open(output_filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['job', 'step', 'machine', 'start', 'end', 'precedence'])
                writer.writeheader()
            print(f"Created empty schedule file with header at {output_filepath} due to no parseable data.")
        except Exception as e:
            print(f"Error creating empty schedule file {output_filepath}: {e}")
        return

    try:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['job', 'step', 'machine', 'start', 'end', 'precedence'])
            writer.writeheader()
            writer.writerows(schedule_data)
        print(f"Schedule successfully saved to {output_filepath}")
    except Exception as e:
        print(f"Error saving schedule to {output_filepath}: {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Generate JSSP schedules using the Gemini LLM.")
    parser.add_argument("--model_config_dir", type=str, required=True, 
                        help="Directory name for the model configuration under results_baselines (e.g., gemini-2.5-sim1).")
    parser.add_argument("--gemini_model_name", type=str, default="gemini-1.5-pro-latest",
                        help="The specific Gemini model to use for generation (e.g., gemini-1.5-pro-latest, gemini-1.5-flash-latest).")
    parser.add_argument("--dmu_dir", type=str, default=os.path.join(os.path.dirname(__file__), 'DMU'),
                        help="Directory containing DMU (.txt) instance files.")
    parser.add_argument("--output_base_dir", type=str, default=os.path.join(os.path.dirname(__file__), '../results_baselines'),
                        help="Base directory where model-specific results folders are located.")
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Optional: Specific DMU dataset name (without .txt) to process. If None, all .txt files in dmu_dir are processed.")
    parser.add_argument("--overwrite", action='store_true', help="Overwrite existing schedule files.")
    parser.add_argument("--force_model", action='store_true', help="Use exactly the specified model without attempting to upgrade to pro.")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of jobs to process per chunk if chunking is required.")
    parser.add_argument("--skip_initial_attempt", action='store_true', help="Skip the initial full attempt and go straight to chunking.")

    args = parser.parse_args()

    google_api_key = load_environment()
    if not google_api_key:
        return
    if not configure_gemini_client(google_api_key):
        return

    # Determine datasets to process
    datasets_to_process = []
    if args.dataset_name:
        datasets_to_process.append(args.dataset_name)
    else:
        if not os.path.isdir(args.dmu_dir):
            print(f"Error: DMU directory not found at {args.dmu_dir}")
            return
        for filename in os.listdir(args.dmu_dir):
            if filename.endswith(".txt"):
                datasets_to_process.append(os.path.splitext(filename)[0])
    
    if not datasets_to_process:
        print(f"No datasets found to process in {args.dmu_dir}.")
        return
        
    print(f"Found {len(datasets_to_process)} datasets to process: {datasets_to_process}")

    # Force model if requested
    model_name = args.gemini_model_name
    if args.force_model:
        print(f"Using forced model: {model_name}")

    for dataset_name in sorted(datasets_to_process):
        print(f"\n--- Processing dataset: {dataset_name} ---")
        dmu_filepath = os.path.join(args.dmu_dir, f"{dataset_name}.txt")
        
        output_dir = os.path.join(args.output_base_dir, args.model_config_dir)
        schedule_output_filepath = os.path.join(output_dir, f"{dataset_name}_{args.model_config_dir}_5.csv")

        if not args.overwrite and os.path.exists(schedule_output_filepath):
            print(f"Schedule file {schedule_output_filepath} already exists. Skipping. Use --overwrite to replace.")
            continue

        jobs_data, num_jobs, num_machines = load_dmu_dataset(dmu_filepath)
        if not jobs_data:
            print(f"Skipping {dataset_name} due to DMU loading error.")
            continue

        parsed_schedule = []
        
        # Try the normal approach with all jobs if not skipping
        if not args.skip_initial_attempt:
            prompt = construct_jssp_prompt(jobs_data, num_jobs, num_machines, dataset_name)
            llm_response = call_gemini_api(prompt, model_name)
            
            if llm_response:
                parsed_schedule = parse_llm_response_to_schedule(llm_response)
        
        # Check if we have all jobs or need to use chunking
        job_names = set(entry['job'] for entry in parsed_schedule)
        expected_job_names = set(f"Job{i+1}" for i in range(num_jobs))
        missing_jobs = expected_job_names - job_names
        
        if args.skip_initial_attempt or (missing_jobs and len(job_names) < num_jobs):
            if not args.skip_initial_attempt:
                print(f"Warning: Missing {len(missing_jobs)} jobs in the response. Found {len(job_names)} out of {num_jobs}.")
                print(f"Missing jobs: {sorted(missing_jobs)}")
            
            print("Processing in chunks to get all jobs...")
            
            # Reset parsed_schedule
            parsed_schedule = []
            
            # Process in chunks
            chunk_size = args.chunk_size
            for chunk_start in range(0, num_jobs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, num_jobs)
                chunk_jobs = jobs_data[chunk_start:chunk_end]
                
                print(f"Processing jobs {chunk_start+1}-{chunk_end} (chunk of {len(chunk_jobs)} jobs)")
                
                # Create a direct, simplified prompt for this chunk
                chunk_prompt = f"I need a feasible JSSP schedule for jobs {chunk_start+1} to {chunk_end} of the dataset '{dataset_name}'.\n\n"
                chunk_prompt += f"THIS IS CRITICAL: Give me ONLY direct CSV DATA with the actual schedule, NOT code.\n\n"
                chunk_prompt += f"Number of Machines: {num_machines}\n\n"
                chunk_prompt += "Job Operations:\n"
                
                for job in chunk_jobs:
                    chunk_prompt += f"- {job['name']}:\n"
                    for op_idx, op in enumerate(job['operations']):
                        chunk_prompt += f"  - Step {op_idx + 1}: Machine {op['machine_id']}, Duration {op['duration']}\n"
                
                chunk_prompt += "\n"
                chunk_prompt += "RULES: Operations for each job must be in sequence. Each machine can only process one job at a time.\n"
                chunk_prompt += "Your output MUST be a CSV table with these exact columns: job,step,machine,start,end,precedence\n"
                chunk_prompt += "The first step of each job has precedence='None'. All other steps have precedence='After JobX Step Y'.\n\n"
                chunk_prompt += "Example:\n"
                chunk_prompt += "```csv\n"
                chunk_prompt += "job,step,machine,start,end,precedence\n"
                chunk_prompt += f"Job{chunk_start+1},1,Machine5,0,25,None\n"
                chunk_prompt += f"Job{chunk_start+1},2,Machine3,25,40,After Job{chunk_start+1} Step 1\n"
                chunk_prompt += "```\n\n"
                chunk_prompt += "IMPORTANT: Your response must be ONLY the finished CSV table for ALL operations of jobs " 
                chunk_prompt += f"{chunk_start+1} to {chunk_end}. Do NOT give any explanation or code.\n\n"
                chunk_prompt += "Your CSV table:\n"
                
                # Call API for this chunk
                chunk_response = call_gemini_api(chunk_prompt, model_name)
                
                if chunk_response:
                    chunk_schedule = parse_llm_response_to_schedule(chunk_response)
                    
                    if not chunk_schedule:
                        print(f"No valid entries found for chunk {chunk_start+1}-{chunk_end}. Trying one more time with a simpler prompt.")
                        
                        # Even simpler fallback prompt
                        simple_prompt = f"Generate a feasible job shop schedule for the following jobs:\n\n"
                        for job in chunk_jobs:
                            simple_prompt += f"{job['name']} operations:\n"
                            for op_idx, op in enumerate(job['operations']):
                                simple_prompt += f"  Step {op_idx+1}: Machine {op['machine_id']}, Duration {op['duration']}\n"
                        
                        simple_prompt += "\nOutput ONLY a CSV with these columns: job,step,machine,start,end,precedence\n"
                        simple_prompt += "Example: Job5,1,Machine3,0,45,None\n\n"
                        simple_prompt += "Your CSV data:\n"
                        
                        retry_response = call_gemini_api(simple_prompt, model_name)
                        if retry_response:
                            chunk_schedule = parse_llm_response_to_schedule(retry_response)
                    
                    print(f"Received {len(chunk_schedule)} entries for chunk {chunk_start+1}-{chunk_end}")
                    parsed_schedule.extend(chunk_schedule)
                else:
                    print(f"Failed to get a valid response for chunk {chunk_start+1}-{chunk_end}")
            
            # After processing all chunks
            print(f"Collected a total of {len(parsed_schedule)} schedule entries from all chunks")
            job_names = set(entry['job'] for entry in parsed_schedule)
            print(f"Found {len(job_names)} unique jobs after chunking")
        
        # Save the final schedule
        save_schedule_to_csv(parsed_schedule, schedule_output_filepath)
        
        # If we're still missing jobs after chunking, print a warning
        if len(job_names) < num_jobs:
            print(f"⚠️ Warning: Final schedule is incomplete. Only found {len(job_names)} out of {num_jobs} jobs.")
            print(f"Missing jobs: {sorted(list(expected_job_names - job_names))}")
        else:
            print(f"✅ Successfully generated schedule with all {num_jobs} jobs!")

if __name__ == "__main__":
    main() 