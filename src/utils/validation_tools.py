"""
Validation Tools for MAPLE JSSP Workflow
Provides comprehensive schedule validation functionality
"""

import json
from typing import Dict, List, Any, Tuple


class ValidationTools:
    """
    Static methods for validating JSSP schedules
    """
    
    @staticmethod
    def validate_schedule(schedule_data: Any) -> Dict[str, Any]:
        """
        Comprehensive schedule validation
        
        Args:
            schedule_data: Schedule data to validate (can be dict, list, or string)
            
        Returns:
            Dict with 'valid' boolean and 'errors' list
        """
        errors = []
        
        try:
            # Parse schedule data if it's a string
            if isinstance(schedule_data, str):
                try:
                    schedule_data = json.loads(schedule_data)
                except json.JSONDecodeError:
                    errors.append("Invalid JSON format in schedule data")
                    return {'valid': False, 'errors': errors}
            
            # Check if schedule is a list/dict
            if not isinstance(schedule_data, (list, dict)):
                errors.append("Schedule must be a list or dictionary")
                return {'valid': False, 'errors': errors}
            
            # Extract schedule entries
            if isinstance(schedule_data, dict):
                if 'schedule' in schedule_data:
                    schedule_entries = schedule_data['schedule']
                elif 'schedules' in schedule_data:
                    schedule_entries = schedule_data['schedules']
                else:
                    schedule_entries = [schedule_data]
            else:
                schedule_entries = schedule_data
            
            if not schedule_entries:
                errors.append("No schedule entries found")
                return {'valid': False, 'errors': errors}
            
            # Validate each schedule entry
            for i, entry in enumerate(schedule_entries):
                entry_errors = ValidationTools._validate_schedule_entry(entry, i)
                errors.extend(entry_errors)
            
            # Check for makespan consistency
            makespan_errors = ValidationTools._validate_makespan(schedule_entries)
            errors.extend(makespan_errors)
            
            # Check critical JSSP constraints
            precedence_errors = ValidationTools.validate_job_precedence(schedule_entries)
            errors.extend(precedence_errors)
            
            machine_errors = ValidationTools.validate_machine_capacity(schedule_entries)
            errors.extend(machine_errors)
            
            # Check dataset constraints (if dataset info is available)
            if hasattr(ValidationTools, '_current_dataset_info'):
                dataset_errors = ValidationTools.validate_dataset_constraints(schedule_entries)
                errors.extend(dataset_errors)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'schedule_entries': len(schedule_entries)
            }
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return {'valid': False, 'errors': errors}
    
    @staticmethod
    def _validate_schedule_entry(entry: Dict[str, Any], index: int) -> List[str]:
        """Validate a single schedule entry"""
        errors = []
        
        # Check required fields
        required_fields = ['job', 'step', 'machine', 'start', 'end']
        for field in required_fields:
            if field not in entry:
                errors.append(f"Entry {index}: Missing required field '{field}'")
        
        # Check data types
        if 'start' in entry and not isinstance(entry['start'], (int, float)):
            errors.append(f"Entry {index}: 'start' must be numeric")
        
        if 'end' in entry and not isinstance(entry['end'], (int, float)):
            errors.append(f"Entry {index}: 'end' must be numeric")
        
        # Check time consistency
        if 'start' in entry and 'end' in entry:
            if entry['start'] >= entry['end']:
                errors.append(f"Entry {index}: start time ({entry['start']}) must be less than end time ({entry['end']})")
        
        return errors
    
    @staticmethod
    def _validate_makespan(schedule_entries: List[Dict[str, Any]]) -> List[str]:
        """Validate makespan consistency"""
        errors = []
        
        if not schedule_entries:
            return errors
        
        # Calculate makespan from schedule
        max_end_time = max(entry.get('end', 0) for entry in schedule_entries)
        
        # Check if makespan is reasonable (not too small or too large)
        if max_end_time <= 0:
            errors.append("Invalid makespan: must be greater than 0")
        elif max_end_time > 100000:  # Arbitrary upper bound
            errors.append(f"Unrealistic makespan: {max_end_time} (too large)")
        
        return errors
    
    @staticmethod
    def validate_job_precedence(schedule_entries: List[Dict[str, Any]]) -> List[str]:
        """Validate job precedence constraints"""
        errors = []
        
        # Group entries by job
        job_entries = {}
        for entry in schedule_entries:
            job = entry.get('job', '')
            if job not in job_entries:
                job_entries[job] = []
            job_entries[job].append(entry)
        
        # Check precedence for each job
        for job, entries in job_entries.items():
            # Sort by step number
            try:
                sorted_entries = sorted(entries, key=lambda x: int(x.get('step', 0)))
                
                for i in range(1, len(sorted_entries)):
                    prev_entry = sorted_entries[i-1]
                    curr_entry = sorted_entries[i]
                    
                    # Check if current step starts after previous step ends
                    if prev_entry.get('end', 0) > curr_entry.get('start', 0):
                        errors.append(f"Job {job}: Step {curr_entry.get('step')} starts before step {prev_entry.get('step')} ends")
                        
            except (ValueError, TypeError) as e:
                errors.append(f"Job {job}: Error sorting steps - {str(e)}")
        
        return errors
    
    @staticmethod
    def validate_machine_capacity(schedule_entries: List[Dict[str, Any]]) -> List[str]:
        """Validate machine capacity constraints"""
        errors = []
        
        # Group entries by machine
        machine_entries = {}
        for entry in schedule_entries:
            machine = entry.get('machine', '')
            if machine not in machine_entries:
                machine_entries[machine] = []
            machine_entries[machine].append(entry)
        
        # Check for overlapping operations on same machine
        for machine, entries in machine_entries.items():
            # Sort by start time
            sorted_entries = sorted(entries, key=lambda x: x.get('start', 0))
            
            for i in range(1, len(sorted_entries)):
                prev_entry = sorted_entries[i-1]
                curr_entry = sorted_entries[i]
                
                # Check for overlap
                if (prev_entry.get('end', 0) > curr_entry.get('start', 0)):
                    errors.append(f"Machine {machine}: Operations overlap - {prev_entry.get('job', '')} step {prev_entry.get('step', '')} and {curr_entry.get('job', '')} step {curr_entry.get('step', '')}")
        
        return errors
    
    @staticmethod
    def validate_dataset_constraints(schedule_entries: List[Dict[str, Any]]) -> List[str]:
        """Validate that schedule matches dataset requirements"""
        errors = []
        
        if not hasattr(ValidationTools, '_current_dataset_info'):
            return errors
        
        dataset_info = ValidationTools._current_dataset_info
        
        # Check if all required jobs are present
        if 'jobs' in dataset_info:
            required_jobs = set(job['name'] for job in dataset_info['jobs'])
            schedule_jobs = set(entry.get('job', '') for entry in schedule_entries)
            
            missing_jobs = required_jobs - schedule_jobs
            if missing_jobs:
                errors.append(f"Missing jobs in schedule: {list(missing_jobs)}")
            
            extra_jobs = schedule_jobs - required_jobs
            if extra_jobs:
                errors.append(f"Extra jobs in schedule: {list(extra_jobs)}")
        
        # Check if all required machines are used
        if 'machine_names' in dataset_info:
            required_machines = set(dataset_info['machine_names'])
            schedule_machines = set(entry.get('machine', '') for entry in schedule_entries)
            
            missing_machines = required_machines - schedule_machines
            if missing_machines:
                errors.append(f"Unused machines: {list(missing_machines)}")
        
        return errors
    
    @staticmethod
    def set_dataset_info(dataset_info: Dict[str, Any]):
        """Set current dataset information for validation"""
        ValidationTools._current_dataset_info = dataset_info
    
    @staticmethod
    def get_upper_bounds() -> Dict[str, int]:
        """
        Get theoretical upper bounds for different datasets
        """
        return {
            # DMU datasets
            'rcmax_20_15_5': 2731,
            'rcmax_20_15_8': 2669,
            'rcmax_20_20_7': 3188,
            'rcmax_20_20_8': 3092,
            'rcmax_30_15_5': 3681,
            'rcmax_30_15_4': 3394,
            'rcmax_30_20_9': 3844,
            'rcmax_30_20_8': 3764,
            'rcmax_40_15_10': 4668,
            'rcmax_40_15_8': 4648,
            'rcmax_40_20_6': 4692,
            'rcmax_40_20_2': 4691,
            'rcmax_50_15_2': 5385,
            'rcmax_50_15_4': 5385,
            'rcmax_50_20_6': 5713,
            'rcmax_50_20_9': 5747,
            
            # TA datasets
            'TA01': 1231,
            'TA02': 1039,
            'TA51': 1234,
            'TA52': 1000,
            'TA61': 1231,
            'TA71': 1659,
            'TA72': 1247,
            
            # ABZ datasets
            'abz07': 656,
            'abz08': 665,
            'abz09': 679,
            
            # SWV datasets
            'swv01': 1397,
            'swv02': 1250,
            'swv03': 1268,
            'swv04': 616,
            'swv05': 1294,
            'swv06': 1268,
            'swv07': 1478,
            'swv08': 1500,
            'swv09': 1659,
            'swv10': 1234,
            'swv11': 1435,
            'swv12': 1794,
            'swv13': 1547,
            'swv14': 1000,
            'swv15': 1000,
            
            # YN datasets
            'yn01': 1165,
            'yn02': 1000,
            'yn03': 892,
            'yn04': 1165
        }
    
    @staticmethod
    def validate_against_upper_bound(schedule_entries: List[Dict[str, Any]], dataset_name: str) -> List[str]:
        """Validate schedule against theoretical upper bound"""
        errors = []
        
        if not schedule_entries:
            return errors
        
        # Calculate makespan
        makespan = max(entry.get('end', 0) for entry in schedule_entries)
        
        # Get upper bound for dataset
        upper_bounds = ValidationTools.get_upper_bounds()
        ub = upper_bounds.get(dataset_name)
        
        if ub is not None:
            # Check if makespan is not smaller than upper bound (UB is optimal)
            if makespan < ub:
                errors.append(f"Makespan {makespan} is below theoretical upper bound {ub} - this is unrealistic")
            elif makespan > ub * 2:  # Allow some tolerance
                errors.append(f"Makespan {makespan} is significantly above upper bound {ub}")
        
        return errors
