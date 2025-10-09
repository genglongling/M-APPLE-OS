#!/usr/bin/env python3
"""
Enhanced MAPLE with competitive optimization methods
Replaces LRCP with effective optimization algorithms
"""

from src.multi_agent.MAPLE import MAPLE
from src.multi_agent.MAPLE_optimization_methods import (
    create_optimization_method, 
    compare_optimization_methods
)
import logging

class EnhancedMAPLE(MAPLE):
    """Enhanced MAPLE with competitive optimization methods"""
    
    def __init__(self, task_spec, optimization_method: str = 'memetic_algorithm', **optimization_kwargs):
        super().__init__(task_spec)
        self.optimization_method = optimization_method
        self.optimization_kwargs = optimization_kwargs
        self.optimizer = None
        self.logger = logging.getLogger(__name__)
    
    def run(self, with_rollback=True, validate=True, use_optimization=True):
        """Enhanced run method with optimization"""
        self.logger.info(f"🚀 Starting Enhanced MAPLE with {self.optimization_method}")
        
        # Initial execution (same as original MAPLE)
        self.executor.execute(with_rollback=with_rollback, adaptation_manager=self.adaptation_manager)
        
        if validate:
            self.executor.self_validate()
        
        # Get initial schedule and makespan
        initial_schedule = self._extract_schedule()
        initial_makespan = self._calculate_makespan(initial_schedule)
        
        self.logger.info(f"📊 Initial makespan: {initial_makespan}")
        
        if use_optimization and initial_schedule:
            # Apply optimization
            optimized_schedule, optimized_makespan = self._apply_optimization(
                initial_schedule, 
                self.task_spec.get('jobs', []),
                self.task_spec.get('machines', [])
            )
            
            if optimized_makespan < initial_makespan:
                self.logger.info(f"✅ Optimization improved makespan: {initial_makespan} → {optimized_makespan}")
                self._update_schedule(optimized_schedule)
            else:
                self.logger.info(f"⚠️ Optimization did not improve makespan: {initial_makespan}")
        
        return self
    
    def _extract_schedule(self) -> list:
        """Extract schedule from executor context"""
        schedule = []
        for agent in self.executor.agents:
            if hasattr(agent, 'context') and isinstance(agent.context, dict):
                if 'schedule' in agent.context:
                    schedule.extend(agent.context['schedule'])
        return schedule
    
    def _calculate_makespan(self, schedule: list) -> float:
        """Calculate makespan from schedule"""
        if not schedule:
            return float('inf')
        return max(entry.get('end', 0) for entry in schedule)
    
    def _apply_optimization(self, initial_schedule: list, jobs: list, machines: list) -> tuple:
        """Apply optimization method to improve schedule"""
        try:
            if not self.optimizer:
                self.optimizer = create_optimization_method(
                    self.optimization_method, 
                    **self.optimization_kwargs
                )
            
            optimized_schedule, optimized_makespan = self.optimizer.optimize(
                initial_schedule, jobs, machines
            )
            
            return optimized_schedule, optimized_makespan
            
        except Exception as e:
            self.logger.error(f"❌ Optimization failed: {e}")
            return initial_schedule, self._calculate_makespan(initial_schedule)
    
    def _update_schedule(self, optimized_schedule: list):
        """Update executor context with optimized schedule"""
        # Find supervisor agent and update its context
        for agent in self.executor.agents:
            if 'Supervisor' in agent.name:
                agent.context = {'schedule': optimized_schedule}
                break
    
    def compare_optimization_methods(self, jobs: list, machines: list) -> dict:
        """Compare different optimization methods"""
        initial_schedule = self._extract_schedule()
        if not initial_schedule:
            self.logger.warning("No initial schedule found for comparison")
            return {}
        
        self.logger.info("🔬 Comparing optimization methods...")
        results = compare_optimization_methods(initial_schedule, jobs, machines)
        
        # Log results
        for method, (schedule, makespan) in results.items():
            if schedule:
                improvement = ((self._calculate_makespan(initial_schedule) - makespan) / 
                              self._calculate_makespan(initial_schedule)) * 100
                self.logger.info(f"{method}: {makespan} ({improvement:.1f}% improvement)")
            else:
                self.logger.info(f"{method}: Failed")
        
        return results

class MAPLEOptimizationComparison:
    """Compare different optimization methods for MAPLE"""
    
    def __init__(self, task_spec):
        self.task_spec = task_spec
        self.logger = logging.getLogger(__name__)
    
    def run_comparison(self, datasets: list = None) -> dict:
        """Run comparison across multiple datasets"""
        if datasets is None:
            datasets = ['rcmax_20_15_5', 'TA01', 'abz07', 'swv01', 'yn01']
        
        results = {}
        
        for dataset in datasets:
            self.logger.info(f"🔬 Comparing optimization methods on {dataset}")
            
            # Create enhanced MAPLE instance
            maple = EnhancedMAPLE(self.task_spec)
            
            # Run initial MAPLE
            maple.run(use_optimization=False)
            
            # Compare optimization methods
            comparison_results = maple.compare_optimization_methods(
                self.task_spec.get('jobs', []),
                self.task_spec.get('machines', [])
            )
            
            results[dataset] = comparison_results
        
        return results

# Example usage and testing
def test_optimization_methods():
    """Test different optimization methods"""
    # Sample task specification
    task_spec = {
        'jobs': [
            {'name': 'Job1', 'steps': [('Machine0', 10), ('Machine1', 20)]},
            {'name': 'Job2', 'steps': [('Machine1', 15), ('Machine2', 25)]}
        ],
        'machines': ['Machine0', 'Machine1', 'Machine2']
    }
    
    # Test different optimization methods
    methods = [
        'simulated_annealing',
        'genetic_algorithm', 
        'tabu_search',
        'variable_neighborhood_search',
        'memetic_algorithm'
    ]
    
    results = {}
    
    for method in methods:
        print(f"\n🧪 Testing {method}...")
        try:
            maple = EnhancedMAPLE(task_spec, optimization_method=method)
            maple.run()
            results[method] = "Success"
        except Exception as e:
            results[method] = f"Error: {e}"
    
    print("\n📊 Results Summary:")
    for method, result in results.items():
        print(f"{method}: {result}")

if __name__ == "__main__":
    test_optimization_methods()
