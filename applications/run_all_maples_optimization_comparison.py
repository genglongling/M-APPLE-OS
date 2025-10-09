#!/usr/bin/env python3
"""
MAPLE Optimization Methods Comparison
Compare different optimization algorithms for JSSP
"""

import sys
import os
import logging
import time
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_agent.MAPLE_enhanced import EnhancedMAPLE, MAPLEOptimizationComparison
from src.multi_agent.MAPLE_optimization_methods import create_optimization_method

def setup_logging():
    """Setup logging for optimization comparison"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimization_comparison.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(dataset_name: str) -> dict:
    """Load dataset specification"""
    # This would load actual dataset - for now using sample
    sample_datasets = {
        'rcmax_20_15_5': {
            'jobs': [{'name': f'Job{i}', 'steps': [('Machine{j}', 10+j) for j in range(15)]} for i in range(20)],
            'machines': [f'Machine{i}' for i in range(15)]
        },
        'TA01': {
            'jobs': [{'name': f'Job{i}', 'steps': [('Machine{j}', 5+j) for j in range(15)]} for i in range(15)],
            'machines': [f'Machine{i}' for i in range(15)]
        },
        'abz07': {
            'jobs': [{'name': f'Job{i}', 'steps': [('Machine{j}', 8+j) for j in range(15)]} for i in range(20)],
            'machines': [f'Machine{i}' for i in range(15)]
        }
    }
    
    return sample_datasets.get(dataset_name, sample_datasets['rcmax_20_15_5'])

def run_single_optimization_test(dataset_name: str, method: str) -> Dict:
    """Run single optimization method test"""
    logger = logging.getLogger(__name__)
    
    try:
        # Load dataset
        task_spec = load_dataset(dataset_name)
        
        # Create enhanced MAPLE
        maple = EnhancedMAPLE(task_spec, optimization_method=method)
        
        # Run without optimization first
        start_time = time.time()
        maple.run(use_optimization=False)
        initial_time = time.time() - start_time
        initial_makespan = maple._calculate_makespan(maple._extract_schedule())
        
        # Run with optimization
        start_time = time.time()
        maple.run(use_optimization=True)
        optimized_time = time.time() - start_time
        optimized_makespan = maple._calculate_makespan(maple._extract_schedule())
        
        # Calculate improvement
        improvement = ((initial_makespan - optimized_makespan) / initial_makespan) * 100 if initial_makespan > 0 else 0
        
        result = {
            'dataset': dataset_name,
            'method': method,
            'initial_makespan': initial_makespan,
            'optimized_makespan': optimized_makespan,
            'improvement_percent': improvement,
            'initial_time': initial_time,
            'optimized_time': optimized_time,
            'status': 'success'
        }
        
        logger.info(f"✅ {method} on {dataset_name}: {initial_makespan} → {optimized_makespan} ({improvement:.1f}% improvement)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ {method} on {dataset_name}: Error - {e}")
        return {
            'dataset': dataset_name,
            'method': method,
            'status': 'error',
            'error': str(e)
        }

def run_comprehensive_comparison():
    """Run comprehensive comparison of optimization methods"""
    logger = setup_logging()
    
    # Test datasets
    datasets = ['rcmax_20_15_5', 'TA01', 'abz07']
    
    # Optimization methods to compare
    methods = [
        'simulated_annealing',
        'genetic_algorithm',
        'tabu_search', 
        'variable_neighborhood_search',
        'memetic_algorithm',
        'lrcp'  # Add LRCP for comparison
    ]
    
    logger.info("🚀 Starting comprehensive optimization comparison")
    logger.info(f"📊 Testing {len(datasets)} datasets with {len(methods)} methods")
    
    results = []
    
    for dataset in datasets:
        logger.info(f"\n🔬 Testing dataset: {dataset}")
        
        for method in methods:
            logger.info(f"  🧪 Testing method: {method}")
            result = run_single_optimization_test(dataset, method)
            results.append(result)
    
    # Analyze results
    analyze_results(results)
    
    return results

def analyze_results(results: List[Dict]):
    """Analyze and summarize results"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n📊 OPTIMIZATION COMPARISON RESULTS")
    logger.info("=" * 60)
    
    # Group by method
    method_stats = {}
    for result in results:
        if result['status'] == 'success':
            method = result['method']
            if method not in method_stats:
                method_stats[method] = {
                    'improvements': [],
                    'times': [],
                    'datasets': []
                }
            
            method_stats[method]['improvements'].append(result['improvement_percent'])
            method_stats[method]['times'].append(result['optimized_time'])
            method_stats[method]['datasets'].append(result['dataset'])
    
    # Calculate statistics for each method
    for method, stats in method_stats.items():
        if stats['improvements']:
            avg_improvement = sum(stats['improvements']) / len(stats['improvements'])
            avg_time = sum(stats['times']) / len(stats['times'])
            success_rate = len(stats['datasets']) / len(set(r['dataset'] for r in results))
            
            logger.info(f"\n🎯 {method.upper()}:")
            logger.info(f"  Average Improvement: {avg_improvement:.1f}%")
            logger.info(f"  Average Time: {avg_time:.2f}s")
            logger.info(f"  Success Rate: {success_rate:.1%}")
            logger.info(f"  Datasets: {', '.join(stats['datasets'])}")
    
    # Find best method
    best_method = None
    best_improvement = -float('inf')
    
    for method, stats in method_stats.items():
        if stats['improvements']:
            avg_improvement = sum(stats['improvements']) / len(stats['improvements'])
            if avg_improvement > best_improvement:
                best_improvement = avg_improvement
                best_method = method
    
    if best_method:
        logger.info(f"\n🏆 BEST METHOD: {best_method.upper()}")
        logger.info(f"   Average Improvement: {best_improvement:.1f}%")
    
    # Method ranking
    logger.info(f"\n📈 METHOD RANKING (by average improvement):")
    ranked_methods = sorted(
        [(method, sum(stats['improvements'])/len(stats['improvements'])) 
         for method, stats in method_stats.items() if stats['improvements']],
        key=lambda x: x[1], reverse=True
    )
    
    for i, (method, improvement) in enumerate(ranked_methods, 1):
        logger.info(f"  {i}. {method}: {improvement:.1f}% improvement")

def run_quick_test():
    """Run quick test with single dataset and method"""
    logger = setup_logging()
    
    logger.info("🧪 Running quick optimization test...")
    
    # Test with memetic algorithm on rcmax_20_15_5
    result = run_single_optimization_test('rcmax_20_15_5', 'memetic_algorithm')
    
    if result['status'] == 'success':
        logger.info(f"✅ Quick test successful!")
        logger.info(f"   Initial makespan: {result['initial_makespan']}")
        logger.info(f"   Optimized makespan: {result['optimized_makespan']}")
        logger.info(f"   Improvement: {result['improvement_percent']:.1f}%")
    else:
        logger.error(f"❌ Quick test failed: {result.get('error', 'Unknown error')}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MAPLE Optimization Comparison')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Test mode: quick (single test) or full (comprehensive)')
    parser.add_argument('--dataset', default='rcmax_20_15_5',
                       help='Dataset for quick test')
    parser.add_argument('--method', default='memetic_algorithm',
                       help='Optimization method for quick test')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        run_quick_test()
    else:
        run_comprehensive_comparison()

if __name__ == "__main__":
    main()
