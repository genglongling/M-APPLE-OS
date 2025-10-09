"""
Optimization Tools for MAPLE JSSP Workflow
Provides various optimization algorithms for schedule improvement
"""

import random
import copy
import math
from typing import Dict, List, Any, Tuple, Optional


class OptimizationTools:
    """
    Static methods for optimizing JSSP schedules
    """
    
    @staticmethod
    def run_optimization_schedule(schedule_data: Any) -> Any:
        """
        Run optimization on schedule data
        
        Args:
            schedule_data: Schedule data to optimize
            
        Returns:
            Optimized schedule data
        """
        try:
            # Parse schedule data if needed
            if isinstance(schedule_data, str):
                import json
                schedule_data = json.loads(schedule_data)
            
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
                return schedule_data
            
            # Apply optimization
            optimized_entries = OptimizationTools._apply_optimization(schedule_entries)
            
            # Return in original format
            if isinstance(schedule_data, dict):
                if 'schedule' in schedule_data:
                    schedule_data['schedule'] = optimized_entries
                elif 'schedules' in schedule_data:
                    schedule_data['schedules'] = optimized_entries
                else:
                    schedule_data = optimized_entries
                return schedule_data
            else:
                return optimized_entries
                
        except Exception as e:
            print(f"Optimization error: {str(e)}")
            return schedule_data
    
    @staticmethod
    def _apply_optimization(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply optimization algorithms to improve schedule"""
        # Try multiple optimization methods
        methods = [
            OptimizationTools._shortest_processing_time,
            OptimizationTools._longest_processing_time,
            OptimizationTools._random_improvement,
            OptimizationTools._local_search
        ]
        
        best_schedule = schedule_entries.copy()
        best_makespan = OptimizationTools._calculate_makespan(schedule_entries)
        
        for method in methods:
            try:
                optimized = method(schedule_entries.copy())
                makespan = OptimizationTools._calculate_makespan(optimized)
                
                if makespan < best_makespan:
                    best_schedule = optimized
                    best_makespan = makespan
            except Exception as e:
                print(f"Optimization method failed: {str(e)}")
                continue
        
        return best_schedule
    
    @staticmethod
    def _calculate_makespan(schedule_entries: List[Dict[str, Any]]) -> int:
        """Calculate makespan of schedule"""
        if not schedule_entries:
            return 0
        return max(entry.get('end', 0) for entry in schedule_entries)
    
    @staticmethod
    def _shortest_processing_time(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """SPT heuristic: prioritize shorter operations"""
        # Calculate processing times
        for entry in schedule_entries:
            entry['processing_time'] = entry.get('end', 0) - entry.get('start', 0)
        
        # Sort by processing time (shortest first)
        sorted_entries = sorted(schedule_entries, key=lambda x: x.get('processing_time', 0))
        
        # Reschedule with SPT priority
        return OptimizationTools._reschedule_with_priority(sorted_entries)
    
    @staticmethod
    def _longest_processing_time(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LPT heuristic: prioritize longer operations"""
        # Calculate processing times
        for entry in schedule_entries:
            entry['processing_time'] = entry.get('end', 0) - entry.get('start', 0)
        
        # Sort by processing time (longest first)
        sorted_entries = sorted(schedule_entries, key=lambda x: x.get('processing_time', 0), reverse=True)
        
        # Reschedule with LPT priority
        return OptimizationTools._reschedule_with_priority(sorted_entries)
    
    @staticmethod
    def _random_improvement(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Random improvement: try random swaps"""
        improved = schedule_entries.copy()
        
        # Try random swaps
        for _ in range(min(10, len(schedule_entries))):
            if len(improved) < 2:
                break
                
            # Random swap
            i, j = random.sample(range(len(improved)), 2)
            improved[i], improved[j] = improved[j], improved[i]
            
            # Check if improvement
            if OptimizationTools._calculate_makespan(improved) < OptimizationTools._calculate_makespan(schedule_entries):
                schedule_entries = improved.copy()
            else:
                # Revert
                improved[i], improved[j] = improved[j], improved[i]
        
        return improved
    
    @staticmethod
    def _local_search(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Local search optimization"""
        current = schedule_entries.copy()
        improved = True
        
        while improved:
            improved = False
            current_makespan = OptimizationTools._calculate_makespan(current)
            
            # Try all possible swaps
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    # Try swap
                    current[i], current[j] = current[j], current[i]
                    new_makespan = OptimizationTools._calculate_makespan(current)
                    
                    if new_makespan < current_makespan:
                        improved = True
                        current_makespan = new_makespan
                    else:
                        # Revert swap
                        current[i], current[j] = current[j], current[i]
        
        return current
    
    @staticmethod
    def _reschedule_with_priority(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reschedule entries with given priority"""
        # Group by machine
        machine_queues = {}
        for entry in schedule_entries:
            machine = entry.get('machine', '')
            if machine not in machine_queues:
                machine_queues[machine] = []
            machine_queues[machine].append(entry)
        
        # Reschedule each machine queue
        rescheduled = []
        for machine, queue in machine_queues.items():
            # Sort queue by priority (already sorted by the calling method)
            current_time = 0
            for entry in queue:
                processing_time = entry.get('processing_time', 5)
                entry['start'] = current_time
                entry['end'] = current_time + processing_time
                current_time += processing_time
                rescheduled.append(entry)
        
        return rescheduled
    
    @staticmethod
    def simulated_annealing(schedule_entries: List[Dict[str, Any]], 
                           initial_temp: float = 1000.0,
                           cooling_rate: float = 0.95,
                           min_temp: float = 1.0) -> List[Dict[str, Any]]:
        """Simulated Annealing optimization"""
        current = schedule_entries.copy()
        best = current.copy()
        current_makespan = OptimizationTools._calculate_makespan(current)
        best_makespan = current_makespan
        temperature = initial_temp
        
        while temperature > min_temp:
            # Generate neighbor
            neighbor = OptimizationTools._generate_neighbor(current)
            neighbor_makespan = OptimizationTools._calculate_makespan(neighbor)
            
            # Accept or reject
            if neighbor_makespan < current_makespan:
                # Always accept improvement
                current = neighbor
                current_makespan = neighbor_makespan
                
                if neighbor_makespan < best_makespan:
                    best = neighbor.copy()
                    best_makespan = neighbor_makespan
            else:
                # Accept with probability
                delta = neighbor_makespan - current_makespan
                probability = math.exp(-delta / temperature)
                
                if random.random() < probability:
                    current = neighbor
                    current_makespan = neighbor_makespan
            
            # Cool down
            temperature *= cooling_rate
        
        return best
    
    @staticmethod
    def _generate_neighbor(schedule_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate a neighbor solution"""
        neighbor = copy.deepcopy(schedule_entries)
        
        if len(neighbor) < 2:
            return neighbor
        
        # Random swap
        i, j = random.sample(range(len(neighbor)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor
    
    @staticmethod
    def genetic_algorithm(schedule_entries: List[Dict[str, Any]], 
                         population_size: int = 20,
                         generations: int = 50) -> List[Dict[str, Any]]:
        """Genetic Algorithm optimization"""
        if len(schedule_entries) < 2:
            return schedule_entries
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = copy.deepcopy(schedule_entries)
            random.shuffle(individual)
            population.append(individual)
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                makespan = OptimizationTools._calculate_makespan(individual)
                fitness_scores.append(1.0 / (1.0 + makespan))  # Higher fitness for lower makespan
            
            # Selection and reproduction
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                parent1 = OptimizationTools._tournament_selection(population, fitness_scores)
                parent2 = OptimizationTools._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = OptimizationTools._crossover(parent1, parent2)
                
                # Mutation
                if random.random() < 0.1:  # 10% mutation rate
                    child = OptimizationTools._mutate(child)
                
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        best_individual = min(population, key=lambda x: OptimizationTools._calculate_makespan(x))
        return best_individual
    
    @staticmethod
    def _tournament_selection(population: List[List[Dict]], fitness_scores: List[float], 
                            tournament_size: int = 3) -> List[Dict[str, Any]]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_index]
    
    @staticmethod
    def _crossover(parent1: List[Dict[str, Any]], parent2: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order crossover for JSSP"""
        if len(parent1) != len(parent2):
            return parent1
        
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        
        # Copy segment from parent1
        child[start:end] = parent1[start:end]
        
        # Fill remaining from parent2
        parent2_index = 0
        for i in range(size):
            if child[i] is None:
                while parent2[parent2_index] in child[start:end]:
                    parent2_index += 1
                child[i] = parent2[parent2_index]
                parent2_index += 1
        
        return child
    
    @staticmethod
    def _mutate(individual: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Swap mutation"""
        if len(individual) < 2:
            return individual
        
        mutated = copy.deepcopy(individual)
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
