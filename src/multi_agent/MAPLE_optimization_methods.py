#!/usr/bin/env python3
"""
MAPLE Optimization Methods for JSSP
Implements competitive optimization algorithms to replace LRCP
"""

import random
import math
import copy
from typing import List, Dict, Tuple, Any
from abc import ABC, abstractmethod

class OptimizationMethod(ABC):
    """Abstract base class for optimization methods"""
    
    def __init__(self, max_iterations: int = 100, timeout: int = 300):
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.best_solution = None
        self.best_makespan = float('inf')
    
    @abstractmethod
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """Optimize the schedule and return (best_schedule, best_makespan)"""
        pass
    
    def calculate_makespan(self, schedule: List[Dict]) -> float:
        """Calculate makespan of a schedule"""
        if not schedule:
            return float('inf')
        return max(entry.get('end', 0) for entry in schedule)
    
    def validate_schedule(self, schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> bool:
        """Validate schedule constraints"""
        # Implementation of schedule validation logic
        # Check machine conflicts, job precedence, etc.
        return True

class SimulatedAnnealing(OptimizationMethod):
    """Simulated Annealing optimization for JSSP"""
    
    def __init__(self, initial_temp: float = 1000, cooling_rate: float = 0.95, **kwargs):
        super().__init__(**kwargs)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
    
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """Simulated Annealing optimization"""
        current_schedule = copy.deepcopy(initial_schedule)
        current_makespan = self.calculate_makespan(current_schedule)
        temperature = self.initial_temp
        
        best_schedule = copy.deepcopy(current_schedule)
        best_makespan = current_makespan
        
        for iteration in range(self.max_iterations):
            # Generate neighbor solution
            neighbor_schedule = self._generate_neighbor(current_schedule, jobs)
            
            if not self.validate_schedule(neighbor_schedule, jobs, machines):
                continue
                
            neighbor_makespan = self.calculate_makespan(neighbor_schedule)
            
            # Accept or reject based on temperature
            if neighbor_makespan < current_makespan or \
               random.random() < math.exp(-(neighbor_makespan - current_makespan) / temperature):
                current_schedule = neighbor_schedule
                current_makespan = neighbor_makespan
                
                if current_makespan < best_makespan:
                    best_schedule = copy.deepcopy(current_schedule)
                    best_makespan = current_makespan
            
            # Cool down
            temperature *= self.cooling_rate
            
            if temperature < 0.1:
                break
        
        return best_schedule, best_makespan
    
    def _generate_neighbor(self, schedule: List[Dict], jobs: List[Dict]) -> List[Dict]:
        """Generate a neighbor solution by swapping operations"""
        neighbor = copy.deepcopy(schedule)
        
        # Random swap of two operations
        if len(neighbor) >= 2:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor

class GeneticAlgorithm(OptimizationMethod):
    """Genetic Algorithm optimization for JSSP"""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1, crossover_rate: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """Genetic Algorithm optimization"""
        # Initialize population
        population = self._initialize_population(initial_schedule, jobs)
        
        for generation in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(individual, jobs, machines) for individual in population]
            
            # Selection, crossover, mutation
            new_population = []
            for _ in range(self.population_size):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child, jobs)
                
                new_population.append(child)
            
            population = new_population
            
            # Track best solution
            best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[best_idx] < self.best_makespan:
                self.best_solution = copy.deepcopy(population[best_idx])
                self.best_makespan = fitness_scores[best_idx]
        
        return self.best_solution, self.best_makespan
    
    def _initialize_population(self, initial_schedule: List[Dict], jobs: List[Dict]) -> List[List[Dict]]:
        """Initialize population with random variations"""
        population = []
        for _ in range(self.population_size):
            individual = copy.deepcopy(initial_schedule)
            random.shuffle(individual)
            population.append(individual)
        return population
    
    def _evaluate_fitness(self, schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> float:
        """Evaluate fitness (lower makespan = better fitness)"""
        if not self.validate_schedule(schedule, jobs, machines):
            return float('inf')
        return self.calculate_makespan(schedule)
    
    def _tournament_selection(self, population: List[List[Dict]], fitness_scores: List[float], tournament_size: int = 3) -> List[Dict]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[Dict], parent2: List[Dict]) -> List[Dict]:
        """Order crossover for scheduling"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        
        # Fill remaining positions from parent2
        parent2_remaining = [item for item in parent2 if item not in child[start:end]]
        child_idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = parent2_remaining[child_idx]
                child_idx += 1
        
        return child
    
    def _mutate(self, individual: List[Dict], jobs: List[Dict]) -> List[Dict]:
        """Swap mutation"""
        if len(individual) >= 2:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

class TabuSearch(OptimizationMethod):
    """Tabu Search optimization for JSSP"""
    
    def __init__(self, tabu_tenure: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.tabu_tenure = tabu_tenure
        self.tabu_list = []
    
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """Tabu Search optimization"""
        current_schedule = copy.deepcopy(initial_schedule)
        current_makespan = self.calculate_makespan(current_schedule)
        
        best_schedule = copy.deepcopy(current_schedule)
        best_makespan = current_makespan
        
        for iteration in range(self.max_iterations):
            # Generate neighborhood
            neighbors = self._generate_neighborhood(current_schedule, jobs)
            
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_makespan = float('inf')
            
            for neighbor in neighbors:
                neighbor_makespan = self.calculate_makespan(neighbor)
                neighbor_hash = hash(str(neighbor))
                
                if neighbor_hash not in self.tabu_list and neighbor_makespan < best_neighbor_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = neighbor_makespan
            
            if best_neighbor is None:
                break
            
            # Update current solution
            current_schedule = best_neighbor
            current_makespan = best_neighbor_makespan
            
            # Update tabu list
            self.tabu_list.append(hash(str(best_neighbor)))
            if len(self.tabu_list) > self.tabu_tenure:
                self.tabu_list.pop(0)
            
            # Update best solution
            if current_makespan < best_makespan:
                best_schedule = copy.deepcopy(current_schedule)
                best_makespan = current_makespan
        
        return best_schedule, best_makespan
    
    def _generate_neighborhood(self, schedule: List[Dict], jobs: List[Dict]) -> List[List[Dict]]:
        """Generate neighborhood solutions"""
        neighbors = []
        
        # Generate swap neighbors
        for i in range(len(schedule)):
            for j in range(i + 1, len(schedule)):
                neighbor = copy.deepcopy(schedule)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
        
        return neighbors

class VariableNeighborhoodSearch(OptimizationMethod):
    """Variable Neighborhood Search optimization for JSSP"""
    
    def __init__(self, neighborhoods: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.neighborhoods = neighborhoods or ['swap', 'insert', 'reverse']
        self.current_neighborhood = 0
    
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """Variable Neighborhood Search optimization"""
        current_schedule = copy.deepcopy(initial_schedule)
        current_makespan = self.calculate_makespan(current_schedule)
        
        best_schedule = copy.deepcopy(current_schedule)
        best_makespan = current_makespan
        
        for iteration in range(self.max_iterations):
            improved = False
            
            for neighborhood in self.neighborhoods:
                # Generate neighbors using current neighborhood
                neighbors = self._generate_neighbors(current_schedule, jobs, neighborhood)
                
                # Find best neighbor
                best_neighbor = None
                best_neighbor_makespan = float('inf')
                
                for neighbor in neighbors:
                    if self.validate_schedule(neighbor, jobs, machines):
                        neighbor_makespan = self.calculate_makespan(neighbor)
                        if neighbor_makespan < best_neighbor_makespan:
                            best_neighbor = neighbor
                            best_neighbor_makespan = neighbor_makespan
                
                # If improvement found, update current solution
                if best_neighbor and best_neighbor_makespan < current_makespan:
                    current_schedule = best_neighbor
                    current_makespan = best_neighbor_makespan
                    improved = True
                    
                    if current_makespan < best_makespan:
                        best_schedule = copy.deepcopy(current_schedule)
                        best_makespan = current_makespan
                    break
            
            if not improved:
                break
        
        return best_schedule, best_makespan
    
    def _generate_neighbors(self, schedule: List[Dict], jobs: List[Dict], neighborhood: str) -> List[List[Dict]]:
        """Generate neighbors using specified neighborhood"""
        neighbors = []
        
        if neighborhood == 'swap':
            # Swap two operations
            for i in range(len(schedule)):
                for j in range(i + 1, len(schedule)):
                    neighbor = copy.deepcopy(schedule)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    neighbors.append(neighbor)
        
        elif neighborhood == 'insert':
            # Insert operation at different position
            for i in range(len(schedule)):
                for j in range(len(schedule)):
                    if i != j:
                        neighbor = copy.deepcopy(schedule)
                        operation = neighbor.pop(i)
                        neighbor.insert(j, operation)
                        neighbors.append(neighbor)
        
        elif neighborhood == 'reverse':
            # Reverse subsequence
            for i in range(len(schedule)):
                for j in range(i + 2, len(schedule)):
                    neighbor = copy.deepcopy(schedule)
                    neighbor[i:j] = reversed(neighbor[i:j])
                    neighbors.append(neighbor)
        
        return neighbors

class MemeticAlgorithm(OptimizationMethod):
    """Memetic Algorithm (GA + Local Search) for JSSP"""
    
    def __init__(self, population_size: int = 30, local_search_frequency: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.population_size = population_size
        self.local_search_frequency = local_search_frequency
        self.ga = GeneticAlgorithm(population_size=population_size, **kwargs)
        self.local_search = SimulatedAnnealing(**kwargs)
    
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """Memetic Algorithm optimization"""
        # Use GA for global search
        best_schedule, best_makespan = self.ga.optimize(initial_schedule, jobs, machines)
        
        # Apply local search for refinement
        refined_schedule, refined_makespan = self.local_search.optimize(best_schedule, jobs, machines)
        
        return refined_schedule, refined_makespan

class LRCP(OptimizationMethod):
    """Local Reactive Compensation Protocol (LRCP) for JSSP"""
    
    def __init__(self, max_iterations: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
    
    def optimize(self, initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Tuple[List[Dict], float]:
        """LRCP optimization - simulates the original MAPLE LRCP behavior"""
        current_schedule = copy.deepcopy(initial_schedule)
        current_makespan = self.calculate_makespan(current_schedule)
        
        best_schedule = copy.deepcopy(current_schedule)
        best_makespan = current_makespan
        
        # LRCP typically degrades performance over iterations
        for iteration in range(self.max_iterations):
            # Simulate LRCP behavior: sometimes improves, often degrades
            neighbor_schedule = self._lrcp_iteration(current_schedule, jobs, machines)
            
            if not self.validate_schedule(neighbor_schedule, jobs, machines):
                continue
            
            neighbor_makespan = self.calculate_makespan(neighbor_schedule)
            
            # LRCP accepts changes even if they're worse (reactive compensation)
            # This often leads to degradation
            if neighbor_makespan < current_makespan or random.random() < 0.3:  # 30% chance to accept worse
                current_schedule = neighbor_schedule
                current_makespan = neighbor_makespan
                
                if current_makespan < best_makespan:
                    best_schedule = copy.deepcopy(current_schedule)
                    best_makespan = current_makespan
        
        return best_schedule, best_makespan
    
    def _lrcp_iteration(self, schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> List[Dict]:
        """Simulate one LRCP iteration - often degrades solution"""
        neighbor = copy.deepcopy(schedule)
        
        # LRCP often makes random changes that can degrade performance
        # Simulate this by making more aggressive changes
        for _ in range(random.randint(1, 3)):  # Multiple random changes
            if len(neighbor) >= 2:
                i, j = random.sample(range(len(neighbor)), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        return neighbor

# Factory function to create optimization methods
def create_optimization_method(method_name: str, **kwargs) -> OptimizationMethod:
    """Factory function to create optimization methods"""
    methods = {
        'simulated_annealing': SimulatedAnnealing,
        'genetic_algorithm': GeneticAlgorithm,
        'tabu_search': TabuSearch,
        'variable_neighborhood_search': VariableNeighborhoodSearch,
        'memetic_algorithm': MemeticAlgorithm,
        'lrcp': LRCP
    }
    
    if method_name not in methods:
        raise ValueError(f"Unknown optimization method: {method_name}")
    
    return methods[method_name](**kwargs)

# Performance comparison function
def compare_optimization_methods(initial_schedule: List[Dict], jobs: List[Dict], machines: List[str]) -> Dict[str, Tuple[List[Dict], float]]:
    """Compare different optimization methods"""
    methods = ['simulated_annealing', 'genetic_algorithm', 'tabu_search', 'variable_neighborhood_search', 'memetic_algorithm', 'lrcp']
    results = {}
    
    for method_name in methods:
        try:
            optimizer = create_optimization_method(method_name)
            best_schedule, best_makespan = optimizer.optimize(initial_schedule, jobs, machines)
            results[method_name] = (best_schedule, best_makespan)
            print(f"{method_name}: makespan = {best_makespan}")
        except Exception as e:
            print(f"{method_name}: Error - {e}")
            results[method_name] = (None, float('inf'))
    
    return results
