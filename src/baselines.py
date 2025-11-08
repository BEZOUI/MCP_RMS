"""
Baseline comparison methods
Includes dispatching rules, GA, SA, and DRL methods
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from copy import deepcopy
import random
from deap import base, creator, tools, algorithms
import logging

logger = logging.getLogger(__name__)


class DispatchingRules:
    """Traditional dispatching rule heuristics"""

    def __init__(self, env):
        self.env = env

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _reset(self):
        """Reset environment before running a heuristic"""
        self.env.reset()

    @staticmethod
    def _can_schedule(job, operation) -> bool:
        """Check precedence feasibility for an operation"""
        return all(
            hasattr(job.operations[p], 'completed') and job.operations[p].completed
            for p in operation.precedence
        )

    def _assign_operation(self, job, operation) -> bool:
        """Try assigning the operation to the earliest available machine"""
        compatible_machines = [m for m in self.env.machines if m.can_process(operation)]

        if not compatible_machines:
            logger.warning(
                "No compatible machines for job %s op %s (requires %s)",
                job.job_id,
                operation.op_id,
                operation.required_capability,
            )
            return False

        compatible_machines.sort(key=lambda m: m.next_available_time)

        for machine in compatible_machines:
            start_time = max(self.env.current_time, machine.next_available_time)
            result = self.env.assign_operation(
                job.job_id,
                operation.op_id,
                machine.machine_id,
                start_time,
            )
            if result['success']:
                return True

        return False

    def _select_job(self, priority_fn: Callable) -> bool:
        """Generic single-operation scheduling loop"""
        candidates = []

        for job in self.env.pending_jobs:
            for operation in job.operations:
                if hasattr(operation, 'completed') and operation.completed:
                    continue

                if not self._can_schedule(job, operation):
                    continue

                priority = priority_fn(job, operation)
                candidates.append((priority, job, operation))
                break

        if not candidates:
            return False

        # Sort by priority (lower is better)
        candidates.sort(key=lambda x: x[0])

        for _, job, operation in candidates:
            if self._assign_operation(job, operation):
                return True

        return False

    # ------------------------------------------------------------------
    # Classic heuristics
    # ------------------------------------------------------------------
    def fifo(self) -> Dict:
        """First In First Out"""
        logger.info("Running FIFO dispatching rule")
        self._reset()

        def priority(job, _operation):
            return job.arrival_time

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def spt(self) -> Dict:
        """Shortest Processing Time"""
        logger.info("Running SPT dispatching rule")
        self._reset()

        def priority(_job, operation):
            return operation.nominal_processing_time

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def edd(self) -> Dict:
        """Earliest Due Date"""
        logger.info("Running EDD dispatching rule")
        self._reset()

        def priority(job, _operation):
            return job.due_date

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def mwkr(self) -> Dict:
        """Most Work Remaining"""
        logger.info("Running MWKR dispatching rule")
        self._reset()

        def priority(job, _operation):
            remaining_time = sum(
                op.nominal_processing_time
                for op in job.operations
                if not (hasattr(op, 'completed') and op.completed)
            )
            return -remaining_time  # negate for descending ordering

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    # ------------------------------------------------------------------
    # Extended heuristics to reach 10+ baseline methods
    # ------------------------------------------------------------------
    def lpt(self) -> Dict:
        """Longest Processing Time"""
        logger.info("Running LPT dispatching rule")
        self._reset()

        def priority(_job, operation):
            return -operation.nominal_processing_time

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def lwkr(self) -> Dict:
        """Least Work Remaining"""
        logger.info("Running LWKR dispatching rule")
        self._reset()

        def priority(job, _operation):
            remaining_time = sum(
                op.nominal_processing_time
                for op in job.operations
                if not (hasattr(op, 'completed') and op.completed)
            )
            return remaining_time

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def critical_ratio(self) -> Dict:
        """Critical Ratio scheduling"""
        logger.info("Running Critical Ratio dispatching rule")
        self._reset()

        def priority(job, operation):
            remaining_time = sum(
                op.nominal_processing_time
                for op in job.operations
                if not (hasattr(op, 'completed') and op.completed)
            )
            time_until_due = max(job.due_date - self.env.current_time, 1e-3)
            processing = max(operation.nominal_processing_time, 1e-3)
            ratio = time_until_due / (remaining_time + processing)
            return ratio

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def slack_per_operation(self) -> Dict:
        """Slack per remaining operation"""
        logger.info("Running Slack per Operation dispatching rule")
        self._reset()

        def priority(job, _operation):
            slack = job.due_date - self.env.current_time
            remaining_ops = max(job.remaining_operations, 1)
            return slack / remaining_ops

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()

    def random_dispatch(self) -> Dict:
        """Pure random dispatching baseline"""
        logger.info("Running Random dispatch baseline")
        self._reset()

        while True:
            jobs = [job for job in self.env.pending_jobs if job.remaining_operations > 0]
            if not jobs:
                break

            job = random.choice(jobs)

            available_ops = [
                op for op in job.operations
                if not (hasattr(op, 'completed') and op.completed)
                and self._can_schedule(job, op)
            ]

            if not available_ops:
                if not self._select_job(lambda *_: random.random()):
                    break
                continue

            operation = random.choice(available_ops)
            if not self._assign_operation(job, operation):
                # fall back to generic selection to progress time
                if not self._select_job(lambda *_: random.random()):
                    break

        return self.env.compute_metrics()

    def apparent_tardiness_cost(self, k: float = 2.0) -> Dict:
        """Apparent Tardiness Cost heuristic"""
        logger.info("Running Apparent Tardiness Cost dispatching rule")
        self._reset()

        avg_processing = np.mean([
            op.nominal_processing_time
            for job in self.env.jobs
            for op in job.operations
        ])

        def priority(job, operation):
            processing = max(operation.nominal_processing_time, 1e-3)
            slack = job.due_date - self.env.current_time - processing
            exp_term = np.exp(-max(slack, 0) / (k * max(avg_processing, 1e-3)))
            score = (exp_term / processing) * max(job.weight, 1e-3)
            return -score  # maximize ATC score

        while self._select_job(priority):
            pass

        return self.env.compute_metrics()


class GeneticAlgorithm:
    """Genetic Algorithm for RMS scheduling"""
    
    def __init__(self, env, population_size=100, generations=500, 
                 crossover_prob=0.8, mutation_prob=0.2):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        
        # Setup DEAP
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
    
    def _encode_solution(self) -> List[Tuple]:
        """Encode solution as chromosome"""
        # Chromosome: list of (job_id, op_id, machine_id) tuples
        chromosome = []
        
        for job in self.env.jobs:
            for op in job.operations:
                # Find compatible machines
                compatible = [
                    m.machine_id for m in self.env.machines
                    if op.required_capability in m.current_config.capabilities
                ]
                
                if compatible:
                    machine_id = random.choice(compatible)
                    chromosome.append((job.job_id, op.op_id, machine_id))
        
        return chromosome
    
    def _decode_and_evaluate(self, chromosome: List[Tuple]) -> Tuple[float]:
        """Decode chromosome and evaluate fitness"""
        self.env.reset()
        
        # Build schedule from chromosome
        scheduled = set()
        attempts = 0
        max_attempts = len(chromosome) * 2
        
        while len(scheduled) < len(chromosome) and attempts < max_attempts:
            for i, (job_id, op_id, machine_id) in enumerate(chromosome):
                if i in scheduled:
                    continue
                
                job = self.env.jobs[job_id]
                op = job.operations[op_id]
                
                # Check precedence
                can_schedule = all(
                    hasattr(job.operations[p], 'completed') and job.operations[p].completed
                    for p in op.precedence
                )
                
                if not can_schedule:
                    continue
                
                # Try to assign
                machine = self.env.machines[machine_id]
                if machine.can_process(op):
                    result = self.env.assign_operation(
                        job_id,
                        op_id,
                        machine_id,
                        max(self.env.current_time, machine.next_available_time)
                    )
                    
                    if result['success']:
                        scheduled.add(i)
            
            attempts += 1
        
        # Calculate fitness (makespan + penalty for unscheduled)
        metrics = self.env.compute_metrics()
        makespan = metrics['makespan']
        unscheduled_penalty = (len(chromosome) - len(scheduled)) * 1000
        
        fitness = makespan + unscheduled_penalty
        
        return (fitness,)
    
    def solve(self) -> Dict:
        """Run genetic algorithm"""
        logger.info(f"Running Genetic Algorithm (pop={self.population_size}, gen={self.generations})")
        
        # Register genetic operators
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self._encode_solution)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._decode_and_evaluate)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Create initial population
        population = self.toolbox.population(n=self.population_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution
        best_fitness_history = []
        
        for gen in range(self.generations):
            # Select next generation
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Track best fitness
            fits = [ind.fitness.values[0] for ind in population]
            best_fitness = min(fits)
            best_fitness_history.append(best_fitness)
            
            if (gen + 1) % 50 == 0:
                logger.info(f"Generation {gen + 1}/{self.generations}: Best fitness = {best_fitness:.2f}")
        
        # Get best solution
        best_ind = tools.selBest(population, 1)[0]
        
        # Decode best solution
        self._decode_and_evaluate(best_ind)
        final_metrics = self.env.compute_metrics()
        
        logger.info(f"GA completed. Final makespan: {final_metrics['makespan']:.2f}")
        
        return {
            **final_metrics,
            'best_fitness_history': best_fitness_history
        }


class SimulatedAnnealing:
    """Simulated Annealing for RMS scheduling"""
    
    def __init__(self, env, initial_temp=1000, cooling_rate=0.95, 
                 iterations=10000):
        self.env = env
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
    
    def _generate_initial_solution(self) -> List[Tuple]:
        """Generate initial solution using FIFO"""
        solution = []
        
        for job in sorted(self.env.jobs, key=lambda j: j.arrival_time):
            for op in job.operations:
                # Find compatible machines
                compatible = [
                    m.machine_id for m in self.env.machines
                    if op.required_capability in m.current_config.capabilities
                ]
                
                if compatible:
                    machine_id = random.choice(compatible)
                    solution.append((job.job_id, op.op_id, machine_id))
        
        return solution
    
    def _evaluate_solution(self, solution: List[Tuple]) -> float:
        """Evaluate solution quality"""
        self.env.reset()
        
        # Schedule operations
        scheduled = set()
        attempts = 0
        max_attempts = len(solution) * 2
        
        while len(scheduled) < len(solution) and attempts < max_attempts:
            for i, (job_id, op_id, machine_id) in enumerate(solution):
                if i in scheduled:
                    continue
                
                job = self.env.jobs[job_id]
                op = job.operations[op_id]
                
                # Check precedence
                can_schedule = all(
                    hasattr(job.operations[p], 'completed') and job.operations[p].completed
                    for p in op.precedence
                )
                
                if not can_schedule:
                    continue
                
                machine = self.env.machines[machine_id]
                if machine.can_process(op):
                    result = self.env.assign_operation(
                        job_id,
                        op_id,
                        machine_id,
                        max(self.env.current_time, machine.next_available_time)
                    )
                    
                    if result['success']:
                        scheduled.add(i)
            
            attempts += 1
        
        metrics = self.env.compute_metrics()
        return metrics['makespan']
    
    def _generate_neighbor(self, solution: List[Tuple]) -> List[Tuple]:
        """Generate neighbor solution"""
        neighbor = solution.copy()
        
        # Random modification
        modification = random.choice(['swap', 'reassign'])
        
        if modification == 'swap' and len(neighbor) > 1:
            # Swap two operations
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        elif modification == 'reassign':
            # Reassign operation to different machine
            idx = random.randint(0, len(neighbor) - 1)
            job_id, op_id, _ = neighbor[idx]
            
            job = self.env.jobs[job_id]
            op = job.operations[op_id]
            
            # Find alternative machines
            compatible = [
                m.machine_id for m in self.env.machines
                if op.required_capability in m.current_config.capabilities
            ]
            
            if len(compatible) > 1:
                new_machine = random.choice(compatible)
                neighbor[idx] = (job_id, op_id, new_machine)
        
        return neighbor
    
    def solve(self) -> Dict:
        """Run simulated annealing"""
        logger.info(f"Running Simulated Annealing (temp={self.initial_temp}, iter={self.iterations})")
        
        # Initialize
        current_solution = self._generate_initial_solution()
        current_cost = self._evaluate_solution(current_solution)
        
        best_solution = current_solution.copy()
        best_cost = current_cost
        
        temperature = self.initial_temp
        cost_history = []
        
        for iteration in range(self.iterations):
            # Generate neighbor
            neighbor = self._generate_neighbor(current_solution)
            neighbor_cost = self._evaluate_solution(neighbor)
            
            # Acceptance criterion
            delta = neighbor_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_solution = neighbor
                current_cost = neighbor_cost
                
                # Update best
                if current_cost < best_cost:
                    best_solution = current_solution.copy()
                    best_cost = current_cost
            
            # Cool down
            temperature *= self.cooling_rate
            
            cost_history.append(best_cost)
            
            if (iteration + 1) % 1000 == 0:
                logger.info(f"Iteration {iteration + 1}/{self.iterations}: Best cost = {best_cost:.2f}")
        
        # Evaluate best solution
        self._evaluate_solution(best_solution)
        final_metrics = self.env.compute_metrics()
        
        logger.info(f"SA completed. Final makespan: {final_metrics['makespan']:.2f}")
        
        return {
            **final_metrics,
            'cost_history': cost_history
        }


class SimpleDQN:
    """Simplified Deep Q-Network baseline (placeholder)"""
    
    def __init__(self, env):
        self.env = env
    
    def solve(self) -> Dict:
        """Run DQN (simplified - uses random policy as placeholder)"""
        logger.info("Running DQN baseline (simplified)")
        
        # For demonstration, use random dispatching
        # In full implementation, would train neural network
        
        self.env.reset()
        
        while self.env.pending_jobs:
            # Random action selection
            job = random.choice(self.env.pending_jobs)
            
            for op in job.operations:
                if hasattr(op, 'completed') and op.completed:
                    continue
                
                # Check precedence
                can_schedule = all(
                    hasattr(job.operations[p], 'completed') and job.operations[p].completed
                    for p in op.precedence
                )
                
                if not can_schedule:
                    continue
                
                # Random compatible machine
                compatible = [
                    m.machine_id for m in self.env.machines
                    if m.can_process(op)
                ]
                
                if compatible:
                    machine_id = random.choice(compatible)
                    self.env.assign_operation(
                        job.job_id,
                        op.op_id,
                        machine_id,
                        self.env.current_time
                    )
                    break
        
        return self.env.compute_metrics()