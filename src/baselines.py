"""Baseline comparison methods for the RMS benchmark suite."""

import os
import random
from collections import namedtuple
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

import importlib

import numpy as np
from deap import algorithms, base, creator, tools

import logging


# Allow libomp/libiomp to coexist when FAISS and PyTorch are both installed.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _load_optional_torch():
    """Load torch-related modules when available."""
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return None, None, None

    torch_module = importlib.import_module("torch")
    nn_module = importlib.import_module("torch.nn")
    optim_module = importlib.import_module("torch.optim")
    return torch_module, nn_module, optim_module


torch, nn, optim = _load_optional_torch()

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import torch as _torch

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


Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "next_actions", "done"],
)


@dataclass
class ActionCandidate:
    """Potential assignment considered by the DQN policy."""

    job: object
    operation: object
    machine: object
    start_time: float
    processing_time: float
    features: np.ndarray


class ReplayBuffer:
    """Simple replay buffer for off-policy training."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Optional[Transition]] = []
        self.position = 0

    def push(self, transition: Transition) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


if torch is not None:

    class QNetwork(nn.Module):
        """Feed-forward network used to approximate Q-values."""

        def __init__(self, input_dim: int):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def forward(self, x: "_torch.Tensor") -> "_torch.Tensor":  # pragma: no cover - thin wrapper
            return self.model(x).squeeze(-1)

else:  # pragma: no cover - optional dependency fallback

    class QNetwork:  # type: ignore[override]
        """Placeholder raising a clear error when torch is unavailable."""

        def __init__(self, _input_dim: int):
            raise RuntimeError(
                "PyTorch is required for the SimpleDQN baseline. Install torch>=2.0.0 to enable it."
            )



class SimpleDQN:
    """Deep Q-Network baseline with on-the-fly scheduling features."""

    @classmethod
    def is_available(cls) -> bool:
        return torch is not None

    def __init__(self, env):
        if not self.is_available():
            raise RuntimeError(
                "PyTorch is required for the SimpleDQN baseline. Install torch>=2.0.0 to enable it."
            )

        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyper-parameters tuned for stability on the benchmark suite
        self.gamma = 0.95
        self.learning_rate = 1e-3
        self.batch_size = 64
        self.replay_capacity = 50000
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995
        self.target_update_interval = 250
        self.max_idle_advances = 200
        self.tardiness_weight = 2.0
        self.energy_weight = 0.1
        self.completion_bonus = 1.0

        # Runtime members initialised during solve()
        self.policy_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None
        self.optimizer: Optional[object] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.training_steps = 0

        torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    def _initialise_statistics(self) -> None:
        jobs = self.env.jobs
        ops = [op for job in jobs for op in job.operations]
        machines = self.env.machines

        self.max_due_date = max((job.due_date for job in jobs), default=1.0)
        self.max_processing_time = max((op.nominal_processing_time for op in ops), default=1.0)
        self.max_ops_per_job = max((len(job.operations) for job in jobs), default=1)
        self.max_priority = max((job.priority for job in jobs), default=1)
        self.max_weight = max((job.weight for job in jobs), default=1.0)
        self.total_jobs = max(len(jobs), 1)
        self.total_operations = max(len(ops), 1)
        self.max_energy_rate = max(
            (cfg.energy_rate for machine in machines for cfg in machine.available_configs),
            default=1.0,
        )

    def _normalise(self, value: float, scale: float) -> float:
        return float(value) / (scale + 1e-6)

    def _state_features(self) -> np.ndarray:
        current_time = self.env.current_time
        pending_jobs = self.env.pending_jobs
        machines = self.env.machines

        if pending_jobs:
            slacks = [job.due_date - current_time for job in pending_jobs]
            remaining_ops = [job.remaining_operations for job in pending_jobs]
        else:
            slacks = [0.0]
            remaining_ops = [0.0]

        utilizations = [
            machine.total_processing_time / max(current_time, 1.0)
            for machine in machines
        ]
        next_available = [machine.next_available_time for machine in machines]

        features = np.array(
            [
                self._normalise(current_time, self.max_due_date),
                len(pending_jobs) / self.total_jobs,
                np.mean(slacks) / (self.max_due_date + 1e-6),
                np.min(slacks) / (self.max_due_date + 1e-6),
                np.mean(remaining_ops) / self.max_ops_per_job,
                np.mean(utilizations),
                np.mean(next_available) / (self.max_due_date + 1e-6),
                np.max(next_available) / (self.max_due_date + 1e-6),
                len(self.env.completed_jobs) / self.total_jobs,
                self.env.compute_metrics()["energy_consumption"] / (
                    self.max_energy_rate * self.max_due_date * self.total_jobs
                ),
            ],
            dtype=np.float32,
        )

        return features

    def _operation_processing_time(self, machine, operation) -> float:
        speed = machine.current_config.processing_speeds.get(operation.operation_type, 1.0)
        return operation.nominal_processing_time / max(speed, 1e-6)

    def _action_features(
        self,
        job,
        operation,
        machine,
        start_time: float,
        processing_time: float,
    ) -> np.ndarray:
        completion_time = start_time + processing_time
        slack_at_completion = job.due_date - completion_time
        tardiness = max(completion_time - job.due_date, 0.0)

        features = np.array(
            [
                job.arrival_time / (self.max_due_date + 1e-6),
                job.due_date / (self.max_due_date + 1e-6),
                job.priority / (self.max_priority + 1e-6),
                job.weight / (self.max_weight + 1e-6),
                job.remaining_operations / self.max_ops_per_job,
                operation.nominal_processing_time / (self.max_processing_time + 1e-6),
                processing_time / (self.max_processing_time + 1e-6),
                machine.next_available_time / (self.max_due_date + 1e-6),
                start_time / (self.max_due_date + 1e-6),
                completion_time / (self.max_due_date + 1e-6),
                slack_at_completion / (self.max_due_date + 1e-6),
                tardiness / (self.max_due_date + 1e-6),
                machine.current_config.energy_rate / (self.max_energy_rate + 1e-6),
            ],
            dtype=np.float32,
        )

        return features

    def _enumerate_actions(self) -> List[ActionCandidate]:
        candidates: List[ActionCandidate] = []

        for job in self.env.pending_jobs:
            for operation in job.operations:
                if getattr(operation, "completed", False):
                    continue

                if not all(
                    getattr(job.operations[p], "completed", False) for p in operation.precedence
                ):
                    continue

                for machine in self.env.machines:
                    if not machine.can_process(operation):
                        continue

                    start_time = max(self.env.current_time, machine.next_available_time)
                    processing_time = self._operation_processing_time(machine, operation)
                    features = self._action_features(job, operation, machine, start_time, processing_time)
                    candidates.append(
                        ActionCandidate(job, operation, machine, start_time, processing_time, features)
                    )

                # Only consider the next unscheduled operation per job
                break

        return candidates

    def _initialise_network(self) -> None:
        state_dim = len(self._state_features())
        actions = self._enumerate_actions()
        if not actions:
            raise RuntimeError("No feasible actions available to initialise DQN")

        action_dim = len(actions[0].features)
        input_dim = state_dim + action_dim

        self.policy_net = QNetwork(input_dim).to(self.device)
        self.target_net = QNetwork(input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.replay_capacity)
        self.training_steps = 0

    # ------------------------------------------------------------------
    # Reinforcement learning loop
    # ------------------------------------------------------------------
    def _combine(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.concatenate([state, action]).astype(np.float32)

    def _advance_time(self) -> bool:
        future_times = [
            machine.next_available_time
            for machine in self.env.machines
            if machine.next_available_time > self.env.current_time
        ]

        if not future_times:
            return False

        next_time = min(future_times)
        logger.debug(
            "Advancing environment time from %.2f to %.2f to unlock schedulable actions",
            self.env.current_time,
            next_time,
        )
        self.env.current_time = next_time
        return True

    def _compute_reward(
        self,
        candidate: ActionCandidate,
        result: Dict,
        job_completed_before: int,
    ) -> float:
        completion_time = result.get("completion_time", candidate.start_time + candidate.processing_time)
        tardiness = max(completion_time - candidate.job.due_date, 0.0)
        energy_used = candidate.machine.current_config.energy_rate * result.get(
            "processing_time", candidate.processing_time
        )

        reward = -(
            completion_time / (self.max_due_date + 1e-6)
            + self.tardiness_weight * tardiness / (self.max_due_date + 1e-6)
            + self.energy_weight * energy_used / (self.max_energy_rate * self.max_due_date + 1e-6)
        )

        job_completed_after = len(self.env.completed_jobs)
        if job_completed_after > job_completed_before:
            # Encourage completing jobs to shorten the makespan
            reward += self.completion_bonus

        return reward

    def _optimise_model(self) -> None:
        if self.replay_buffer is None or len(self.replay_buffer) < self.batch_size:
            return

        assert self.policy_net is not None
        assert self.target_net is not None
        assert self.optimizer is not None

        transitions = self.replay_buffer.sample(self.batch_size)

        state_actions = np.stack([self._combine(t.state, t.action) for t in transitions])
        rewards = np.array([t.reward for t in transitions], dtype=np.float32)

        targets = []
        for t in transitions:
            if t.done or len(t.next_actions) == 0:
                targets.append(t.reward)
                continue

            next_inputs = np.stack([self._combine(t.next_state, na) for na in t.next_actions])
            next_tensor = torch.tensor(next_inputs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                max_next_q = torch.max(self.target_net(next_tensor)).item()
            targets.append(t.reward + self.gamma * max_next_q)

        state_action_tensor = torch.tensor(state_actions, dtype=torch.float32, device=self.device)
        reward_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)

        predicted_q = self.policy_net(state_action_tensor)
        loss = nn.SmoothL1Loss()(predicted_q, reward_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def solve(self) -> Dict:
        """Train and deploy a DQN policy to schedule operations."""

        logger.info("Running DQN baseline with neural policy")

        self.env.reset()
        self._initialise_statistics()

        # Reset learning state for every instance to keep comparisons fair
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.replay_buffer = None
        self.training_steps = 0

        epsilon = self.epsilon_start
        total_reward = 0.0
        decision_count = 0
        idle_advances = 0

        # Lazily initialise the neural networks once state dimensions are known
        actions = self._enumerate_actions()
        if not actions:
            if not self._advance_time():
                logger.warning("DQN encountered an instance with no feasible actions")
                return self.env.compute_metrics()
            actions = self._enumerate_actions()

        self._initialise_network()

        while True:
            state_features = self._state_features()
            actions = self._enumerate_actions()

            if not actions:
                if not self._advance_time():
                    logger.warning("DQN stalled without feasible actions; terminating early")
                    break

                idle_advances += 1
                if idle_advances > self.max_idle_advances:
                    logger.warning("Exceeded idle advance limit in DQN; aborting run")
                    break
                continue

            idle_advances = 0

            candidates_matrix = np.stack([self._combine(state_features, a.features) for a in actions])

            if random.random() < epsilon:
                action_index = random.randrange(len(actions))
            else:
                with torch.no_grad():
                    tensor = torch.tensor(candidates_matrix, dtype=torch.float32, device=self.device)
                    q_values = self.policy_net(tensor).cpu().numpy()
                action_index = int(np.argmax(q_values))

            selected = actions[action_index]
            prev_completed_jobs = len(self.env.completed_jobs)

            result = self.env.assign_operation(
                selected.job.job_id,
                selected.operation.op_id,
                selected.machine.machine_id,
                selected.start_time,
            )

            if not result.get("success", False):
                logger.debug(
                    "Assignment failed for job %s op %s on machine %s: %s",
                    selected.job.job_id,
                    selected.operation.op_id,
                    selected.machine.machine_id,
                    result.get("message", "unknown error"),
                )
                epsilon = min(1.0, epsilon * 1.05)
                continue

            decision_count += 1

            reward = self._compute_reward(selected, result, prev_completed_jobs)
            total_reward += reward

            next_state = self._state_features()
            next_actions = self._enumerate_actions()
            transition = Transition(
                state=state_features.copy(),
                action=selected.features.copy(),
                reward=reward,
                next_state=next_state.copy(),
                next_actions=[a.features.copy() for a in next_actions],
                done=len(self.env.pending_jobs) == 0,
            )

            assert self.replay_buffer is not None
            self.replay_buffer.push(transition)
            self._optimise_model()

            epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

            if transition.done:
                break

        metrics = self.env.compute_metrics()
        metrics.update(
            {
                "training_reward": total_reward,
                "decisions": decision_count,
                "epsilon_final": epsilon,
            }
        )

        return metrics
