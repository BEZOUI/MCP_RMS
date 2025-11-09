"""
Experiment runner
Runs experiments comparing all methods
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import time
from tqdm import tqdm
import logging
import multiprocessing as mp
import traceback
import queue as queue_module
from copy import deepcopy
from contextlib import ExitStack, redirect_stdout, redirect_stderr

from src.environment import RMSEnvironment
from src.memory import MemorySystem
from src.llm_client import LLMClient
from src.mcp_server import MCPServer
from src.baselines import (
    DispatchingRules,
    GeneticAlgorithm,
    SimulatedAnnealing,
    SimpleDQN,
)
from src.utils import save_results
from experiments.benchmark_generator import BenchmarkGenerator


def _simple_dqn_worker(instance: Dict, trial: int, queue: "mp.Queue", log_path: str) -> None:
    """Execute the SimpleDQN baseline in an isolated subprocess."""

    from pathlib import Path
    from contextlib import redirect_stdout, redirect_stderr

    start_time = time.time()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_stream, redirect_stdout(log_stream), redirect_stderr(log_stream):
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.handlers = []
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        try:
            instance_copy = deepcopy(instance)
            metadata = instance_copy.get("metadata", {})
            base_seed = metadata.get("seed") or 0

            env = RMSEnvironment(
                num_machines=instance_copy["num_machines"],
                seed=base_seed + trial,
            )
            env.load_instance(instance_copy)
            env.reset()

            dqn = SimpleDQN(env)
            metrics = dqn.solve()

            queue.put(
                {
                    "success": True,
                    "metrics": metrics,
                    "time": time.time() - start_time,
                }
            )
        except BaseException as exc:  # pragma: no cover - defensive logging
            traceback.print_exc()
            queue.put(
                {
                    "success": False,
                    "error": repr(exc),
                    "time": time.time() - start_time,
                }
            )

logger = logging.getLogger(__name__)


DISPATCH_BASELINES = {
    "fifo": "fifo",
    "spt": "spt",
    "edd": "edd",
    "mwkr": "mwkr",
    "lpt": "lpt",
    "lwkr": "lwkr",
    "critical_ratio": "critical_ratio",
    "slack_per_operation": "slack_per_operation",
    "random_dispatch": "random_dispatch",
    "apparent_tardiness_cost": "apparent_tardiness_cost",
}

class ExperimentRunner:
    """Run comprehensive experiments"""

    def __init__(self, config: Dict):
        self.config = config
        default_results_dir = config.get('experiments', {}).get('results_dir', 'data/results')
        self.results_dir = Path(config.get('results_dir', default_results_dir))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_cfg = config.get('baselines', {})
        self.llm_cfg = config.get('llm', {})
        
        # Initialize components
        self.benchmark_gen = BenchmarkGenerator()
        
        # Results storage
        self.all_results = []

    def _filter_methods(self, methods: List[str]) -> List[str]:
        """Remove methods whose dependencies are unavailable."""
        filtered: List[str] = []

        for method in methods:
            if method == "simple_dqn" and not SimpleDQN.is_available():
                logger.warning(
                    "Skipping simple_dqn baseline because PyTorch is not installed."
                )
                continue

            filtered.append(method)

        return filtered
    
    def run_method(
        self,
        method_name: str,
        env: RMSEnvironment,
        trial: int = 0,
        log_path: Optional[Path] = None,
    ) -> Dict:
        """Run single method on instance"""
        logger.info(f"Running {method_name} (trial {trial})")

        start_time = time.time()
        log_handler: Optional[logging.Handler] = None
        log_stream = None

        try:
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_stream = open(log_path, "w")
                log_handler = logging.FileHandler(log_path, mode="w")
                log_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                )
                logging.getLogger().addHandler(log_handler)

            with ExitStack() as stack:
                if log_stream is not None:
                    stack.enter_context(redirect_stdout(log_stream))
                    stack.enter_context(redirect_stderr(log_stream))

                if method_name == "adaptive_mcp_rms":
                    # Initialize memory and LLM
                    memory = MemorySystem()
                    llm = LLMClient(
                        base_url=self.llm_cfg.get('base_url'),
                        model=self.llm_cfg.get('model'),
                        temperature=self.llm_cfg.get('temperature', 0.7),
                        max_tokens=self.llm_cfg.get('max_tokens', 500),
                        request_timeout=self.llm_cfg.get('request_timeout', 60),
                    )

                    # Create MCP server
                    server = MCPServer(env, memory, llm)

                    # Solve
                    result = server.solve(
                        max_iterations=self.config.get('max_iterations', 50),
                        objective="minimize_makespan"
                    )

                    metrics = result['final_metrics']

                elif method_name in DISPATCH_BASELINES:
                    dr = DispatchingRules(env)
                    dispatch_fn = getattr(dr, DISPATCH_BASELINES[method_name])
                    metrics = dispatch_fn()

                elif method_name == "genetic_algorithm":
                    ga = GeneticAlgorithm(
                        env,
                        population_size=self.baselines_cfg.get('ga_population', 100),
                        generations=self.baselines_cfg.get('ga_generations', 500)
                    )
                    metrics = ga.solve()

                elif method_name == "simulated_annealing":
                    sa = SimulatedAnnealing(
                        env,
                        iterations=self.baselines_cfg.get('sa_iterations', 10000)
                    )
                    metrics = sa.solve()

                elif method_name == "simple_dqn":
                    dqn = SimpleDQN(env)
                    metrics = dqn.solve()

                else:
                    raise ValueError(f"Unknown method: {method_name}")

            elapsed = time.time() - start_time

            return {
                'success': True,
                'metrics': metrics,
                'time': elapsed,
                'log_file': str(log_path) if log_path is not None else None,
            }

        except Exception as e:
            logger.error(f"Error running {method_name}: {str(e)}")
            elapsed = time.time() - start_time

            return {
                'success': False,
                'error': str(e),
                'time': elapsed,
                'log_file': str(log_path) if log_path is not None else None,
            }
        finally:
            if log_handler is not None:
                logging.getLogger().removeHandler(log_handler)
                log_handler.close()
            if log_stream is not None:
                log_stream.flush()
                log_stream.close()

    def _run_simple_dqn_isolated(
        self,
        instance: Dict,
        trial: int,
        log_path: Path,
    ) -> Dict:
        """Run SimpleDQN in a subprocess to guard against hard crashes."""

        start_time = time.time()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        queue: mp.Queue = mp.Queue()
        process = mp.Process(
            target=_simple_dqn_worker,
            args=(instance, trial, queue, str(log_path)),
            daemon=True,
        )

        process.start()
        timeout = self.config.get('experiments', {}).get('simple_dqn_timeout')
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            elapsed = time.time() - start_time
            queue.close()
            queue.join_thread()
            return {
                'success': False,
                'error': f"simple_dqn timed out after {timeout} seconds",
                'time': elapsed,
                'log_file': str(log_path),
            }

        elapsed = time.time() - start_time

        result_data = None
        try:
            result_data = queue.get_nowait()
        except queue_module.Empty:
            result_data = None
        finally:
            queue.close()
            queue.join_thread()

        if result_data is None:
            exit_code = process.exitcode
            error_message = (
                f"simple_dqn subprocess exited with code {exit_code}. "
                f"Check log file at {log_path} for details."
            )
            return {
                'success': False,
                'error': error_message,
                'time': elapsed,
                'log_file': str(log_path),
                'exit_code': exit_code,
            }

        result_data.setdefault('time', elapsed)
        result_data['log_file'] = str(log_path)
        return result_data
    
    def run_single_instance(self, instance: Dict, methods: List[str],
                           num_trials: int = 10) -> Dict:
        """Run all methods on single instance"""
        logger.info(f"Running instance: {instance['name']}")

        results = {
            'instance_name': instance['name'],
            'num_machines': instance['num_machines'],
            'num_jobs': instance['num_jobs'],
            'methods': {}
        }

        logs_dir = self.results_dir / "logs" / instance['name']
        logs_dir.mkdir(parents=True, exist_ok=True)

        for method in self._filter_methods(methods):
            logger.info(f"Method: {method}")

            method_results = []

            for trial in range(num_trials):
                log_path = logs_dir / f"{method}_trial{trial}.log"

                if method == "simple_dqn" and self.config.get('experiments', {}).get('isolate_simple_dqn', True):
                    result = self._run_simple_dqn_isolated(instance, trial, log_path)
                else:
                    # Create fresh environment
                    base_seed = (instance.get('metadata', {}) or {}).get('seed') or 0
                    env = RMSEnvironment(
                        num_machines=instance['num_machines'],
                        seed=base_seed + trial
                    )
                    env.load_instance(deepcopy(instance))

                    # Run method
                    result = self.run_method(method, env, trial, log_path)

                method_results.append(result)

            # Aggregate results
            if any(r['success'] for r in method_results):
                successful = [r for r in method_results if r['success']]

                makespans = [r['metrics']['makespan'] for r in successful]
                tardiness = [r['metrics']['total_tardiness'] for r in successful]
                energy = [r['metrics']['energy_consumption'] for r in successful]
                times = [r['time'] for r in successful]

                results['methods'][method] = {
                    'makespan_mean': np.mean(makespans),
                    'makespan_std': np.std(makespans),
                    'makespan_min': np.min(makespans),
                    'makespan_max': np.max(makespans),
                    'tardiness_mean': np.mean(tardiness),
                    'energy_mean': np.mean(energy),
                    'time_mean': np.mean(times),
                    'success_rate': len(successful) / num_trials,
                    'raw_results': method_results
                }
            else:
                results['methods'][method] = {
                    'success_rate': 0.0,
                    'error': "All trials failed",
                    'raw_results': method_results
                }

        return results
    
    def run_benchmark_suite(self, suite_name: str = "benchmark_suite",
                           methods: List[str] = None,
                           num_trials: int = 10):
        """Run all methods on benchmark suite"""
        if methods is None:
            methods = [
                "adaptive_mcp_rms",
                "fifo",
                "spt",
                "edd",
                "mwkr",
                "lpt",
                "lwkr",
                "critical_ratio",
                "slack_per_operation",
                "random_dispatch",
                "apparent_tardiness_cost",
                "genetic_algorithm",
                "simulated_annealing",
                "simple_dqn",
            ]

        methods = self._filter_methods(methods)

        logger.info(f"Running benchmark suite: {suite_name}")
        logger.info(f"Methods: {methods}")
        logger.info(f"Trials per instance: {num_trials}")
        
        # Load instances
        instances = self.benchmark_gen.load_suite(suite_name)
        
        # Run experiments
        for instance in tqdm(instances, desc="Instances"):
            result = self.run_single_instance(instance, methods, num_trials)
            self.all_results.append(result)
            
            # Save intermediate results
            save_results(
                self.all_results,
                self.results_dir / f"{suite_name}_results.json"
            )
        
        logger.info("Benchmark suite completed!")
        
        return self.all_results
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create summary table of results"""
        rows = []
        
        for result in self.all_results:
            for method, data in result['methods'].items():
                if data.get('success_rate', 0) > 0:
                    rows.append({
                        'Instance': result['instance_name'],
                        'Size': f"{result['num_machines']}m_{result['num_jobs']}j",
                        'Method': method,
                        'Makespan': data['makespan_mean'],
                        'Std': data['makespan_std'],
                        'Tardiness': data['tardiness_mean'],
                        'Energy': data['energy_mean'],
                        'Time': data['time_mean'],
                        'Success Rate': data['success_rate']
                    })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV
        df.to_csv(self.results_dir / "summary_table.csv", index=False)

        return df
