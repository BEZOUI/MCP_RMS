"""
Experiment runner
Runs experiments comparing all methods
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List
import time
from tqdm import tqdm
import logging

from src.environment import RMSEnvironment
from src.memory import MemorySystem
from src.llm_client import LLMClient
from src.mcp_server import MCPServer
from src.baselines import DispatchingRules, GeneticAlgorithm, SimulatedAnnealing
from src.utils import save_results
from experiments.benchmark_generator import BenchmarkGenerator

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Run comprehensive experiments"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = Path(config.get('results_dir', 'data/results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.benchmark_gen = BenchmarkGenerator()
        
        # Results storage
        self.all_results = []
    
    def run_method(self, method_name: str, env: RMSEnvironment, 
                   trial: int = 0) -> Dict:
        """Run single method on instance"""
        logger.info(f"Running {method_name} (trial {trial})")
        
        start_time = time.time()
        
        try:
            if method_name == "adaptive_mcp_rms":
                # Initialize memory and LLM
                memory = MemorySystem()
                llm = LLMClient(
                    api_key=self.config.get('anthropic_api_key'),
                    temperature=0.3
                )
                
                # Create MCP server
                server = MCPServer(env, memory, llm)
                
                # Solve
                result = server.solve(
                    max_iterations=self.config.get('max_iterations', 50),
                    objective="minimize_makespan"
                )
                
                metrics = result['final_metrics']
                
            elif method_name == "fifo":
                dr = DispatchingRules(env)
                metrics = dr.fifo()
                
            elif method_name == "spt":
                dr = DispatchingRules(env)
                metrics = dr.spt()
                
            elif method_name == "edd":
                dr = DispatchingRules(env)
                metrics = dr.edd()
                
            elif method_name == "mwkr":
                dr = DispatchingRules(env)
                metrics = dr.mwkr()
                
            elif method_name == "genetic_algorithm":
                ga = GeneticAlgorithm(
                    env,
                    population_size=self.config.get('ga_population', 100),
                    generations=self.config.get('ga_generations', 500)
                )
                metrics = ga.solve()
                
            elif method_name == "simulated_annealing":
                sa = SimulatedAnnealing(
                    env,
                    iterations=self.config.get('sa_iterations', 10000)
                )
                metrics = sa.solve()
                
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            elapsed = time.time() - start_time
            
            return {
                'success': True,
                'metrics': metrics,
                'time': elapsed
            }
            
        except Exception as e:
            logger.error(f"Error running {method_name}: {str(e)}")
            elapsed = time.time() - start_time
            
            return {
                'success': False,
                'error': str(e),
                'time': elapsed
            }
    
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
        
        for method in methods:
            logger.info(f"Method: {method}")
            
            method_results = []
            
            for trial in range(num_trials):
                # Create fresh environment
                env = RMSEnvironment(
                    num_machines=instance['num_machines'],
                    seed=instance['metadata']['seed'] + trial
                )
                env.load_instance(instance)
                
                # Run method
                result = self.run_method(method, env, trial)
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
                    'error': "All trials failed"
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
                "genetic_algorithm",
                "simulated_annealing"
            ]
        
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