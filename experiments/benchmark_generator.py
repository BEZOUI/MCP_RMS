"""
Benchmark instance generator
Creates test instances of various sizes and characteristics
"""

import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List
from src.environment import RMSEnvironment, Job, Operation
import logging

logger = logging.getLogger(__name__)


class BenchmarkGenerator:
    """Generate benchmark RMS instances"""
    
    def __init__(self, save_dir: str = "data/instances"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_instance(self,
                         name: str,
                         num_machines: int,
                         num_jobs: int,
                         min_ops: int = 3,
                         max_ops: int = 8,
                         arrival_pattern: str = 'uniform',
                         due_date_tightness: float = 3.0,
                         seed: int = None) -> Dict:
        """Generate single instance"""
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Generating instance: {name}")
        
        # Create environment
        env = RMSEnvironment(
            num_machines=num_machines,
            num_configs_per_machine=4,
            seed=seed if seed else np.random.randint(0, 10000)
        )
        
        # Generate jobs based on pattern
        if arrival_pattern == 'uniform':
            env.generate_jobs(num_jobs, min_ops, max_ops)
        elif arrival_pattern == 'poisson':
            # Poisson arrivals
            jobs = []
            arrival_time = 0.0
            
            for j_id in range(num_jobs):
                arrival_time += np.random.exponential(5.0)
                
                num_ops = np.random.randint(min_ops, max_ops + 1)
                operations = []
                
                capability_types = ['drilling', 'milling', 'turning', 'grinding']
                
                for op_idx in range(num_ops):
                    op_type = np.random.choice(capability_types)
                    op = Operation(
                        op_id=op_idx,
                        job_id=j_id,
                        operation_type=op_type,
                        required_capability=op_type,
                        nominal_processing_time=np.random.uniform(5, 30),
                        precedence=[op_idx - 1] if op_idx > 0 else []
                    )
                    operations.append(op)
                
                processing_time_sum = sum(op.nominal_processing_time for op in operations)
                due_date = arrival_time + processing_time_sum * due_date_tightness
                
                job = Job(
                    job_id=j_id,
                    arrival_time=arrival_time,
                    due_date=due_date,
                    priority=np.random.randint(1, 4),
                    operations=operations
                )
                jobs.append(job)
            
            env.jobs = jobs
            env.pending_jobs = jobs.copy()
        
        # Create instance data
        instance = {
            'name': name,
            'num_machines': num_machines,
            'num_jobs': num_jobs,
            'machines': env.machines,
            'jobs': env.jobs,
            'metadata': {
                'min_ops': min_ops,
                'max_ops': max_ops,
                'arrival_pattern': arrival_pattern,
                'due_date_tightness': due_date_tightness,
                'seed': seed
            }
        }
        
        return instance
    
    def generate_suite(self, suite_name: str = "benchmark_suite") -> List[Dict]:
        """Generate complete benchmark suite"""
        logger.info(f"Generating benchmark suite: {suite_name}")
        
        instances = []
        
        # Small instances
        for i in range(10):
            instance = self.generate_instance(
                name=f"small_{i:02d}",
                num_machines=5,
                num_jobs=20,
                min_ops=3,
                max_ops=5,
                arrival_pattern='uniform',
                seed=1000 + i
            )
            instances.append(instance)
        
        # Medium instances
        for i in range(10):
            instance = self.generate_instance(
                name=f"medium_{i:02d}",
                num_machines=15,
                num_jobs=50,
                min_ops=3,
                max_ops=8,
                arrival_pattern='poisson',
                seed=2000 + i
            )
            instances.append(instance)
        
        # Large instances
        for i in range(10):
            instance = self.generate_instance(
                name=f"large_{i:02d}",
                num_machines=30,
                num_jobs=100,
                min_ops=4,
                max_ops=10,
                arrival_pattern='poisson',
                seed=3000 + i
            )
            instances.append(instance)
        
        # Save suite
        suite_path = self.save_dir / f"{suite_name}.pkl"
        with open(suite_path, 'wb') as f:
            pickle.dump(instances, f)
        
        logger.info(f"Saved {len(instances)} instances to {suite_path}")
        
        return instances
    
    def save_instance(self, instance: Dict, filename: str = None):
        """Save single instance"""
        if filename is None:
            filename = f"{instance['name']}.pkl"
        
        filepath = self.save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(instance, f)
        
        logger.info(f"Saved instance to {filepath}")
    
    def load_instance(self, filename: str) -> Dict:
        """Load single instance"""
        filepath = self.save_dir / filename
        
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
        
        return instance
    
    def load_suite(self, suite_name: str = "benchmark_suite") -> List[Dict]:
        """Load benchmark suite"""
        suite_path = self.save_dir / f"{suite_name}.pkl"
        
        with open(suite_path, 'rb') as f:
            instances = pickle.load(f)
        
        logger.info(f"Loaded {len(instances)} instances from {suite_path}")
        
        return instances
