"""
RMS Simulation Environment
Discrete-event simulation of reconfigurable manufacturing systems
"""

import simpy
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MachineState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    RECONFIGURING = "reconfiguring"


@dataclass
class Configuration:
    """Machine configuration specification"""
    config_id: int
    name: str
    capabilities: List[str]
    processing_speeds: Dict[str, float]
    energy_rate: float
    setup_time_from: Dict[int, float] = field(default_factory=dict)
    setup_cost_from: Dict[int, float] = field(default_factory=dict)


@dataclass
class Operation:
    """Manufacturing operation"""
    op_id: int
    job_id: int
    operation_type: str
    required_capability: str
    nominal_processing_time: float
    precedence: List[int] = field(default_factory=list)


@dataclass
class Job:
    """Manufacturing job"""
    job_id: int
    arrival_time: float
    due_date: float
    priority: int
    operations: List[Operation]
    weight: float = 1.0
    
    @property
    def remaining_operations(self) -> int:
        return sum(1 for op in self.operations if not hasattr(op, 'completed') or not op.completed)


@dataclass
class Machine:
    """Reconfigurable machine"""
    machine_id: int
    name: str
    available_configs: List[Configuration]
    current_config: Configuration
    state: MachineState = MachineState.IDLE
    failure_rate: float = 0.0
    next_available_time: float = 0.0
    total_processing_time: float = 0.0
    total_idle_time: float = 0.0
    total_energy: float = 0.0
    reconfiguration_count: int = 0
    
    def can_process(self, operation: Operation) -> bool:
        """Check if machine can process operation in current config"""
        can_do = operation.required_capability in self.current_config.capabilities
        if not can_do:
            logger.debug(f"Machine {self.machine_id} cannot process {operation.required_capability}. "
                        f"Current capabilities: {self.current_config.capabilities}")
        return can_do

class RMSEnvironment:
    """Reconfigurable Manufacturing System Environment"""
    
    def __init__(self, 
                 num_machines: int = 10,
                 num_configs_per_machine: int = 4,
                 seed: int = 42):
        self.env = simpy.Environment()
        self.num_machines = num_machines
        self.seed = seed
        np.random.seed(seed)
        
        # System components
        self.machines: List[Machine] = []
        self.jobs: List[Job] = []
        self.pending_jobs: List[Job] = []
        self.completed_jobs: List[Job] = []
        
        # State tracking
        self.current_time = 0.0
        self.makespan = 0.0
        self.total_tardiness = 0.0
        self.total_energy = 0.0
        self.schedule: List[Dict] = []
        
        # Performance metrics
        self.metrics = {
            'makespan': 0.0,
            'total_tardiness': 0.0,
            'total_flowtime': 0.0,
            'avg_tardiness': 0.0,
            'utilization': 0.0,
            'energy_consumption': 0.0,
            'reconfigurations': 0,
            'completed_jobs': 0
        }
        
        # Initialize system
        self._initialize_machines(num_configs_per_machine)
        
    def _initialize_machines(self, num_configs: int):
        """Initialize machines with configurations"""
        capability_types = ['drilling', 'milling', 'turning', 'grinding', 
                           'welding', 'assembly', 'inspection', 'polishing']
        
        for m_id in range(self.num_machines):
            configs = []
            for c_id in range(num_configs):
                # Random capability subset
                num_caps = np.random.randint(2, min(5, len(capability_types)))
                capabilities = list(np.random.choice(capability_types, num_caps, replace=False))
                
                # Processing speeds for each capability
                speeds = {cap: np.random.uniform(0.8, 1.5) for cap in capabilities}
                
                # Energy consumption
                energy_rate = np.random.uniform(5.0, 15.0)
                
                config = Configuration(
                    config_id=c_id,
                    name=f"Config_{m_id}_{c_id}",
                    capabilities=capabilities,
                    processing_speeds=speeds,
                    energy_rate=energy_rate
                )
                
                # Setup times and costs between configurations
                for prev_c_id in range(c_id):
                    config.setup_time_from[prev_c_id] = np.random.uniform(10, 30)
                    config.setup_cost_from[prev_c_id] = np.random.uniform(50, 200)
                
                configs.append(config)
            
            # Symmetric setup times
            for i, config in enumerate(configs):
                for j, other_config in enumerate(configs):
                    if i != j and j not in config.setup_time_from:
                        config.setup_time_from[j] = np.random.uniform(10, 30)
                        config.setup_cost_from[j] = np.random.uniform(50, 200)
            
            machine = Machine(
                machine_id=m_id,
                name=f"Machine_{m_id}",
                available_configs=configs,
                current_config=configs[0],
                failure_rate=np.random.uniform(0.0, 0.1)
            )
            self.machines.append(machine)
    
    def generate_jobs(self, num_jobs: int, min_ops: int = 3, max_ops: int = 8):
        """Generate random jobs"""
        capability_types = ['drilling', 'milling', 'turning', 'grinding', 
                           'welding', 'assembly', 'inspection', 'polishing']
        
        self.jobs = []
        for j_id in range(num_jobs):
            num_ops = np.random.randint(min_ops, max_ops + 1)
            operations = []
            
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
            
            arrival_time = np.random.exponential(5.0) if j_id > 0 else 0.0
            processing_time_sum = sum(op.nominal_processing_time for op in operations)
            due_date = arrival_time + processing_time_sum * np.random.uniform(2.0, 4.0)
            
            job = Job(
                job_id=j_id,
                arrival_time=arrival_time,
                due_date=due_date,
                priority=np.random.randint(1, 4),
                operations=operations,
                weight=np.random.uniform(0.5, 2.0)
            )
            self.jobs.append(job)
            self.pending_jobs.append(job)
    
    def load_instance(self, instance_data: Dict):
        """Load problem instance from data"""
        self.jobs = instance_data['jobs']
        self.pending_jobs = self.jobs.copy()
        if 'machines' in instance_data:
            self.machines = instance_data['machines']
    
    def reconfigure_machine(self, machine_id: int, new_config_id: int) -> Dict:
        """Reconfigure a machine to new configuration"""
        machine = self.machines[machine_id]
        
        if machine.current_config.config_id == new_config_id:
            return {'success': True, 'message': 'Already in requested configuration', 'time': 0.0}
        
        new_config = machine.available_configs[new_config_id]
        old_config_id = machine.current_config.config_id
        
        # Get setup time and cost
        setup_time = new_config.setup_time_from.get(old_config_id, 20.0)
        setup_cost = new_config.setup_cost_from.get(old_config_id, 100.0)
        
        # Update machine state
        machine.state = MachineState.RECONFIGURING
        machine.next_available_time = self.current_time + setup_time
        machine.current_config = new_config
        machine.reconfiguration_count += 1
        
        # Log reconfiguration
        self.schedule.append({
            'time': self.current_time,
            'event': 'reconfiguration',
            'machine_id': machine_id,
            'from_config': old_config_id,
            'to_config': new_config_id,
            'setup_time': setup_time,
            'setup_cost': setup_cost
        })
        
        logger.info(f"Time {self.current_time:.2f}: Machine {machine_id} reconfiguring to Config {new_config_id}")
        
        return {
            'success': True,
            'setup_time': setup_time,
            'setup_cost': setup_cost,
            'message': f'Machine {machine_id} reconfiguring'
        }
    
    def assign_operation(self, job_id: int, op_id: int, machine_id: int, start_time: float) -> Dict:
        """Assign operation to machine"""
        # Find job and operation
        job = next((j for j in self.pending_jobs + self.completed_jobs if j.job_id == job_id), None)
        if not job:
            return {'success': False, 'message': f'Job {job_id} not found'}
        
        if op_id >= len(job.operations):
            return {'success': False, 'message': f'Operation {op_id} not in job'}
        
        operation = job.operations[op_id]
        machine = self.machines[machine_id]
        
        # Check capability
        if not machine.can_process(operation):
            return {
                'success': False, 
                'message': f'Machine {machine_id} cannot process operation (missing capability {operation.required_capability})'
            }
        
        # Check precedence
        for prec_op_id in operation.precedence:
            prec_op = job.operations[prec_op_id]
            if not hasattr(prec_op, 'completed') or not prec_op.completed:
                return {
                    'success': False,
                    'message': f'Precedence constraint violated: operation {prec_op_id} not completed'
                }
        
        # Calculate processing time with variation
        base_time = operation.nominal_processing_time
        speed_factor = machine.current_config.processing_speeds.get(operation.operation_type, 1.0)
        processing_time = base_time / speed_factor
        
        # Add stochasticity
        if hasattr(self, 'stochastic') and self.stochastic:
            processing_time *= np.random.normal(1.0, 0.1)
            processing_time = max(0.1, processing_time)
        
        # Ensure start time respects machine availability
        actual_start_time = max(start_time, machine.next_available_time)
        completion_time = actual_start_time + processing_time
        
        # Update machine state
        machine.state = MachineState.BUSY
        machine.next_available_time = completion_time
        machine.total_processing_time += processing_time
        machine.total_energy += processing_time * machine.current_config.energy_rate
        
        # Mark operation as completed
        operation.completed = True
        operation.start_time = actual_start_time
        operation.completion_time = completion_time
        operation.machine_id = machine_id
        
        # Log assignment
        self.schedule.append({
            'time': actual_start_time,
            'event': 'operation_start',
            'job_id': job_id,
            'operation_id': op_id,
            'machine_id': machine_id,
            'processing_time': processing_time,
            'completion_time': completion_time
        })
        
        # Check if job is complete
        if all(hasattr(op, 'completed') and op.completed for op in job.operations):
            job.completion_time = max(op.completion_time for op in job.operations)
            job.flowtime = job.completion_time - job.arrival_time
            job.tardiness = max(0, job.completion_time - job.due_date)
            
            if job in self.pending_jobs:
                self.pending_jobs.remove(job)
                self.completed_jobs.append(job)
            
            logger.info(f"Job {job_id} completed at time {job.completion_time:.2f}")
        
        self.current_time = max(self.current_time, actual_start_time)
        
        return {
            'success': True,
            'actual_start_time': actual_start_time,
            'completion_time': completion_time,
            'processing_time': processing_time
        }
    
    def get_state(self) -> Dict:
        """Get current system state"""
        # Machine status
        machine_status = []
        for m in self.machines:
            machine_status.append({
                'machine_id': m.machine_id,
                'state': m.state.value,
                'current_config': m.current_config.config_id,
                'capabilities': m.current_config.capabilities,
                'next_available': m.next_available_time,
                'utilization': m.total_processing_time / max(self.current_time, 1.0),
                'energy': m.total_energy,
                'reconfigurations': m.reconfiguration_count
            })
        
        # Job queue status
        job_queue = []
        for j in self.pending_jobs:
            remaining_ops = [op for op in j.operations if not (hasattr(op, 'completed') and op.completed)]
            job_queue.append({
                'job_id': j.job_id,
                'arrival_time': j.arrival_time,
                'due_date': j.due_date,
                'priority': j.priority,
                'remaining_operations': len(remaining_ops),
                'total_operations': len(j.operations),
                'waiting_time': self.current_time - j.arrival_time,
                'slack': j.due_date - self.current_time
            })
        
        # Performance metrics
        current_metrics = self.compute_metrics()
        
        return {
            'current_time': self.current_time,
            'machines': machine_status,
            'job_queue': job_queue,
            'pending_jobs': len(self.pending_jobs),
            'completed_jobs': len(self.completed_jobs),
            'metrics': current_metrics
        }
    
    def compute_metrics(self) -> Dict:
        """Compute performance metrics"""
        if self.completed_jobs:
            makespan = max(j.completion_time for j in self.completed_jobs)
            total_tardiness = sum(j.tardiness for j in self.completed_jobs)
            avg_tardiness = total_tardiness / len(self.completed_jobs)
            total_flowtime = sum(j.flowtime for j in self.completed_jobs)
        else:
            makespan = self.current_time
            total_tardiness = 0.0
            avg_tardiness = 0.0
            total_flowtime = 0.0
        
        total_processing = sum(m.total_processing_time for m in self.machines)
        total_possible = self.current_time * self.num_machines
        utilization = total_processing / max(total_possible, 1.0)
        
        total_energy = sum(m.total_energy for m in self.machines)
        total_reconfigs = sum(m.reconfiguration_count for m in self.machines)
        
        return {
            'makespan': makespan,
            'total_tardiness': total_tardiness,
            'avg_tardiness': avg_tardiness,
            'total_flowtime': total_flowtime,
            'utilization': utilization,
            'energy_consumption': total_energy,
            'reconfigurations': total_reconfigs,
            'completed_jobs': len(self.completed_jobs)
        }
    
    def reset(self):
        """Reset environment to initial state"""
        self.env = simpy.Environment()
        self.current_time = 0.0
        self.schedule = []
        self.completed_jobs = []
        self.pending_jobs = self.jobs.copy()
        
        # Reset machines
        for machine in self.machines:
            machine.state = MachineState.IDLE
            machine.current_config = machine.available_configs[0]
            machine.next_available_time = 0.0
            machine.total_processing_time = 0.0
            machine.total_idle_time = 0.0
            machine.total_energy = 0.0
            machine.reconfiguration_count = 0
        
        # Reset operations
        for job in self.jobs:
            for op in job.operations:
                if hasattr(op, 'completed'):
                    delattr(op, 'completed')
                if hasattr(op, 'start_time'):
                    delattr(op, 'start_time')
                if hasattr(op, 'completion_time'):
                    delattr(op, 'completion_time')