"""
MCP Primitives for RMS
All primitive operations exposed to LLM
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.environment import RMSEnvironment


@dataclass
class PrimitiveResult:
    """Result from primitive execution"""
    success: bool
    data: Any
    message: str
    metadata: Dict = None


class ConfigurationPrimitives:
    """Configuration management primitives"""
    
    def __init__(self, env: RMSEnvironment, memory: 'MemorySystem'):
        self.env = env
        self.memory = memory
    
    def reconfigure_machine(self, machine_id: int, new_config: int) -> PrimitiveResult:
        """Reconfigure machine to new configuration"""
        result = self.env.reconfigure_machine(machine_id, new_config)
        
        return PrimitiveResult(
            success=result['success'],
            data=result,
            message=result.get('message', 'Reconfiguration completed'),
            metadata={'machine_id': machine_id, 'config': new_config}
        )
    
    def suggest_config(self, job_set: List[int]) -> PrimitiveResult:
        """Suggest configuration based on historical performance"""
        # Extract job characteristics
        jobs = [self.env.jobs[jid] for jid in job_set if jid < len(self.env.jobs)]
        
        if not jobs:
            return PrimitiveResult(
                success=False,
                data=None,
                message="No valid jobs in set"
            )
        
        # Get operation types needed
        op_types = set()
        for job in jobs:
            for op in job.operations:
                op_types.add(op.operation_type)
        
        # Query memory for similar scenarios
        similar_configs = self.memory.query_config_performance(op_types)
        
        # Recommend configuration for each machine
        recommendations = []
        for machine in self.env.machines:
            # Find configs that cover required operations
            compatible_configs = []
            for config in machine.available_configs:
                coverage = len(op_types.intersection(set(config.capabilities)))
                if coverage > 0:
                    # Get historical performance
                    hist_perf = similar_configs.get(f"{machine.machine_id}_{config.config_id}", {})
                    score = coverage + hist_perf.get('success_rate', 0.5)
                    compatible_configs.append((config.config_id, score, coverage))
            
            if compatible_configs:
                best_config = max(compatible_configs, key=lambda x: x[1])
                recommendations.append({
                    'machine_id': machine.machine_id,
                    'recommended_config': best_config[0],
                    'score': best_config[1],
                    'coverage': best_config[2]
                })
        
        return PrimitiveResult(
            success=True,
            data=recommendations,
            message=f"Generated {len(recommendations)} configuration recommendations"
        )


class SchedulingPrimitives:
    """Scheduling operation primitives"""
    
    def __init__(self, env: RMSEnvironment, memory: 'MemorySystem'):
        self.env = env
        self.memory = memory
    
    def assign_operation(self, job_id: int, op_id: int, 
                        machine_id: int, start_time: float) -> PrimitiveResult:
        """Assign operation to machine"""
        result = self.env.assign_operation(job_id, op_id, machine_id, start_time)
        
        # Record in memory
        if result['success']:
            self.memory.record_assignment(
                job_id=job_id,
                op_id=op_id,
                machine_id=machine_id,
                result=result
            )
        
        return PrimitiveResult(
            success=result['success'],
            data=result,
            message=result.get('message', 'Operation assigned')
        )
    
    def batch_assign(self, assignments: List[Dict], strategy: str = 'FIFO') -> PrimitiveResult:
        """Batch assign operations using strategy"""
        results = []
        
        # Sort based on strategy
        if strategy == 'FIFO':
            # Sort by arrival time
            sorted_assigns = sorted(assignments, 
                                   key=lambda x: self.env.jobs[x['job_id']].arrival_time)
        elif strategy == 'SPT':
            # Sort by processing time
            sorted_assigns = sorted(assignments,
                                   key=lambda x: self.env.jobs[x['job_id']].operations[x['op_id']].nominal_processing_time)
        elif strategy == 'EDD':
            # Sort by due date
            sorted_assigns = sorted(assignments,
                                   key=lambda x: self.env.jobs[x['job_id']].due_date)
        else:
            sorted_assigns = assignments
        
        for assign in sorted_assigns:
            result = self.assign_operation(
                assign['job_id'],
                assign['op_id'],
                assign['machine_id'],
                assign.get('start_time', self.env.current_time)
            )
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        
        return PrimitiveResult(
            success=True,
            data=results,
            message=f"Batch assigned: {success_count}/{len(assignments)} successful"
        )
    
    def insert_urgent(self, job_id: int) -> PrimitiveResult:
        """Insert urgent job with priority"""
        job = self.env.jobs[job_id]
        job.priority = 10  # Highest priority
        
        # Find earliest available slots
        schedules = []
        for op in job.operations:
            if hasattr(op, 'completed') and op.completed:
                continue
            
            # Find compatible machines
            compatible = []
            for machine in self.env.machines:
                if machine.can_process(op):
                    compatible.append((machine.machine_id, machine.next_available_time))
            
            if compatible:
                # Choose earliest available
                best_machine, avail_time = min(compatible, key=lambda x: x[1])
                schedules.append({
                    'job_id': job_id,
                    'op_id': op.op_id,
                    'machine_id': best_machine,
                    'start_time': avail_time
                })
        
        # Execute assignments
        if schedules:
            result = self.batch_assign(schedules, strategy='none')
            return PrimitiveResult(
                success=True,
                data=result.data,
                message=f"Urgent job {job_id} scheduled on {len(schedules)} operations"
            )
        else:
            return PrimitiveResult(
                success=False,
                data=None,
                message=f"No compatible machines for urgent job {job_id}"
            )


class QueryPrimitives:
    """State query primitives"""
    
    def __init__(self, env: RMSEnvironment):
        self.env = env
    
    def get_machine_status(self, machine_id: Optional[int] = None) -> PrimitiveResult:
        """Get status of machine(s)"""
        state = self.env.get_state()
        
        if machine_id is not None:
            machine_data = next((m for m in state['machines'] if m['machine_id'] == machine_id), None)
            if machine_data:
                return PrimitiveResult(
                    success=True,
                    data=machine_data,
                    message=f"Machine {machine_id} status retrieved"
                )
            else:
                return PrimitiveResult(
                    success=False,
                    data=None,
                    message=f"Machine {machine_id} not found"
                )
        else:
            return PrimitiveResult(
                success=True,
                data=state['machines'],
                message=f"Retrieved status for {len(state['machines'])} machines"
            )
    
    def check_feasibility(self, job_id: int, op_id: int, machine_id: int) -> PrimitiveResult:
        """Check if assignment is feasible"""
        job = self.env.jobs[job_id]
        operation = job.operations[op_id]
        machine = self.env.machines[machine_id]
        
        # Check capability
        can_process = machine.can_process(operation)
        
        # Check precedence
        precedence_ok = True
        for prec_id in operation.precedence:
            prec_op = job.operations[prec_id]
            if not (hasattr(prec_op, 'completed') and prec_op.completed):
                precedence_ok = False
                break
        
        feasible = can_process and precedence_ok
        
        return PrimitiveResult(
            success=True,
            data={
                'feasible': feasible,
                'can_process': can_process,
                'precedence_ok': precedence_ok,
                'machine_config': machine.current_config.config_id,
                'required_capability': operation.required_capability
            },
            message=f"Feasibility check: {'Yes' if feasible else 'No'}"
        )
    
    def evaluate_makespan(self, schedule: List[Dict]) -> PrimitiveResult:
        """Evaluate makespan for proposed schedule"""
        # Create temporary completion times
        completion_times = {}
        
        for assignment in schedule:
            job_id = assignment['job_id']
            op_id = assignment['op_id']
            machine_id = assignment['machine_id']
            start_time = assignment.get('start_time', 0)
            
            job = self.env.jobs[job_id]
            operation = job.operations[op_id]
            machine = self.env.machines[machine_id]
            
            # Calculate processing time
            base_time = operation.nominal_processing_time
            speed = machine.current_config.processing_speeds.get(operation.operation_type, 1.0)
            proc_time = base_time / speed
            
            completion = start_time + proc_time
            completion_times[f"{job_id}_{op_id}"] = completion
        
        makespan = max(completion_times.values()) if completion_times else 0.0
        
        return PrimitiveResult(
            success=True,
            data={'makespan': makespan, 'completion_times': completion_times},
            message=f"Estimated makespan: {makespan:.2f}"
        )
    
    def find_bottleneck(self) -> PrimitiveResult:
        """Identify bottleneck machine"""
        state = self.env.get_state()
        
        # Find machine with highest utilization or longest queue
        bottleneck = None
        max_load = 0.0
        
        for machine_data in state['machines']:
            load = machine_data['utilization']
            if load > max_load:
                max_load = load
                bottleneck = machine_data
        
        return PrimitiveResult(
            success=True,
            data=bottleneck,
            message=f"Bottleneck: Machine {bottleneck['machine_id']} (util: {max_load:.2%})"
        )


class LearningPrimitives:
    """Memory and learning primitives"""
    
    def __init__(self, memory: 'MemorySystem'):
        self.memory = memory
    
    def recall_similar_state(self, current_state: Dict, k: int = 5) -> PrimitiveResult:
        """Recall similar past states"""
        similar_states = self.memory.find_similar_states(current_state, k)
        
        return PrimitiveResult(
            success=True,
            data=similar_states,
            message=f"Retrieved {len(similar_states)} similar past states"
        )
    
    def get_performance_history(self, config_id: str) -> PrimitiveResult:
        """Get historical performance for configuration"""
        history = self.memory.get_config_history(config_id)
        
        if history:
            avg_makespan = np.mean([h['makespan'] for h in history])
            success_rate = np.mean([h['success'] for h in history])
            
            return PrimitiveResult(
                success=True,
                data={
                    'history': history,
                    'avg_makespan': avg_makespan,
                    'success_rate': success_rate,
                    'count': len(history)
                },
                message=f"Retrieved {len(history)} historical records"
            )
        else:
            return PrimitiveResult(
                success=False,
                data=None,
                message="No historical data found"
            )
    
    def compare_strategies(self, strategy_a: str, strategy_b: str) -> PrimitiveResult:
        """Compare historical performance of strategies"""
        perf_a = self.memory.get_strategy_performance(strategy_a)
        perf_b = self.memory.get_strategy_performance(strategy_b)
        
        comparison = {
            'strategy_a': {
                'name': strategy_a,
                'avg_makespan': perf_a.get('avg_makespan', float('inf')),
                'success_rate': perf_a.get('success_rate', 0.0),
                'count': perf_a.get('count', 0)
            },
            'strategy_b': {
                'name': strategy_b,
                'avg_makespan': perf_b.get('avg_makespan', float('inf')),
                'success_rate': perf_b.get('success_rate', 0.0),
                'count': perf_b.get('count', 0)
            }
        }
        
        if perf_a.get('avg_makespan', float('inf')) < perf_b.get('avg_makespan', float('inf')):
            comparison['recommendation'] = strategy_a
        else:
            comparison['recommendation'] = strategy_b
        
        return PrimitiveResult(
            success=True,
            data=comparison,
            message=f"Comparison: {comparison['recommendation']} performs better"
        )
    
    def record_outcome(self, state: Dict, action: Dict, reward: float) -> PrimitiveResult:
        """Record episode outcome in memory"""
        self.memory.store_episode(state, action, reward)
        
        return PrimitiveResult(
            success=True,
            data={'reward': reward},
            message="Outcome recorded in memory"
        )


class PrimitiveRegistry:
    """Registry of all available primitives"""
    
    def __init__(self, env: RMSEnvironment, memory: 'MemorySystem'):
        self.config_prims = ConfigurationPrimitives(env, memory)
        self.schedule_prims = SchedulingPrimitives(env, memory)
        self.query_prims = QueryPrimitives(env)
        self.learning_prims = LearningPrimitives(memory)
        
        # Build primitive catalog
        self.primitives = {
            # Configuration
            'reconfigure_machine': self.config_prims.reconfigure_machine,
            'suggest_config': self.config_prims.suggest_config,
            
            # Scheduling
            'assign_operation': self.schedule_prims.assign_operation,
            'batch_assign': self.schedule_prims.batch_assign,
            'insert_urgent': self.schedule_prims.insert_urgent,
            
            # Query
            'get_machine_status': self.query_prims.get_machine_status,
            'check_feasibility': self.query_prims.check_feasibility,
            'evaluate_makespan': self.query_prims.evaluate_makespan,
            'find_bottleneck': self.query_prims.find_bottleneck,
            
            # Learning
            'recall_similar_state': self.learning_prims.recall_similar_state,
            'get_performance_history': self.learning_prims.get_performance_history,
            'compare_strategies': self.learning_prims.compare_strategies,
            'record_outcome': self.learning_prims.record_outcome
        }
    
    def execute(self, primitive_name: str, **kwargs) -> PrimitiveResult:
        """Execute primitive by name"""
        if primitive_name not in self.primitives:
            return PrimitiveResult(
                success=False,
                data=None,
                message=f"Primitive '{primitive_name}' not found"
            )
        
        try:
            result = self.primitives[primitive_name](**kwargs)
            return result
        except Exception as e:
            return PrimitiveResult(
                success=False,
                data=None,
                message=f"Error executing '{primitive_name}': {str(e)}"
            )
    
    def get_primitive_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all primitives"""
        return {
            'reconfigure_machine': 'Reconfigure a machine to a new configuration',
            'suggest_config': 'Suggest optimal configurations based on job requirements and history',
            'assign_operation': 'Assign a single operation to a machine at specified time',
            'batch_assign': 'Assign multiple operations using a dispatching strategy (FIFO/SPT/EDD)',
            'insert_urgent': 'Schedule urgent job with highest priority',
            'get_machine_status': 'Get current status and metrics for machine(s)',
            'check_feasibility': 'Check if operation assignment is feasible',
            'evaluate_makespan': 'Evaluate makespan for a proposed schedule',
            'find_bottleneck': 'Identify the bottleneck machine in the system',
            'recall_similar_state': 'Retrieve similar past states from memory',
            'get_performance_history': 'Get historical performance data for a configuration',
            'compare_strategies': 'Compare performance of two scheduling strategies',
            'record_outcome': 'Store episode outcome in memory for learning'
        }