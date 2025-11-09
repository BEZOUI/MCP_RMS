"""
MCP Server
Coordinates primitives, memory, and LLM
"""

import logging
from typing import Dict, List, Any, Optional
from src.environment import RMSEnvironment
from src.primitives import PrimitiveRegistry, PrimitiveResult
from src.memory import MemorySystem
from src.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MCPServer:
    """Model Context Protocol Server for RMS"""
    
    def __init__(self,
                 env: RMSEnvironment,
                 memory: MemorySystem,
                 llm: LLMClient):
        self.env = env
        self.memory = memory
        self.llm = llm
        
        # Initialize primitive registry
        self.primitives = PrimitiveRegistry(env, memory)
        
        # Execution history
        self.execution_history = []
        
        # Episode tracking
        self.current_episode = {
            'actions': [],
            'rewards': [],
            'states': []
        }

        # LLM health tracking
        self.llm_enabled = True
        self._llm_failure_count = 0
        self._max_llm_failures = 3
        self._llm_disable_reason: Optional[str] = None
        self._llm_disable_announced = False
    
    def solve(self, 
              max_iterations: int = 50,
              objective: str = "minimize_makespan") -> Dict:
        """Main solving loop"""
        logger.info("Starting MCP-RMS solving process")
        
        # Initial state
        initial_state = self.env.get_state()
        self.current_episode['states'].append(initial_state)
        
        # Initial prompt
        initial_prompt = f"""
You are tasked with optimizing this Reconfigurable Manufacturing System.

Objective: {objective}

Please analyze the current state and develop a strategy to achieve the objective.
Use the available primitives to:
1. Configure machines optimally
2. Schedule operations efficiently
3. Handle disruptions adaptively
4. Learn from past experiences

Start by querying similar past states, then propose your strategy.
"""
        
        iteration = 0
        converged = False
        
        while iteration < max_iterations and not converged:
            logger.info(f"Iteration {iteration + 1}/{max_iterations}")
            
            # Get current state
            current_state = self.env.get_state()
            
            # Query memory for similar states
            similar_episodes = self.memory.find_similar_states(current_state, k=5)
            
            actions: List[Dict[str, Any]]
            if self.llm_enabled:
                llm_response = self.llm.generate_response(
                    user_message=initial_prompt if iteration == 0 else "Continue optimization based on current state.",
                    state=current_state,
                    similar_episodes=similar_episodes,
                    primitive_descriptions=self.primitives.get_primitive_descriptions()
                )

                if not llm_response['success']:
                    self._llm_failure_count += 1
                    logger.error(
                        "LLM generation failed: %s", llm_response.get('error')
                    )

                    disable_reason: Optional[str] = None
                    if llm_response.get('fatal'):
                        disable_reason = llm_response.get('error') or 'fatal error reported by LLM'
                    elif self._llm_failure_count >= self._max_llm_failures:
                        disable_reason = f"{self._llm_failure_count} consecutive failures"

                    if disable_reason:
                        self.llm_enabled = False
                        self._llm_disable_reason = disable_reason
                        self._llm_disable_announced = False
                        logger.error("Disabling LLM client (%s)", disable_reason)

                    actions = self._generate_default_actions(current_state)
                    if not actions:
                        logger.error("No fallback actions available - stopping optimisation loop")
                        break
                    logger.info(
                        "Using heuristic fallback actions because the LLM request did not succeed"
                    )
                else:
                    self._llm_failure_count = 0
                    # Parse actions from response
                    actions = self.llm.parse_action_from_response(llm_response['response'])

                    if not actions:
                        logger.warning("No actions parsed from LLM response")
                        # Try to schedule remaining jobs with default strategy
                        actions = self._generate_default_actions(current_state)
            else:
                if not self._llm_disable_announced:
                    reason = self._llm_disable_reason or "previous errors"
                    logger.warning(
                        "LLM disabled (%s). Continuing with heuristic actions only.",
                        reason,
                    )
                    self._llm_disable_announced = True
                actions = self._generate_default_actions(current_state)
                if not actions:
                    logger.error("No fallback actions available - stopping optimisation loop")
                    break
                logger.info("Using heuristic fallback actions because the LLM client is disabled")

            # Execute actions
            iteration_reward = 0.0
            for action in actions:
                result = self._execute_action(action)
                
                # Calculate reward
                reward = self._calculate_reward(result, current_state)
                iteration_reward += reward
                
                # Record
                self.current_episode['actions'].append(action)
                self.current_episode['rewards'].append(reward)
            
            # Store in memory
            next_state = self.env.get_state()
            self.memory.store_episode(
                state=current_state,
                action={'iteration': iteration, 'actions': actions},
                reward=iteration_reward,
                next_state=next_state,
                metadata={'iteration': iteration}
            )
            
            self.current_episode['states'].append(next_state)
            
            # Check convergence
            if next_state['pending_jobs'] == 0:
                converged = True
                logger.info("All jobs scheduled - converged!")
            
            iteration += 1
        
        # Rebuild memory index
        if iteration % 10 == 0:
            self.memory.build_index()
        
        # Final metrics
        final_metrics = self.env.compute_metrics()
        
        logger.info(f"Solving completed in {iteration} iterations")
        logger.info(f"Final makespan: {final_metrics['makespan']:.2f}")
        
        return {
            'success': True,
            'iterations': iteration,
            'converged': converged,
            'final_metrics': final_metrics,
            'execution_history': self.execution_history
        }
    
    def _execute_action(self, action: Dict) -> PrimitiveResult:
        """Execute a single primitive action"""
        primitive_name = action['primitive']
        parameters = action['parameters']
        
        logger.debug(f"Executing: {primitive_name}({parameters})")
        
        # Execute primitive
        result = self.primitives.execute(primitive_name, **parameters)
        
        # Record execution
        self.execution_history.append({
            'primitive': primitive_name,
            'parameters': parameters,
            'result': result.to_dict() if hasattr(result, 'to_dict') else result.__dict__,
            'timestamp': self.env.current_time
        })
        
        return result
    
    def _calculate_reward(self, result: PrimitiveResult, state: Dict) -> float:
        """Calculate reward for action"""
        if not result.success:
            return -1.0  # Penalty for failed action
        
        # Get metrics before and after
        old_metrics = state['metrics']
        new_metrics = self.env.compute_metrics()
        
        # Reward components
        reward = 0.0
        
        # Makespan improvement
        if new_metrics['completed_jobs'] > old_metrics['completed_jobs']:
            reward += 10.0  # Reward for completing jobs
        
        # Tardiness penalty
        tardiness_delta = new_metrics['total_tardiness'] - old_metrics['total_tardiness']
        reward -= tardiness_delta * 0.5
        
        # Energy efficiency
        energy_delta = new_metrics['energy_consumption'] - old_metrics['energy_consumption']
        reward -= energy_delta * 0.01
        
        # Reconfiguration penalty
        reconfig_delta = new_metrics['reconfigurations'] - old_metrics['reconfigurations']
        reward -= reconfig_delta * 2.0
        
        return reward
    
    def _generate_default_actions(self, state: Dict) -> List[Dict]:
        """Generate robust heuristic actions when the LLM is unavailable."""

        ready_operations: List[Dict[str, Any]] = []
        current_time = self.env.current_time

        # Identify the first ready (precedence satisfied) operation for each pending job
        for job in self.env.pending_jobs:
            for operation in job.operations:
                if getattr(operation, "completed", False):
                    continue

                if not all(
                    getattr(job.operations[p], "completed", False) for p in operation.precedence
                ):
                    # Predecessors not completed yet
                    continue

                ready_operations.append(
                    {
                        "job": job,
                        "operation": operation,
                    }
                )
                break  # only consider the first available operation per job

        if not ready_operations:
            return []

        # Sort jobs by earliest due date then arrival/priority to mimic EDD/priority dispatching
        ready_operations.sort(
            key=lambda item: (
                item["job"].due_date,
                item["job"].priority,
                item["job"].arrival_time,
            )
        )

        best_plan: Optional[Dict[str, Any]] = None

        for item in ready_operations:
            job = item["job"]
            operation = item["operation"]

            for machine in self.env.machines:
                available_time = max(current_time, machine.next_available_time)
                current_config = machine.current_config

                # Helper to evaluate candidate plans
                def consider_plan(
                    target_config,
                    requires_reconfig: bool,
                    setup_time: float,
                ) -> None:
                    nonlocal best_plan
                    processing_speed = target_config.processing_speeds.get(operation.operation_type, 1.0)
                    if processing_speed <= 0:
                        return

                    start_time = available_time
                    if requires_reconfig:
                        start_time += setup_time

                    completion_time = start_time + operation.nominal_processing_time / processing_speed

                    candidate = {
                        "job": job,
                        "operation": operation,
                        "machine": machine,
                        "start_time": start_time,
                        "available_time": available_time,
                        "requires_reconfig": requires_reconfig,
                        "target_config": target_config,
                        "setup_time": setup_time,
                        "completion_time": completion_time,
                    }

                    if (best_plan is None) or (
                        completion_time < best_plan["completion_time"]
                    ):
                        best_plan = candidate

                if operation.required_capability in current_config.capabilities:
                    consider_plan(current_config, False, 0.0)
                    continue

                target_config = next(
                    (
                        cfg
                        for cfg in machine.available_configs
                        if operation.required_capability in cfg.capabilities
                    ),
                    None,
                )

                if not target_config:
                    continue

                setup_time = target_config.setup_time_from.get(
                    current_config.config_id, 0.0
                )
                consider_plan(target_config, True, setup_time)

        if not best_plan:
            # No feasible action right now; try advancing time to the next machine availability
            next_available = min(m.next_available_time for m in self.env.machines)
            if next_available > current_time + 1e-6:
                logger.info(
                    "Advancing time to %.2f to await machine availability", next_available
                )
                self.env.current_time = next_available
                return self._generate_default_actions(self.env.get_state())

        if not best_plan:
            logger.warning(
                "No feasible heuristic action found for %d pending jobs", len(self.env.pending_jobs)
            )
            return []

        planned_actions: List[Dict[str, Any]] = []

        if best_plan["requires_reconfig"] and self.env.current_time < best_plan["available_time"]:
            # Fast-forward the environment clock so the reconfiguration reflects the correct start time
            self.env.current_time = best_plan["available_time"]

        if best_plan["requires_reconfig"] and (
            best_plan["target_config"].config_id != best_plan["machine"].current_config.config_id
        ):
            planned_actions.append(
                {
                    "primitive": "reconfigure_machine",
                    "parameters": {
                        "machine_id": best_plan["machine"].machine_id,
                        "new_config_id": best_plan["target_config"].config_id,
                    },
                }
            )

        planned_actions.append(
            {
                "primitive": "assign_operation",
                "parameters": {
                    "job_id": best_plan["job"].job_id,
                    "op_id": best_plan["operation"].op_id,
                    "machine_id": best_plan["machine"].machine_id,
                    "start_time": best_plan["start_time"],
                },
            }
        )

        return planned_actions
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'environment': self.env.compute_metrics(),
            'memory': self.memory.get_statistics(),
            'llm': self.llm.get_statistics(),
            'executions': len(self.execution_history)
        }