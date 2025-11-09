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
            
            # Generate LLM response
            llm_response = self.llm.generate_response(
                user_message=initial_prompt if iteration == 0 else "Continue optimization based on current state.",
                state=current_state,
                similar_episodes=similar_episodes,
                primitive_descriptions=self.primitives.get_primitive_descriptions()
            )

            actions: List[Dict[str, Any]]
            if not llm_response['success']:
                logger.error(
                    "LLM generation failed: %s", llm_response.get('error')
                )
                actions = self._generate_default_actions(current_state)
                if not actions:
                    logger.error("No fallback actions available - stopping optimisation loop")
                    break
                logger.info(
                    "Using heuristic fallback actions because the LLM request did not succeed"
                )
            else:
                # Parse actions from response
                actions = self.llm.parse_action_from_response(llm_response['response'])

                if not actions:
                    logger.warning("No actions parsed from LLM response")
                    # Try to schedule remaining jobs with default strategy
                    actions = self._generate_default_actions(current_state)

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
        """Generate default actions when LLM doesn't provide any"""
        actions = []
        
        # Simple FIFO scheduling
        job_queue = state['job_queue']
        if not job_queue:
            return actions
        
        # Get first job
        job = job_queue[0]
        
        # Find next unscheduled operation
        for machine in state['machines']:
            if machine['state'] == 'idle':
                # Try to assign an operation
                actions.append({
                    'primitive': 'assign_operation',
                    'parameters': {
                        'job_id': job['job_id'],
                        'op_id': 0,
                        'machine_id': machine['machine_id'],
                        'start_time': state['current_time']
                    }
                })
                break
        
        return actions
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'environment': self.env.compute_metrics(),
            'memory': self.memory.get_statistics(),
            'llm': self.llm.get_statistics(),
            'executions': len(self.execution_history)
        }