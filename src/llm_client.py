"""
LLM Reasoning Engine
Interfaces with Claude or other LLMs
"""

import os
from typing import List, Dict, Any, Optional
import anthropic
import json
import logging
import time

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM-based reasoning"""
    
    def __init__(self, 
                 model: str = "claude-sonnet-4-20250514",
                 api_key: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 8192):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize Anthropic client
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Conversation history
        self.conversation_history = []
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'avg_response_time': 0.0
        }
    
    def create_system_prompt(self, primitive_descriptions: Dict[str, str]) -> str:
        """Create system prompt with primitive descriptions"""
        prompt = """You are an expert in manufacturing optimization and scheduling, specifically for Reconfigurable Manufacturing Systems (RMS). Your role is to optimize production schedules by intelligently composing primitive operations.

Available Primitives:
"""
        for name, description in primitive_descriptions.items():
            prompt += f"\n- {name}: {description}"
        
        prompt += """

Your Objectives:
1. Minimize makespan (total completion time)
2. Minimize tardiness (late job penalties)
3. Minimize energy consumption
4. Minimize unnecessary reconfigurations

Decision-Making Process:
1. Analyze the current system state
2. Query memory for similar past situations
3. Reason about bottlenecks and constraints
4. Select and compose primitives strategically
5. Record outcomes for future learning

Always explain your reasoning before taking actions. Use past experiences to make better decisions."""

        return prompt
    
    def format_state_for_llm(self, state: Dict) -> str:
        """Format system state for LLM consumption"""
        formatted = "## Current System State\n\n"
        
        # Time and jobs
        formatted += f"**Current Time:** {state['current_time']:.2f}\n"
        formatted += f"**Pending Jobs:** {state['pending_jobs']}\n"
        formatted += f"**Completed Jobs:** {state['completed_jobs']}\n\n"
        
        # Machines
        formatted += "### Machines Status\n"
        for machine in state['machines'][:5]:  # Show first 5 machines
            formatted += f"- Machine {machine['machine_id']}: "
            formatted += f"State={machine['state']}, "
            formatted += f"Config={machine['current_config']}, "
            formatted += f"Util={machine['utilization']:.1%}, "
            formatted += f"Next Available={machine['next_available']:.2f}\n"
        
        if len(state['machines']) > 5:
            formatted += f"... and {len(state['machines']) - 5} more machines\n"
        
        # Job queue
        formatted += "\n### Job Queue (Top 10)\n"
        for job in state['job_queue'][:10]:
            formatted += f"- Job {job['job_id']}: "
            formatted += f"Priority={job['priority']}, "
            formatted += f"Remaining Ops={job['remaining_operations']}, "
            formatted += f"Due={job['due_date']:.2f}, "
            formatted += f"Slack={job['slack']:.2f}\n"
        
        if len(state['job_queue']) > 10:
            formatted += f"... and {len(state['job_queue']) - 10} more jobs\n"
        
        # Metrics
        formatted += "\n### Performance Metrics\n"
        metrics = state['metrics']
        formatted += f"- Makespan: {metrics['makespan']:.2f}\n"
        formatted += f"- Avg Tardiness: {metrics['avg_tardiness']:.2f}\n"
        formatted += f"- Utilization: {metrics['utilization']:.1%}\n"
        formatted += f"- Energy: {metrics['energy_consumption']:.2f}\n"
        formatted += f"- Reconfigurations: {metrics['reconfigurations']}\n"
        
        return formatted
    
    def format_similar_episodes(self, episodes: List[Dict]) -> str:
        """Format similar past episodes"""
        if not episodes:
            return "No similar past episodes found.\n"
        
        formatted = "## Similar Past Episodes\n\n"
        
        for i, ep in enumerate(episodes[:3], 1):  # Show top 3
            formatted += f"### Episode {ep['episode_id']} (Distance: {ep['distance']:.3f})\n"
            formatted += f"**Reward:** {ep['reward']:.2f}\n"
            formatted += f"**Action Taken:** {ep['action'].get('type', 'unknown')}\n"
            
            # Key metrics from that episode
            if 'state' in ep and 'metrics' in ep['state']:
                metrics = ep['state']['metrics']
                formatted += f"- Makespan: {metrics.get('makespan', 0):.2f}\n"
                formatted += f"- Tardiness: {metrics.get('avg_tardiness', 0):.2f}\n"
            
            formatted += "\n"
        
        return formatted
    
    def generate_response(self, 
                         user_message: str,
                         state: Dict,
                         similar_episodes: List[Dict] = None,
                         primitive_descriptions: Dict = None) -> Dict:
        """Generate LLM response"""
        start_time = time.time()
        
        # Build message
        formatted_state = self.format_state_for_llm(state)
        
        full_message = f"{user_message}\n\n{formatted_state}"
        
        if similar_episodes:
            full_message += "\n" + self.format_similar_episodes(similar_episodes)
        
        # Create messages
        messages = self.conversation_history + [
            {"role": "user", "content": full_message}
        ]
        
        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.create_system_prompt(primitive_descriptions) if primitive_descriptions else None,
                messages=messages
            )
            
            # Extract response
            response_text = response.content[0].text
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": full_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Update statistics
            elapsed = time.time() - start_time
            self.stats['total_requests'] += 1
            self.stats['total_tokens'] += response.usage.input_tokens + response.usage.output_tokens
            self.stats['avg_response_time'] = (
                (self.stats['avg_response_time'] * (self.stats['total_requests'] - 1) + elapsed)
                / self.stats['total_requests']
            )
            
            logger.info(f"LLM response generated in {elapsed:.2f}s")
            
            return {
                'success': True,
                'response': response_text,
                'tokens': response.usage.input_tokens + response.usage.output_tokens,
                'time': elapsed
            }
            
        except Exception as e:
            logger.error(f"LLM API error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'response': None
            }
    
    def parse_action_from_response(self, response_text: str) -> List[Dict]:
        """Parse primitive actions from LLM response"""
        actions = []
        
        # Look for action patterns in response
        # Format: ACTION: primitive_name(param1=value1, param2=value2)
        
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('ACTION:') or line.startswith('PRIMITIVE:'):
                # Extract action
                action_str = line.split(':', 1)[1].strip()
                
                try:
                    # Parse primitive call
                    if '(' in action_str and ')' in action_str:
                        prim_name = action_str[:action_str.index('(')].strip()
                        params_str = action_str[action_str.index('(') + 1:action_str.rindex(')')]
                        
                        # Parse parameters
                        params = {}
                        if params_str:
                            for param in params_str.split(','):
                                if '=' in param:
                                    key, value = param.split('=', 1)
                                    key = key.strip()
                                    value = value.strip()
                                    
                                    # Convert value type
                                    try:
                                        if value.lower() == 'true':
                                            value = True
                                        elif value.lower() == 'false':
                                            value = False
                                        elif value.isdigit():
                                            value = int(value)
                                        elif '.' in value:
                                            value = float(value)
                                        elif value.startswith('[') and value.endswith(']'):
                                            value = json.loads(value)
                                        elif value.startswith('"') or value.startswith("'"):
                                            value = value.strip('"\'')
                                    except:
                                        pass
                                    
                                    params[key] = value
                        
                        actions.append({
                            'primitive': prim_name,
                            'parameters': params
                        })
                
                except Exception as e:
                    logger.warning(f"Failed to parse action: {action_str}, Error: {e}")
        
        return actions
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_statistics(self) -> Dict:
        """Get LLM usage statistics"""
        return self.stats.copy()