import pdb
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from dataclasses import dataclass
import time

@dataclass
class TokenUsageEntry:
    """Represents a single token usage entry"""
    timestamp: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    prompt_id: Optional[str] = None
    cost: float = 0.0
    total_tokens: int = 0
    processing_time: float = 0.0
    processing_speed: float = 0.0

class TokenUsageTracker:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TokenUsageTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.instance_id = id(self)
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Creating TokenUsageTracker singleton instance {self.instance_id}")
            
            self._usage_timeline = []
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_tokens = 0
            self._model_usage = {}
            
            self._cost_rates = {
                'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
                'gpt-4': {'prompt': 0.03, 'completion': 0.06},
                'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
                'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
                'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
                'llama3-70b-8192': {'prompt': 0.0007, 'completion': 0.0007}
            }
            self._initialized = True
    def _get_processing_time(self) -> float:
        """Simulate processing time based on token count"""
        return 0.5  # Simulated processing time in seconds

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        prompt_id: Optional[str] = None
    ) -> None:
        """Add a new token usage entry"""
        self.logger.info(f"Adding usage to instance {self.instance_id}")
        self.logger.info(f"Adding new token usage entry - Model: {model}, Prompt ID: {prompt_id or 'N/A'}")
        
        try:
            # Calculate cost
            cost = 0.0
            if model in self._cost_rates:
                self.logger.info(f"Calculating costs using rates for model {model}")
                rates = self._cost_rates[model]
                prompt_cost = (prompt_tokens / 1000) * rates['prompt']
                completion_cost = (completion_tokens / 1000) * rates['completion']
                cost = prompt_cost + completion_cost
                self.logger.info(f"Calculated costs - Prompt: ${prompt_cost:.6f}, Completion: ${completion_cost:.6f}, Total: ${cost:.6f}")
            else:
                self.logger.warning(f"No cost rates available for model {model}")

            # Calculate metrics
            total_tokens = prompt_tokens + completion_tokens
            processing_time = self._get_processing_time()
            processing_speed = total_tokens / processing_time if processing_time > 0 else 0

            self.logger.info(f"Starting inference with model {model} at {datetime.now().isoformat()}")
            self.logger.info(f"Inference completed in {processing_time:.2f}s")
            self.logger.info(f"Tokens used - Input: {prompt_tokens:,d}, Output: {completion_tokens:,d}, Total: {total_tokens:,d}")
            self.logger.info(f"Processing speed - {processing_speed:.2f} tokens/second")
            
            # Create and store usage entry
            usage_entry = TokenUsageEntry(
                timestamp=datetime.now().isoformat(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model,
                prompt_id=prompt_id,
                cost=cost,
                total_tokens=total_tokens,
                processing_time=processing_time,
                processing_speed=processing_speed
            )
            
            self.logger.debug(f"Created usage entry: {usage_entry}")
            
            # Store entry and update stats
            self._usage_timeline.append(usage_entry)
            self.logger.info(f"Usage timeline updated - Current entries: {len(self._usage_timeline)} for instance {self.instance_id}")
            
            # Update running totals
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_tokens += total_tokens
            
            self.logger.info(f"Cumulative statistics updated - Total tokens: {self._total_tokens:,d}")
            self.logger.debug(f"Timeline state after add: {[e.model for e in self._usage_timeline]}")

        except Exception as e:
            self.logger.error(f"Error adding token usage: {str(e)}", exc_info=True)
            self.logger.error(f"Failed parameters - Model: {model}, Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
            raise

    def get_usage_stats(self) -> Dict:
        """Get comprehensive usage statistics"""
        self.logger.info(f"Getting stats from instance {self.instance_id}")
        self.logger.debug(f"Current timeline state: {[e.model for e in self._usage_timeline]}")
        
        try:
            if not self._usage_timeline:
                self.logger.warning(f"No usage timeline found for instance {self.instance_id} - returning default stats")
                return {
                    'model': 'no_model',
                    'tokens': {
                        'input': 0,
                        'output': 0,
                        'total': 0
                    },
                    'processing': {
                        'time': 0,
                        'speed': 0
                    },
                    'cost': 0.0
                }

            self.logger.info(f"Processing {len(self._usage_timeline)} entries for instance {self.instance_id}")
            
            # Calculate totals from timeline
            total_prompt_tokens = sum(entry.prompt_tokens for entry in self._usage_timeline)
            total_completion_tokens = sum(entry.completion_tokens for entry in self._usage_timeline)
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost = sum(entry.cost for entry in self._usage_timeline)
            total_time = sum(entry.processing_time for entry in self._usage_timeline)
            
            # Calculate speed
            avg_speed = total_tokens / total_time if total_time > 0 else 0

            # Get most used model
            model_counts = {}
            for entry in self._usage_timeline:
                model_counts[entry.model] = model_counts.get(entry.model, 0) + 1
            
            self.logger.debug(f"Model usage distribution: {model_counts}")
            most_used_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else 'no_model'
            
            stats = {
                'model': most_used_model,
                'tokens': {
                    'input': total_prompt_tokens,
                    'output': total_completion_tokens,
                    'total': total_tokens
                },
                'processing': {
                    'time': total_time,
                    'speed': avg_speed
                },
                'cost': total_cost
            }
            
            self.logger.info(f"Final statistics compiled - Total tokens: {total_tokens:,d}, Total cost: ${total_cost:.6f}")
            return stats

        except Exception as e:
            self.logger.error(f"Error getting usage stats for instance {self.instance_id}: {str(e)}", exc_info=True)
            raise

    def _get_prompt_usage(self) -> Dict:
        """Aggregate token usage by prompt ID"""
        prompt_usage = {}
        for entry in self._usage_timeline:
            if entry.prompt_id:
                if entry.prompt_id not in prompt_usage:
                    prompt_usage[entry.prompt_id] = {
                        'total_tokens': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_cost': 0
                    }
                
                prompt_usage[entry.prompt_id]['total_tokens'] += entry.total_tokens
                prompt_usage[entry.prompt_id]['prompt_tokens'] += entry.prompt_tokens
                prompt_usage[entry.prompt_id]['completion_tokens'] += entry.completion_tokens
                prompt_usage[entry.prompt_id]['total_cost'] += entry.cost
        
        return prompt_usage

    def get_total_usage(self) -> Dict:
        """Get total usage across all interactions"""
        return {
            'total_tokens': self._total_tokens,
            'total_prompt_tokens': self._total_prompt_tokens,
            'total_completion_tokens': self._total_completion_tokens
        }