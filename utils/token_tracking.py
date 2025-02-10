from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dataclasses import dataclass
from langfuse import Langfuse
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

@dataclass
class TokenUsageEntry:
    timestamp: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    prompt_id: Optional[str] = None
    cost: float = 0.0
    total_tokens: int = 0
    processing_time: float = 0.0
    processing_speed: float = 0.0

class LangfuseTracker:
   def __init__(self, 
                public_key: Optional[str] = None, 
                secret_key: Optional[str] = None, 
                host: Optional[str] = None):
       # Use environment variables if not explicitly provided
       public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
       secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
       host = host or os.getenv('LANGFUSE_HOST')

       if not public_key or not secret_key:
           raise ValueError("Langfuse public and secret keys must be provided")

       self.langfuse = Langfuse(
           public_key=public_key, 
           secret_key=secret_key, 
           host=host
       )
       self._executor = ThreadPoolExecutor(max_workers=4)

    async def track_usage(self, trace_id: str, entry: TokenUsageEntry) -> None:
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name="token-usage",
                value=entry.total_tokens,
                metadata={
                    'prompt_tokens': entry.prompt_tokens,
                    'completion_tokens': entry.completion_tokens,
                    'model': entry.model,
                    'cost': entry.cost,
                    'processing_time': entry.processing_time,
                    'processing_speed': entry.processing_speed,
                    'timestamp': entry.timestamp
                }
            )
        except Exception as e:
            logger.error(f"Error tracking token usage: {str(e)}", exc_info=True)

class TokenUsageTracker:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TokenUsageTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        if not getattr(self, '_initialized', False):
            self.instance_id = id(self)
            self.logger = logging.getLogger(__name__)
            self.langfuse_tracker = langfuse_tracker
            
            self._usage_timeline = []
            self._total_prompt_tokens = 0
            self._total_completion_tokens = 0
            self._total_tokens = 0
            
            self._cost_rates = {
                'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
                'gpt-4': {'prompt': 0.03, 'completion': 0.06},
                'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03},
                'claude-3-opus': {'prompt': 0.015, 'completion': 0.075},
                'claude-3-sonnet': {'prompt': 0.003, 'completion': 0.015},
                'llama3-70b-8192': {'prompt': 0.0007, 'completion': 0.0007}
            }
            self._initialized = True

    async def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        prompt_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> None:
        try:
            cost = self._calculate_cost(prompt_tokens, completion_tokens, model)
            total_tokens = prompt_tokens + completion_tokens
            processing_time = self._get_processing_time()
            processing_speed = total_tokens / processing_time if processing_time > 0 else 0

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
            
            self._usage_timeline.append(usage_entry)
            self._update_totals(usage_entry)

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_usage(trace_id, usage_entry)

        except Exception as e:
            self.logger.error(f"Error adding token usage: {str(e)}", exc_info=True)
            raise

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str) -> float:
        if model not in self._cost_rates:
            return 0.0
        rates = self._cost_rates[model]
        return ((prompt_tokens / 1000) * rates['prompt'] + 
                (completion_tokens / 1000) * rates['completion'])

    def _get_processing_time(self) -> float:
        return 0.5

    def _update_totals(self, entry: TokenUsageEntry) -> None:
        self._total_prompt_tokens += entry.prompt_tokens
        self._total_completion_tokens += entry.completion_tokens
        self._total_tokens += entry.total_tokens

    def get_usage_stats(self) -> Dict:
        try:
            if not self._usage_timeline:
                return self._empty_stats()

            total_prompt_tokens = sum(entry.prompt_tokens for entry in self._usage_timeline)
            total_completion_tokens = sum(entry.completion_tokens for entry in self._usage_timeline)
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost = sum(entry.cost for entry in self._usage_timeline)
            total_time = sum(entry.processing_time for entry in self._usage_timeline)
            avg_speed = total_tokens / total_time if total_time > 0 else 0

            model_counts = {}
            for entry in self._usage_timeline:
                model_counts[entry.model] = model_counts.get(entry.model, 0) + 1
            most_used_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else 'no_model'

            return {
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

        except Exception as e:
            self.logger.error(f"Error getting usage stats: {str(e)}", exc_info=True)
            raise

    def _empty_stats(self) -> Dict:
        return {
            'model': 'no_model',
            'tokens': {'input': 0, 'output': 0, 'total': 0},
            'processing': {'time': 0, 'speed': 0},
            'cost': 0.0
        }

    def get_prompt_usage(self) -> Dict:
        usage = {}
        for entry in self._usage_timeline:
            if entry.prompt_id:
                if entry.prompt_id not in usage:
                    usage[entry.prompt_id] = {
                        'total_tokens': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_cost': 0
                    }
                stats = usage[entry.prompt_id]
                stats['total_tokens'] += entry.total_tokens
                stats['prompt_tokens'] += entry.prompt_tokens
                stats['completion_tokens'] += entry.completion_tokens
                stats['total_cost'] += entry.cost
        return usage

def create_token_usage_tracker(
   langfuse_public_key: Optional[str] = None,
   langfuse_secret_key: Optional[str] = None,
   langfuse_host: Optional[str] = None
) -> TokenUsageTracker:
   # Use environment variables if not explicitly provided
   langfuse_public_key = langfuse_public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
   langfuse_secret_key = langfuse_secret_key or os.getenv('LANGFUSE_SECRET_KEY')
   langfuse_host = langfuse_host or os.getenv('LANGFUSE_HOST')

   langfuse_tracker = None
   if langfuse_public_key and langfuse_secret_key:
       try:
           langfuse_tracker = LangfuseTracker(
               public_key=langfuse_public_key,
               secret_key=langfuse_secret_key,
               host=langfuse_host
           )
           logger.info("Langfuse integration enabled")
       except Exception as e:
           logger.error(f"Failed to initialize Langfuse: {str(e)}")
   
   return TokenUsageTracker(langfuse_tracker=langfuse_tracker)