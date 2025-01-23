import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from research_agent.tracers import QueryTrace
from utils.token_tracking import TokenUsageTracker

def setup_logging():
    """Set up comprehensive logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('research_agent.log'),
            logging.StreamHandler()
        ]
    )
    
    # Add custom log levels
    logging.addLevelName(logging.INFO, "🔵 INFO")
    logging.addLevelName(logging.WARNING, "🟠 WARNING")
    logging.addLevelName(logging.ERROR, "🔴 ERROR")

def update_token_stats(trace: QueryTrace, prompt_tokens: int, completion_tokens: int, 
                      model: str, prompt_id: Optional[str] = None) -> None:
    """
    Update token usage statistics in QueryTrace
    """
    try:
        # Ensure token tracker is initialized
        if not hasattr(trace, 'token_tracker'):
            trace.token_tracker = TokenUsageTracker()
        
        # Force update token usage
        trace.add_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=model,
            prompt_id=prompt_id
        )
        
        # Debug logging
        print("DEBUG: Updated token usage stats:")
        print(json.dumps(trace.token_tracker.get_usage_stats(), indent=2))
    except Exception as e:
        logging.error(f"Error updating token stats: {str(e)}")

def get_token_usage(trace: QueryTrace) -> Dict[str, Any]:
    """
    Get token usage statistics from a trace
    Args:
        trace: QueryTrace object to get stats from
    Returns:
        Dict containing token usage statistics
    """
    # Always use TokenUsageTracker as source of truth
    if hasattr(trace, 'token_tracker'):
        token_stats = trace.token_tracker.get_usage_stats()
        
        # Add debug logging
        print(f"DEBUG: Getting token usage for trace {trace.trace_id}")
        print(json.dumps(token_stats, indent=2))
        
        return token_stats
    
    # Fallback for older traces or error cases
    logging.warning(f"No TokenUsageTracker found for trace {trace.trace_id}")
    return {
        'total_usage': {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        },
        'usage_by_model': {},
        'usage_by_prompt': {},
        'usage_timeline': []
    }

def load_research_history() -> List[QueryTrace]:
    """Load and process research history into QueryTrace objects"""
    try:
        if not os.path.exists('research_traces.jsonl'):
            return []
            
        traces = []
        with open('research_traces.jsonl', 'r') as f:
            for line in f:
                trace_data = json.loads(line)
                trace = QueryTrace(trace_data['query'])
                trace.data = trace_data
                if 'token_usage' in trace_data:
                    token_usage = trace_data['token_usage']
                    if 'total_usage' in token_usage:
                        trace.token_tracker.prompt_tokens = token_usage['total_usage'].get('prompt_tokens', 0)
                        trace.token_tracker.completion_tokens = token_usage['total_usage'].get('completion_tokens', 0)
                        trace.token_tracker.total_tokens = token_usage['total_usage'].get('total_tokens', 0)
                    trace.token_tracker.usage_by_model = token_usage.get('usage_by_model', {})
                    trace.token_tracker.usage_by_prompt = token_usage.get('usage_by_prompt', {})
                traces.append(trace)
        return traces
    except Exception as e:
        logging.error(f"Error loading traces: {str(e)}")
        return []