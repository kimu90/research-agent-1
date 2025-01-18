from datetime import datetime
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any

class QueryTrace:
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())
        self.data = {
            "trace_id": self.trace_id,
            "query": query,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "tools_used": [],
            "prompts_used": [],  # Add this
            "prompt_compilations": [],  # Add this
            "outline": None,
            "dag": None,
            "results": None,
            "final_report": None,
            "end_time": None,
            "duration": None,
            "error": None
        }

    def add_prompt_usage(self, prompt_id: str, agent_type: str, compilation_result: str):
        """Track prompt usage"""
        self.data["prompts_used"].append({
            "prompt_id": prompt_id,
            "agent_type": agent_type,
            "timestamp": datetime.now().isoformat()
        })
        
        self.data["prompt_compilations"].append({
            "prompt_id": prompt_id,
            "agent_type": agent_type,
            "result": compilation_result,
            "timestamp": datetime.now().isoformat()
        })

class TokenUsageTracker:
    """Tracks token usage across different model calls"""
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage_by_model = {}
        self.usage_by_prompt = {}
        
    def add_usage(self, 
                 prompt_tokens: int, 
                 completion_tokens: int, 
                 model_name: str,
                 prompt_id: Optional[str] = None):
        """Add token usage from a model call"""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        
        # Track by model
        if model_name not in self.usage_by_model:
            self.usage_by_model[model_name] = {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            }
        self.usage_by_model[model_name]['prompt_tokens'] += prompt_tokens
        self.usage_by_model[model_name]['completion_tokens'] += completion_tokens
        self.usage_by_model[model_name]['total_tokens'] += (prompt_tokens + completion_tokens)
        
        # Track by prompt if provided
        if prompt_id:
            if prompt_id not in self.usage_by_prompt:
                self.usage_by_prompt[prompt_id] = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                }
            self.usage_by_prompt[prompt_id]['prompt_tokens'] += prompt_tokens
            self.usage_by_prompt[prompt_id]['completion_tokens'] += completion_tokens
            self.usage_by_prompt[prompt_id]['total_tokens'] += (prompt_tokens + completion_tokens)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get comprehensive token usage statistics"""
        return {
            'total_usage': {
                'prompt_tokens': self.prompt_tokens,
                'completion_tokens': self.completion_tokens,
                'total_tokens': self.total_tokens
            },
            'usage_by_model': self.usage_by_model,
            'usage_by_prompt': self.usage_by_prompt
        }

# Update QueryTrace class
class QueryTrace:
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())
        self.token_tracker = TokenUsageTracker()
        self.data = {
            "trace_id": self.trace_id,
            "query": query,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "tools_used": [],
            "prompts_used": [],
            "prompt_compilations": [],
            "token_usage": {
                "total": 0,
                "by_model": {},
                "by_prompt": {},
                "usage_timeline": []
            },
            "outline": None,
            "dag": None,
            "results": None,
            "final_report": None,
            "end_time": None,
            "duration": None,
            "error": None
        }
    
    def add_token_usage(self, 
                       prompt_tokens: int, 
                       completion_tokens: int, 
                       model_name: str,
                       prompt_id: Optional[str] = None):
        """Track token usage for a model call"""
        self.token_tracker.add_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_name=model_name,
            prompt_id=prompt_id
        )
        
        # Update token usage in trace data
        usage_stats = self.token_tracker.get_usage_stats()
        self.data["token_usage"].update(usage_stats)
        
        # Add to timeline
        self.data["token_usage"]["usage_timeline"].append({
            "timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "model": model_name,
            "prompt_id": prompt_id
        })

class CustomTracer:
    def __init__(self):
        self.traces_file = "research_traces.jsonl"
        self.prompt_traces_file = "prompt_traces.jsonl"  

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(trace_id)s] - %(message)s',
            filename='research_detailed.log'
        )
        self.logger = logging.getLogger(__name__)

    
    def log_prompt_usage(self, trace: QueryTrace, prompt_id: str, variables: dict, result: str, metadata: dict):
        """Log prompt usage with variables, result, and metadata"""
        step_data = {
            "step": "prompt_compilation",
            "prompt_id": prompt_id,
            "variables": variables,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        trace.data["steps"].append(step_data)
        trace.add_prompt_usage(prompt_id, result, metadata)
        
        self.logger.info(
            f"Prompt used: {prompt_id}",
            extra={
                'trace_id': trace.trace_id,
                'prompt_details': {
                    'variables': variables,
                    'metadata': metadata
                }
            }
        )

    def save_trace(self, trace: QueryTrace):
        with open(self.traces_file, 'a') as f:
            f.write(json.dumps(trace.data) + '\n')

    def log_step(self, trace: QueryTrace, step_name: str, details: dict = None):
        step_data = {
            "step": step_name,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        trace.data["steps"].append(step_data)
        self.logger.info(
            f"Step: {step_name}",
            extra={
                'trace_id': trace.trace_id,
                'details': details
            }
        )
    def log_model_call(self, 
                      trace: QueryTrace, 
                      model_name: str, 
                      prompt_id: str,
                      usage: Dict[str, int],
                      metadata: Optional[Dict] = None):
        """Log model call with token usage"""
        trace.add_token_usage(
            prompt_tokens=usage.get('prompt_tokens', 0),
            completion_tokens=usage.get('completion_tokens', 0),
            model_name=model_name,
            prompt_id=prompt_id
        )
        
        step_data = {
            "step": "model_call",
            "model": model_name,
            "prompt_id": prompt_id,
            "token_usage": usage,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        trace.data["steps"].append(step_data)
        
        self.logger.info(
            f"Model call: {model_name} - Tokens used: {usage.get('total_tokens', 0)}",
            extra={
                'trace_id': trace.trace_id,
                'model_call_details': {
                    'model': model_name,
                    'prompt_id': prompt_id,
                    'usage': usage
                }
            }
        )