from datetime import datetime
import json
import logging
import time
import uuid
from typing import Dict, List, Optional

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

    
    def log_prompt_usage(self, trace: QueryTrace, prompt_id: str, agent_type: str, 
                        variables: dict, result: str):
        """Log prompt usage with variables and result"""
        step_data = {
            "step": "prompt_compilation",
            "prompt_id": prompt_id,
            "agent_type": agent_type,
            "variables": variables,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        trace.data["steps"].append(step_data)
        trace.add_prompt_usage(prompt_id, agent_type, result)
        
        self.logger.info(
            f"Prompt used: {prompt_id}",
            extra={
                'trace_id': trace.trace_id,
                'prompt_details': {
                    'agent_type': agent_type,
                    'variables': variables
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