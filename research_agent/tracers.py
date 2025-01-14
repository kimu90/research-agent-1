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
            "outline": None,
            "dag": None,
            "results": None,
            "final_report": None,
            "end_time": None,
            "duration": None,
            "error": None
        }

class CustomTracer:
    def __init__(self):
        self.traces_file = "research_traces.jsonl"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - [%(trace_id)s] - %(message)s',
            filename='research_detailed.log'
        )
        self.logger = logging.getLogger(__name__)

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