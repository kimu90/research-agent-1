from datetime import datetime
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from utils.token_tracking import TokenUsageTracker

class QueryTrace:
    _token_tracker = None  # Class level token tracker
    
    def __init__(self, query: str):
        self.trace_id = str(uuid.uuid4())
        if QueryTrace._token_tracker is None:
            QueryTrace._token_tracker = TokenUsageTracker()
        self.token_tracker = QueryTrace._token_tracker
        self.timestamp = datetime.now()  # Add timestamp as instance attribute
        self.data = {
            "trace_id": self.trace_id,
            "query": query,
            "start_time": self.timestamp.isoformat(),
            "steps": [],
            "timestamp": self.timestamp.isoformat(),  # Use the same timestamp
            "tools_used": [],
            "prompts_used": [],
            "prompt_compilations": [],
            "token_usage": None,  
            "dag": None,
            "results": None,
            "final_report": None,
            "end_time": None,
            "duration": None,
            "error": None,
            "processing_steps": []  # Added missing required key
        }

    def add_prompt_usage(self, prompt_id: str, agent_type: str, compilation_result: str):
        """Track prompt usage"""
        current_time = datetime.now().isoformat()
        self.data["prompts_used"].append({
            "prompt_id": prompt_id,
            "agent_type": agent_type,
            "timestamp": current_time
        })
        
        self.data["prompt_compilations"].append({
            "prompt_id": prompt_id,
            "agent_type": agent_type,
            "result": compilation_result,
            "timestamp": current_time
        })
    
    def add_token_usage(self,
                    prompt_tokens: int,
                    completion_tokens: int,
                    model_name: str,
                    prompt_id: Optional[str] = None):
        """Track token usage for a model call"""
        logger = logging.getLogger(__name__)
        logger.info(f"Adding token usage - Model: {model_name}, Prompt ID: {prompt_id or 'N/A'}")
        
        try:
            # Let TokenUsageTracker handle everything
            self.token_tracker.add_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=model_name,
                prompt_id=prompt_id
            )
            logger.debug(f"Token usage entry added successfully - Total tokens: {prompt_tokens + completion_tokens}")

        except Exception as e:
            logger.error(f"Error adding token usage: {str(e)}", exc_info=True)
            logger.error(f"Failed parameters - Model: {model_name}, "
                        f"Prompt tokens: {prompt_tokens}, "
                        f"Completion tokens: {completion_tokens}")
            raise

    def finalize(self):
        """Finalize the trace by setting end time and duration"""
        self.data["end_time"] = datetime.now().isoformat()
        start_time = datetime.fromisoformat(self.data["start_time"])
        end_time = datetime.fromisoformat(self.data["end_time"])
        self.data["duration"] = (end_time - start_time).total_seconds()


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
        trace.add_prompt_usage(prompt_id, "unknown", result)
        
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
        """Save the trace with updated token usage statistics"""
        logger = logging.getLogger(__name__)
        logger.info(f"Starting trace save operation - Trace ID: {trace.trace_id}")
        
        try:
            # Finalize the trace before saving
            trace.finalize()
            
            # Get token stats from TokenUsageTracker
            token_stats = trace.token_tracker.get_usage_stats()
            logger.debug(f"Retrieved final token stats: {json.dumps(token_stats, indent=2)}")
            
            # Set the complete token usage stats in trace data
            trace.data["token_usage"] = token_stats
            
            # Validate trace data before saving
            logger.info("Validating trace data before save")
            required_keys = ["token_usage", "tools_used", "processing_steps"]
            missing_keys = [key for key in required_keys if key not in trace.data]
            if missing_keys:
                logger.warning(f"Missing required keys in trace data: {missing_keys}")
            
            # Calculate and log data size
            trace_json = json.dumps(trace.data)
            data_size = len(trace_json)
            logger.info(f"Trace data size: {data_size:,d} bytes")
            
            # Save the trace
            with open(self.traces_file, 'a') as f:
                f.write(trace_json + '\n')
            
            # Log success with key metrics
            logger.info(f"Trace successfully saved - Total tokens: {token_stats['tokens']['total']:,d}")
            logger.info(f"Processing time: {token_stats['processing']['time']:.2f}s, "
                    f"Speed: {token_stats['processing']['speed']:.2f} tokens/second")
            logger.info(f"Total cost: ${token_stats['cost']:.6f}")
            
            # Log detailed final statistics
            logger.debug("Final trace statistics:")
            logger.debug(f"- Model used: {token_stats['model']}")
            logger.debug(f"- Input tokens: {token_stats['tokens']['input']:,d}")
            logger.debug(f"- Output tokens: {token_stats['tokens']['output']:,d}")
            logger.debug(f"- Processing time: {token_stats['processing']['time']:.2f}s")
            logger.debug(f"- Processing speed: {token_stats['processing']['speed']:.2f} tokens/second")
            
        except Exception as e:
            # Enhanced error logging
            logger.error(f"Error saving trace: {str(e)}", exc_info=True)
            logger.error("Error context:")
            logger.error(f"- Trace file path: {self.traces_file}")
            logger.error(f"- Token stats available: {bool(token_stats) if 'token_stats' in locals() else False}")
            
            # Log the trace data state if possible
            try:
                logger.error("Trace data at time of error:")
                logger.error(json.dumps(trace.data, indent=2))
            except Exception as json_error:
                logger.error(f"Could not serialize trace data: {str(json_error)}")
            
            raise

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
        pass