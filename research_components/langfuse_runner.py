import os
import logging
from datetime import datetime
import asyncio
from typing import Optional, Dict, Any, Tuple
from langfuse import Langfuse
from dotenv import load_dotenv

from utils.evaluation import create_factual_accuracy_evaluator
from utils.source_coverage import create_source_coverage_evaluator
from utils.logical_coherence import create_logical_coherence_evaluator
from utils.answer_relevance import create_answer_relevance_evaluator
from utils.automated_tests import create_automated_test_evaluator
from utils.analysis_evaluator import create_analysis_evaluator
from utils.token_tracking import create_token_usage_tracker
from tools import GeneralAgent, AnalysisAgent
from utils.token_tracking import TokenUsageTracker

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class LangfuseRunner:
    def __init__(
        self, 
        public_key: Optional[str] = None, 
        secret_key: Optional[str] = None, 
        host: Optional[str] = None
    ):
        # Use environment variables if not explicitly provided
        self.public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
        self.secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
        self.host = host or os.getenv('LANGFUSE_HOST')

        self._validate_config()
        
        # Initialize Langfuse with configuration
        langfuse_config = {
            'public_key': self.public_key,
            'secret_key': self.secret_key
        }
        if self.host:
            langfuse_config['host'] = self.host

        self.langfuse = Langfuse(**langfuse_config)
        
        # Use default or environment variable for model costs
        model_costs = {
            'default': float(os.getenv('DEFAULT_MODEL_COST', 0.001))
        }
        self.token_tracker = TokenUsageTracker(model_costs)
        
        self._setup_evaluators()

    def _validate_config(self):
        if not self.public_key or not self.secret_key:
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")

    def _setup_evaluators(self):
        self.evaluators = {
            "General Agent": {
                "factual": create_factual_accuracy_evaluator(self.langfuse),
                "coverage": create_source_coverage_evaluator(self.langfuse),
                "coherence": create_logical_coherence_evaluator(self.langfuse),
                "relevance": create_answer_relevance_evaluator(self.langfuse),
                "automated": create_automated_test_evaluator(self.langfuse)
            },
            "Analysis Agent": {
                "analysis": create_analysis_evaluator(self.langfuse)
            }
        }

    def run_tool(
        self,
        tool_name: str,
        query: str,
        dataset: Optional[str] = None,
        analysis_type: Optional[str] = None,
        tool: Optional[Any] = None,
        prompt_name: str = "research.txt"
    ) -> Tuple[Any, Dict[str, Any]]:
        trace = self.langfuse.trace(
            name=f"{tool_name.lower().replace(' ', '-')}-execution",
            metadata={
                "tool": tool_name,
                "query": query,
                "dataset": dataset,
                "analysis_type": analysis_type,
                "prompt_name": prompt_name,
                "timestamp": datetime.now().isoformat()
            }
        )

        try:
            start_time = datetime.now()
            generation = trace.generation(name=f"{tool_name.lower()}-generation")
            
            if tool_name == "General Agent":
                result = self._run_general_agent(
                    generation, query, tool, prompt_name
                )
            elif tool_name == "Analysis Agent":
                result = self._run_analysis_agent(
                    generation, query, dataset, tool, prompt_name
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            if result:
                self._track_metrics(trace, tool_name, result, query)
                generation.end(success=True)
            
            trace_data = self._prepare_trace_data(
                start_time=start_time,
                success=bool(result),
                tool_name=tool_name,
                prompt_name=prompt_name
            )
            return result, trace_data

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in run_tool: {error_msg}", exc_info=True)
            trace.error(error_msg)
            generation.end(success=False, error=error_msg)
            return None, {"error": error_msg}

        finally:
            self.langfuse.flush()

    def _run_general_agent(self, generation, query: str, tool: Optional[Any], prompt_name: str):
        tool = tool or GeneralAgent(include_summary=True, prompt_name=prompt_name)
        result = tool.invoke(input={"query": query})
        
        if result:
            content_count = len(result.content) if result.content else 0
            generation.update(
                metadata={
                    "content_count": content_count,
                    "has_summary": bool(result.summary)
                }
            )
        return result

    def _run_analysis_agent(self, generation, query: str, dataset: str, 
                            tool: Optional[Any], prompt_name: str):
        tool = tool or AnalysisAgent(data_folder="./data", prompt_name=prompt_name)
        result = tool.invoke_analysis(input={"query": query, "dataset": dataset})
        
        if result:
            generation.update(
                metadata={
                    "dataset": dataset,
                    "analysis_length": len(result.analysis) if result.analysis else 0
                }
            )
        return result

    def _track_metrics(self, trace, tool_name: str, result: Any, query: str):
        if tool_name == "General Agent":
            for name, evaluator in self.evaluators["General Agent"].items():
                try:
                    if name == "relevance":
                        evaluator.evaluate_answer_relevance(result, query, trace.id)
                    else:
                        evaluator.evaluate(result, trace.id)
                except Exception as e:
                    logger.error(f"Error tracking {name} metric: {str(e)}")
        
        elif tool_name == "Analysis Agent" and result:
            try:
                evaluator = self.evaluators["Analysis Agent"]["analysis"]
                evaluator.evaluate_analysis(result, query, trace.id)
            except Exception as e:
                logger.error(f"Error tracking analysis metric: {str(e)}")

    def _prepare_trace_data(self, start_time: datetime, success: bool, 
                          tool_name: str, prompt_name: str) -> Dict[str, Any]:
        duration = (datetime.now() - start_time).total_seconds()
        usage_stats = self.token_tracker.get_usage_stats()
        
        return {
            "duration": duration,
            "success": success,
            "token_usage": usage_stats,
            "tool": tool_name,
            "prompt_used": prompt_name,
            "timestamp": datetime.now().isoformat()
        }

def create_tool_runner() -> LangfuseRunner:
    return LangfuseRunner()