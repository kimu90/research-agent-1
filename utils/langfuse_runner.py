import logging
from datetime import datetime
import asyncio
from typing import Optional, Dict, Any, Tuple
from langfuse import Langfuse
from research_agent.config.settings import LANGFUSE_CONFIG, MODEL_COSTS
from research_agent.evaluators import (
    create_factual_accuracy_evaluator,
    create_source_coverage_evaluator,
    create_logical_coherence_evaluator,
    create_answer_relevance_evaluator,
    create_automated_test_evaluator,
    create_analysis_evaluator
)
from research_agent.tools import GeneralAgent, AnalysisAgent
from research_agent.tracers.token_tracker import TokenUsageTracker

logger = logging.getLogger(__name__)

class LangfuseRunner:
    def __init__(self):
        self._validate_config()
        self.langfuse = Langfuse(**LANGFUSE_CONFIG)
        self.token_tracker = TokenUsageTracker(MODEL_COSTS)
        self._setup_evaluators()

    def _validate_config(self):
        if not LANGFUSE_CONFIG['public_key'] or not LANGFUSE_CONFIG['secret_key']:
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

    async def run_tool(
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
                result = await self._run_general_agent(
                    generation, query, tool, prompt_name
                )
            elif tool_name == "Analysis Agent":
                result = await self._run_analysis_agent(
                    generation, query, dataset, tool, prompt_name
                )
            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            if result:
                await self._track_metrics(trace, tool_name, result, query)
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

    async def _run_general_agent(self, generation, query: str, tool: Optional[Any], prompt_name: str):
        tool = tool or GeneralAgent(include_summary=True, prompt_name=prompt_name)
        result = await tool.invoke(input={"query": query})
        
        if result:
            content_count = len(result.content) if result.content else 0
            generation.update(
                metadata={
                    "content_count": content_count,
                    "has_summary": bool(result.summary)
                }
            )
        return result

    async def _run_analysis_agent(self, generation, query: str, dataset: str, 
                                tool: Optional[Any], prompt_name: str):
        tool = tool or AnalysisAgent(data_folder="./data", prompt_name=prompt_name)
        result = await tool.invoke_analysis(input={"query": query, "dataset": dataset})
        
        if result:
            generation.update(
                metadata={
                    "dataset": dataset,
                    "analysis_length": len(result.analysis) if result.analysis else 0
                }
            )
        return result

    async def _track_metrics(self, trace, tool_name: str, result: Any, query: str):
        evaluation_tasks = []
        
        if tool_name == "General Agent":
            for name, evaluator in self.evaluators["General Agent"].items():
                if name == "relevance":
                    evaluation_tasks.append(
                        evaluator.evaluate_answer_relevance(result, query, trace.id)
                    )
                else:
                    evaluation_tasks.append(
                        evaluator.evaluate(result, trace.id)
                    )
        
        elif tool_name == "Analysis Agent" and result:
            evaluator = self.evaluators["Analysis Agent"]["analysis"]
            evaluation_tasks.append(
                evaluator.evaluate_analysis(result, query, trace.id)
            )

        await asyncio.gather(*evaluation_tasks)

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