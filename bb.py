import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from langfuse import Langfuse
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from dotenv import load_dotenv

# Your existing EVAL_TYPES and setup_logging() remain the same...

# Add prompt configurations
PROMPT_CONFIGS = {
    "general": {
        "template": """Analyze the dataset with a focus on {{focus_area}}. 
        Consider the key patterns, trends, and insights that emerge from the data. 
        Provide a comprehensive analysis that highlights significant findings.""",
        "config": {"model": "gpt-4", "temperature": 0.7},
        "variants": {
            "concise": "Provide a brief analysis of {{dataset}} focusing on {{focus_area}}",
            "detailed": "Perform an in-depth analysis of {{dataset}}, examining {{focus_area}}"
        }
    },
    "detailed": {
        "template": """Perform a detailed analysis of {{dataset}} focusing on {{metrics}}. 
        Include statistical measures, outliers, and relationships between variables.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "variants": {
            "technical": "Technical analysis of {{dataset}} with focus on {{metrics}}",
            "business": "Business-focused analysis of {{dataset}} examining {{metrics}}"
        }
    },
    "comparative": {
        "template": """Compare the following aspects of {{dataset}}: {{aspects}}. 
        Analyze similarities, differences, and relationships.""",
        "config": {"model": "gpt-4", "temperature": 0.3},
        "variants": {
            "simple": "Basic comparison of {{aspects}} in {{dataset}}",
            "complex": "Detailed comparison with statistical analysis of {{aspects}} in {{dataset}}"
        }
    }
}

class LangfuseRunner:
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        eval_model: str = "gpt-3.5-turbo-instruct"
    ):
        logger.info("=== Initializing LangfuseRunner ===")
        
        self.public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
        self.secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
        self.host = host or os.getenv('LANGFUSE_HOST')
        self.eval_model = eval_model
        
        try:
            self._validate_config()
            self._initialize_langfuse()
            self._initialize_evaluators()
            self._initialize_prompts()  # New method for prompt initialization
            logger.info("Initialization complete")
            print("✓ LangfuseRunner initialized successfully")
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

    def _initialize_prompts(self):
        """Initialize prompt templates in Langfuse"""
        try:
            self.prompts = {}
            for analysis_type, config in PROMPT_CONFIGS.items():
                # Create base prompt
                prompt = self.langfuse.create_prompt(
                    name=f"{analysis_type}-analysis",
                    prompt=config["template"],
                    config=config["config"],
                    labels=["production", analysis_type]
                )
                self.prompts[analysis_type] = prompt

                # Create variants
                for variant_name, variant_template in config["variants"].items():
                    self.langfuse.create_prompt(
                        name=f"{analysis_type}-{variant_name}",
                        prompt=variant_template,
                        config=config["config"],
                        labels=["variant", analysis_type, variant_name]
                    )
            
            logger.info(f"Initialized prompts for types: {list(self.prompts.keys())}")
        except Exception as e:
            error_msg = f"Failed to initialize prompts: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def get_prompt(self, analysis_type: str, version: str = None, variant: str = None):
        """Get appropriate prompt based on analysis type and version"""
        try:
            base_prompt = self.prompts.get(analysis_type, self.prompts["general"])
            
            if version:
                return self.langfuse.get_prompt_version(base_prompt.name, version)
            elif variant:
                variant_name = f"{analysis_type}-{variant}"
                return self.langfuse.get_prompt(variant_name)
            
            return base_prompt
        except Exception as e:
            logger.error(f"Error getting prompt: {str(e)}")
            return self.prompts["general"]

    def run_tool(
        self,
        tool_name: str,
        query: str,
        dataset: Optional[str] = None,
        analysis_type: Optional[str] = None,
        tool: Optional[Any] = None,
        prompt_name: str = "research.txt",
        prompt_version: Optional[str] = None,
        prompt_variant: Optional[str] = None,
        run_evaluation: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute a tool and optionally evaluate its output with prompt management"""
        logger.info(f"=== Starting tool execution: {tool_name} ===")
        print(f"\nExecuting {tool_name}...")

        start_time = datetime.now()
        
        # Get appropriate prompt
        prompt = self.get_prompt(
            analysis_type=analysis_type or "general",
            version=prompt_version,
            variant=prompt_variant
        )

        trace = self.langfuse.trace(
            name=f"{tool_name.lower().replace(' ', '-')}-execution",
            metadata={
                "tool": tool_name,
                "query": query,
                "dataset": dataset,
                "analysis_type": analysis_type,
                "prompt_name": prompt.name,
                "prompt_version": prompt_version,
                "prompt_variant": prompt_variant,
                "timestamp": datetime.now().isoformat()
            }
        )

        try:
            generation = trace.generation(
                name=f"{tool_name.lower()}-generation",
                metadata={
                    "prompt_used": prompt.name,
                    "prompt_version": prompt_version,
                    "prompt_variant": prompt_variant
                }
            )
            
            # Track prompt usage
            self.langfuse.track_prompt_usage(
                prompt_name=prompt.name,
                version=prompt_version,
                variant=prompt_variant,
                metadata={
                    "tool": tool_name,
                    "analysis_type": analysis_type,
                    "dataset": dataset
                }
            )

            if tool_name == "Analysis Agent":
                from tools import AnalysisAgent
                # Initialize AnalysisAgent with Langfuse prompt management
                tool = tool or AnalysisAgent(
                    langfuse_client=self.langfuse,  # Pass the langfuse client from LangfuseRunner
                    data_folder="./data",
                    prompt_name=prompt_name,
                    prompt_version=prompt_version,
                    prompt_variant=prompt_variant
                )
                result = tool.invoke_analysis(
                    input={"query": query, "dataset": dataset}
                )
            else:
                from tools import GeneralAgent
                tool = tool or GeneralAgent(
                    include_summary=True, 
                    prompt_name=prompt.name,
                    prompt_config=prompt.config
                )
                result = tool.invoke(input={"query": query})

            evaluation_results = {}

            if result:
                if run_evaluation and hasattr(result, 'analysis'):
                    evaluation_results = self._evaluate_generation(
                        generation=generation,
                        input_text=query,
                        output_text=str(result.analysis)
                    )
                            
                generation.end(
                    success=True,
                    metadata={
                        "prompt_performance": evaluation_results
                    }
                )
                logger.info("Tool execution completed successfully")
                print("✓ Tool execution successful")
            else:
                logger.warning("Tool execution completed but returned no results")
                print("⚠ Tool execution completed with no results")

            trace_data = self._prepare_trace_data(
                start_time=start_time,
                success=bool(result),
                tool_name=tool_name,
                prompt_info={
                    "name": prompt.name,
                    "version": prompt_version,
                    "variant": prompt_variant,
                    "config": prompt.config
                }
            )
            
            trace_data['evaluation_results'] = evaluation_results
            
            return result, trace_data

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in run_tool: {error_msg}", exc_info=True)
            print(f"ERROR: Tool execution failed - {error_msg}")
            
            trace.update(status="ERROR")
            
            generation.end(
                success=False, 
                error_message=error_msg,
                metadata={
                    "prompt_name": prompt.name,
                    "prompt_version": prompt_version,
                    "prompt_variant": prompt_variant
                }
            )
            
            return None, {"error": error_msg}

        finally:
            logger.debug("Flushing Langfuse traces")
            self.langfuse.flush()

    def _prepare_trace_data(
        self,
        start_time: datetime,
        success: bool,
        tool_name: str,
        prompt_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare trace data with execution metrics and prompt information"""
        duration = (datetime.now() - start_time).total_seconds()
        
        trace_data = {
            "duration": duration,
            "success": success,
            "tool": tool_name,
            "prompt": prompt_info,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Prepared trace data: {trace_data}")
        print(f"Execution time: {duration:.2f} seconds")
        return trace_data

def create_tool_runner(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None
) -> LangfuseRunner:
    """Create a new LangfuseRunner instance with prompt management"""
    logger.info("Creating new LangfuseRunner instance")
    print("\nInitializing new LangfuseRunner...")
    return LangfuseRunner(public_key=public_key, secret_key=secret_key, host=host)