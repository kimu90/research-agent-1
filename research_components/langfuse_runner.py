import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from langfuse import Langfuse
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from dotenv import load_dotenv

# Evaluation types for LLM outputs
EVAL_TYPES = {
    "hallucination": True,
    "conciseness": True,
    "relevance": True,
    "coherence": True,
    "harmfulness": True,
    "helpfulness": True,
    "maliciousness": True,
    "controversiality": True,
    "misogyny": True,
    "criminality": True,
    "insensitivity": True
}

def setup_logging():
    """Configure logging with a custom format and multiple handlers"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    formatter = logging.Formatter(log_format, date_format)
    
    file_handler = logging.FileHandler('langfuse_runner.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logger = logging.getLogger('LangfuseRunner')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
        
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("Logging system initialized")
    logger.debug("Debug logging enabled")
    
    return logger

logger = setup_logging()

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
        
        logger.debug(f"Configuration loaded - Host: {self.host}")
        logger.debug(f"Public Key present: {'Yes' if self.public_key else 'No'}")
        logger.debug(f"Secret Key present: {'Yes' if self.secret_key else 'No'}")

        try:
            self._validate_config()
            self._initialize_langfuse()
            self._initialize_evaluators()
            logger.info("Initialization complete")
            print("✓ LangfuseRunner initialized successfully")
        except Exception as e:
            error_msg = f"Initialization failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(f"ERROR: {error_msg}")
            raise

        # Model costs configuration
        self.model_costs = {
            'default': float(os.getenv('DEFAULT_MODEL_COST', 0.001))
        }
        logger.debug(f"Model costs configured: {self.model_costs}")

    def _validate_config(self):
        """Validate the configuration settings"""
        logger.debug("Starting configuration validation")
        missing_vars = []
        
        if not self.public_key:
            missing_vars.append("LANGFUSE_PUBLIC_KEY")
        if not self.secret_key:
            missing_vars.append("LANGFUSE_SECRET_KEY")
            
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _initialize_langfuse(self):
        """Initialize Langfuse with configuration"""
        try:
            langfuse_config = {
                'public_key': self.public_key,
                'secret_key': self.secret_key
            }
            if self.host:
                langfuse_config['host'] = self.host
            
            self.langfuse = Langfuse(**langfuse_config)
            logger.info("Successfully initialized Langfuse client")
        except Exception as e:
            error_msg = f"Failed to initialize Langfuse client: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def _initialize_evaluators(self):
        """Initialize evaluators for different criteria"""
        try:
            llm = OpenAI(temperature=0, model=self.eval_model)
            
            self.evaluators = {}
            # Initialize standard evaluators
            for criterion, enabled in EVAL_TYPES.items():
                if criterion != "hallucination" and enabled:
                    self.evaluators[criterion] = load_evaluator("criteria", criteria=criterion, llm=llm)
            
            # Special case for hallucination
            if EVAL_TYPES.get("hallucination"):
                criteria = {
                    "hallucination": "Does this submission contain information not present in the input or reference?"
                }
                self.evaluators["hallucination"] = LabeledCriteriaEvalChain.from_llm(
                    llm=llm,
                    criteria=criteria
                )
            
            logger.info(f"Initialized evaluators: {list(self.evaluators.keys())}")
        except Exception as e:
            error_msg = f"Failed to initialize evaluators: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

    def run_tool(
        self,
        tool_name: str,
        query: str,
        dataset: Optional[str] = None,
        analysis_type: Optional[str] = None,
        tool: Optional[Any] = None,
        prompt_name: str = "research.txt",
        run_evaluation: bool = True
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute a tool and optionally evaluate its output"""
        logger.info(f"=== Starting tool execution: {tool_name} ===")
        print(f"\nExecuting {tool_name}...")

        start_time = datetime.now()
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
            generation = trace.generation(name=f"{tool_name.lower()}-generation")
            
            if tool_name == "Analysis Agent":
                from tools import AnalysisAgent
                tool = tool or AnalysisAgent(data_folder="./data", prompt_name=prompt_name)
                result = tool.invoke_analysis(
                    input={"query": query, "dataset": dataset}
                )
            else:
                from tools import GeneralAgent
                tool = tool or GeneralAgent(include_summary=True, prompt_name=prompt_name)
                result = tool.invoke(input={"query": query})

            # Initialize evaluation_results to avoid UnboundLocalError
            evaluation_results = {}

            if result:
                if run_evaluation and hasattr(result, 'analysis'):
                    evaluation_results = self._evaluate_generation(
                        generation=generation,
                        input_text=query,
                        output_text=str(result.analysis)  # Changed from content to analysis
                    )
                            
                generation.end(success=True)
                logger.info("Tool execution completed successfully")
                print("✓ Tool execution successful")
            else:
                logger.warning("Tool execution completed but returned no results")
                print("⚠ Tool execution completed with no results")

            trace_data = self._prepare_trace_data(
                start_time=start_time,
                success=bool(result),
                tool_name=tool_name,
                prompt_name=prompt_name
            )
            
            # Include evaluation results in trace data
            trace_data['evaluation_results'] = evaluation_results
            
            return result, trace_data

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in run_tool: {error_msg}", exc_info=True)
            print(f"ERROR: Tool execution failed - {error_msg}")
            
            # Instead of using trace.error(), use a more appropriate method
            trace.update(status="ERROR")
            
            generation.end(
                success=False, 
                error_message=error_msg
            )
            
            return None, {"error": error_msg}

        finally:
            logger.debug("Flushing Langfuse traces")
            self.langfuse.flush()

    def _evaluate_generation(self, generation, input_text: str, output_text: str):
        """Evaluate a single generation with enhanced logging"""
        try:
            print("\n=== Evaluation Results ===")
            logger.info("Starting comprehensive generation evaluation")
            
            # Track evaluation results
            evaluation_summary = {}
            
            for criterion, evaluator in self.evaluators.items():
                try:
                    # Different evaluation approach for hallucination
                    if criterion == "hallucination":
                        eval_result = evaluator.evaluate_strings(
                            prediction=output_text,
                            input=input_text,
                            reference=input_text
                        )
                    else:
                        eval_result = evaluator.evaluate_strings(
                            prediction=output_text,
                            input=input_text
                        )
                    
                    # Ensure evaluation result contains expected keys
                    if eval_result and "score" in eval_result and "reasoning" in eval_result:
                        # Record score in Langfuse
                        self.langfuse.score(
                            name=criterion,
                            trace_id=generation.trace_id,
                            observation_id=generation.id,
                            value=eval_result["score"],
                            comment=eval_result["reasoning"]
                        )
                        
                        # Log and print detailed evaluation
                        logger.info(f"Evaluation for {criterion}:")
                        logger.info(f"  Score: {eval_result['score']}")
                        logger.info(f"  Reasoning: {eval_result['reasoning']}")
                        
                        print(f"{criterion.capitalize()} Evaluation:")
                        print(f"  Score: {eval_result['score']}")
                        print(f"  Reasoning: {eval_result['reasoning']}")
                        
                        # Store in summary
                        evaluation_summary[criterion] = {
                            "score": eval_result["score"],
                            "reasoning": eval_result["reasoning"]
                        }
                    
                    else:
                        logger.warning(f"Incomplete evaluation result for {criterion}")
                        print(f"Warning: Incomplete evaluation for {criterion}")
                
                except Exception as criterion_error:
                    logger.error(f"Error evaluating {criterion}: {str(criterion_error)}")
                    print(f"Error evaluating {criterion}: {str(criterion_error)}")
            
            # Overall evaluation summary
            print("\n=== Evaluation Summary ===")
            logger.info("Evaluation Summary:")
            for criterion, details in evaluation_summary.items():
                print(f"{criterion.capitalize()}:")
                print(f"  Score: {details['score']}")
                logger.info(f"{criterion.capitalize()} Score: {details['score']}")
            
            return evaluation_summary
        
        except Exception as e:
            error_msg = f"Critical error in generation evaluation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            raise

    def _prepare_trace_data(
        self,
        start_time: datetime,
        success: bool,
        tool_name: str,
        prompt_name: str
    ) -> Dict[str, Any]:
        """Prepare trace data with execution metrics"""
        duration = (datetime.now() - start_time).total_seconds()
        
        trace_data = {
            "duration": duration,
            "success": success,
            "tool": tool_name,
            "prompt_used": prompt_name,
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
    """Create a new LangfuseRunner instance"""
    logger.info("Creating new LangfuseRunner instance")
    print("\nInitializing new LangfuseRunner...")
    return LangfuseRunner(public_key=public_key, secret_key=secret_key, host=host)

if __name__ == "__main__":
    try:
        # Create runner instance
        runner = create_tool_runner()
        
        # Example of running a general agent
        result, trace_data = runner.run_tool(
            tool_name="General Agent",
            query="What is machine learning?",
            prompt_name="research.txt"
        )
        
        # Example of running an analysis agent
        result, trace_data = runner.run_tool(
            tool_name="Analysis Agent",
            query="Analyze the trends in this dataset",
            dataset="example_dataset",
            prompt_name="analysis.txt"
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        print(f"ERROR: {str(e)}")