import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from langfuse import Langfuse
from langchain.evaluation import load_evaluator
from langchain_openai import OpenAI
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from dotenv import load_dotenv
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

# Add prompt configurations
PROMPT_CONFIGS = {
    "general": {
        "template": """Analyze the dataset with a focus on {{focus_area}}. 
        Consider the key patterns, trends, and insights that emerge from the data. 
        Provide a comprehensive analysis that highlights significant findings.""",
        "config": {"model": "gpt-4", "temperature": 0.7},
        "labels": ["analysis", "general", "basic"],
        "variants": {
            "concise": {
                "template": "Provide a brief analysis of {{dataset}} focusing on {{focus_area}}",
                "labels": ["analysis", "general", "concise"]
            },
            "detailed": {
                "template": "Perform an in-depth analysis of {{dataset}}, examining {{focus_area}}",
                "labels": ["analysis", "general", "detailed"]
            }
        }
    },
    "detailed": {
        "template": """Perform a detailed analysis of {{dataset}} focusing on {{metrics}}. 
        Include statistical measures, outliers, and relationships between variables.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "labels": ["analysis", "detailed", "statistical"],
        "variants": {
            "technical": {
                "template": "Technical analysis of {{dataset}} with focus on {{metrics}}",
                "labels": ["analysis", "detailed", "technical"]
            },
            "business": {
                "template": "Business-focused analysis of {{dataset}} examining {{metrics}}",
                "labels": ["analysis", "detailed", "business"]
            }
        }
    },
    "comparative": {
        "template": """Compare the following aspects of {{dataset}}: {{aspects}}. 
        Analyze similarities, differences, and relationships.""",
        "config": {"model": "gpt-4", "temperature": 0.3},
        "labels": ["analysis", "comparative"],
        "variants": {
            "simple": {
                "template": "Basic comparison of {{aspects}} in {{dataset}}",
                "labels": ["analysis", "comparative", "simple"]
            },
            "complex": {
                "template": "Detailed comparison with statistical analysis of {{aspects}} in {{dataset}}",
                "labels": ["analysis", "comparative", "complex"]
            }
        }
    },
    "species": {
        "template": """Analyze the species data focusing on {{focus_area}}. 
        Consider biological characteristics, habitat preferences, and population trends.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "labels": ["analysis", "species", "biological"],
        "variants": {
            "ecological": {
                "template": "Analyze ecological aspects of species in {{dataset}}",
                "labels": ["analysis", "species", "ecological"]
            },
            "conservation": {
                "template": "Analyze conservation status and threats in {{dataset}}",
                "labels": ["analysis", "species", "conservation"]
            }
        }
    }
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

# Add prompt configurations with clear categories
PROMPT_CONFIGS = {
    "general": {
        "template": """Analyze the dataset with a focus on {{focus_area}}. 
        Consider the key patterns, trends, and insights that emerge from the data. 
        Provide a comprehensive analysis that highlights significant findings.""",
        "config": {"model": "gpt-4", "temperature": 0.7},
        "labels": ["analysis", "general", "basic"],
        "variants": {
            "concise": {
                "template": "Provide a brief analysis of {{dataset}} focusing on {{focus_area}}",
                "labels": ["analysis", "general", "concise"]
            },
            "detailed": {
                "template": "Perform an in-depth analysis of {{dataset}}, examining {{focus_area}}",
                "labels": ["analysis", "general", "detailed"]
            }
        }
    },
    "detailed": {
        "template": """Perform a detailed analysis of {{dataset}} focusing on {{metrics}}. 
        Include statistical measures, outliers, and relationships between variables.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "labels": ["analysis", "detailed", "statistical"],
        "variants": {
            "technical": {
                "template": "Technical analysis of {{dataset}} with focus on {{metrics}}",
                "labels": ["analysis", "detailed", "technical"]
            },
            "business": {
                "template": "Business-focused analysis of {{dataset}} examining {{metrics}}",
                "labels": ["analysis", "detailed", "business"]
            }
        }
    },
    "comparative": {
        "template": """Compare the following aspects of {{dataset}}: {{aspects}}. 
        Analyze similarities, differences, and relationships.""",
        "config": {"model": "gpt-4", "temperature": 0.3},
        "labels": ["analysis", "comparative"],
        "variants": {
            "simple": {
                "template": "Basic comparison of {{aspects}} in {{dataset}}",
                "labels": ["analysis", "comparative", "simple"]
            },
            "complex": {
                "template": "Detailed comparison with statistical analysis of {{aspects}} in {{dataset}}",
                "labels": ["analysis", "comparative", "complex"]
            }
        }
    },
    "species": {
        "template": """Analyze the species data focusing on {{focus_area}}. 
        Consider biological characteristics, habitat preferences, and population trends.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "labels": ["analysis", "species", "biological"],
        "variants": {
            "ecological": {
                "template": "Analyze ecological aspects of species in {{dataset}}",
                "labels": ["analysis", "species", "ecological"]
            },
            "conservation": {
                "template": "Analyze conservation status and threats in {{dataset}}",
                "labels": ["analysis", "species", "conservation"]
            }
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
        self._all_labels = set()  # Store all unique labels
        self.prompts = {}  # Store initialized prompts
        
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
        """Initialize prompt templates in Langfuse with enhanced label handling"""
        try:
            # Collect all unique labels first
            for analysis_type, config in PROMPT_CONFIGS.items():
                self._all_labels.update(config.get("labels", []))
                for variant_config in config.get("variants", {}).values():
                    if isinstance(variant_config, dict):  # Check if it's a dict with labels
                        self._all_labels.update(variant_config.get("labels", []))
            
            logger.info(f"Found labels: {sorted(list(self._all_labels))}")
            
            # Create base prompts
            for analysis_type, config in PROMPT_CONFIGS.items():
                try:
                    prompt = self.langfuse.create_prompt(
                        name=f"{analysis_type}-analysis",
                        prompt=config["template"],
                        config=config["config"],
                        labels=config.get("labels", ["analysis"])
                    )
                    self.prompts[analysis_type] = prompt
                    logger.debug(f"Created base prompt: {analysis_type}-analysis")

                    # Create variants
                    for variant_name, variant_config in config.get("variants", {}).items():
                        variant_template = variant_config
                        variant_labels = ["analysis"]
                        
                        if isinstance(variant_config, dict):
                            variant_template = variant_config["template"]
                            variant_labels.extend(variant_config.get("labels", []))
                        
                        self.langfuse.create_prompt(
                            name=f"{analysis_type}-{variant_name}",
                            prompt=variant_template,
                            config=config["config"],
                            labels=variant_labels
                        )
                        logger.debug(f"Created variant prompt: {analysis_type}-{variant_name}")
                
                except Exception as prompt_error:
                    logger.error(f"Failed to create prompt for {analysis_type}: {str(prompt_error)}")
                    continue
            
            logger.info(f"Initialized prompts for types: {list(self.prompts.keys())}")
        except Exception as e:
            error_msg = f"Failed to initialize prompts: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise

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

    def get_prompt(self, analysis_type: str, version: str = None, variant: str = None):
        """Get appropriate prompt based on analysis type and version"""
        try:
            base_prompt = self.prompts.get(analysis_type, self.prompts.get("general"))
            if not base_prompt:
                logger.warning(f"No prompt found for {analysis_type}, using general")
                return self.prompts.get("general")
            
            if version:
                return self.langfuse.get_prompt_version(base_prompt.name, version)
            elif variant:
                variant_name = f"{analysis_type}-{variant}"
                return self.langfuse.get_prompt(variant_name)
            
            return base_prompt
        except Exception as e:
            logger.error(f"Error getting prompt: {str(e)}")
            return self.prompts.get("general")

    def search_prompts(self, query: str = None, labels: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search prompts by query string and/or labels.
        """
        try:
            matching_prompts = []
            
            for prompt_name, prompt in self.prompts.items():
                # Get prompt details
                try:
                    prompt_details = self.langfuse.get_prompt(prompt_name)
                    if not prompt_details:
                        continue
                        
                    # Check search criteria
                    if self._matches_criteria(
                        prompt_details.name,
                        prompt_details.prompt,
                        prompt_details.labels,
                        query,
                        labels
                    ):
                        matching_prompts.append({
                            "name": prompt_details.name,
                            "content": prompt_details.prompt,
                            "config": prompt_details.config,
                            "labels": prompt_details.labels,
                            "version": prompt_details.version
                        })
                except Exception as prompt_error:
                    logger.warning(f"Error fetching details for {prompt_name}: {str(prompt_error)}")
                    continue
            
            return matching_prompts
            
        except Exception as e:
            logger.error(f"Error searching prompts: {str(e)}")
            return []

    def _matches_criteria(self, name: str, content: str, prompt_labels: List[str], 
                         query: str = None, labels: List[str] = None) -> bool:
        """Check if a prompt matches search criteria"""
        # Check query
        if query:
            query = query.lower()
            if not (query in name.lower() or query in content.lower()):
                return False
        
        # Check labels
        if labels:
            if not all(label in prompt_labels for label in labels):
                return False
        
        return True

    def get_available_labels(self) -> List[str]:
        """Get all available labels"""
        return sorted(list(self._all_labels))

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

    def run_tool(
        self,
        query: str,
        tool_name: str = "Analysis Agent",
        dataset: str = None,
        analysis_type: str = "general",
        prompt_name: str = "research.txt",
        prompt_version: str = None,
        prompt_variant: str = None,
        run_evaluation: bool = True,
        prompt_tags: list = None,
        tool: Optional[Any] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Execute a tool and optionally evaluate its output with prompt management"""
        logger.info(f"=== Starting tool execution: {tool_name} ===")
        print(f"\nExecuting {tool_name}...")

        start_time = datetime.now()
        
        # Get appropriate prompt
        prompt = self.get_prompt(
            analysis_type=analysis_type,
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
                "prompt_tags": prompt_tags or []
            }
        )

        try:
            generation = trace.generation(
                name=f"{tool_name.lower()}-generation",
                metadata={
                    "prompt_used": prompt.name,
                    "prompt_version": prompt_version,
                    "prompt_variant": prompt_variant,
                    "prompt_tags": prompt_tags or []
                }
            )

            if tool_name == "Analysis Agent":
                from tools import AnalysisAgent
                # Initialize AnalysisAgent with Langfuse prompt management
                tool = tool or AnalysisAgent(
                    langfuse_client=self.langfuse,
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
                    "config": prompt.config,
                    "tags": prompt_tags or []
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
                    "prompt_variant": prompt_variant,
                    "prompt_tags": prompt_tags or []
                }
            )
            
            return None, {"error": error_msg}

        finally:
            logger.debug("Flushing Langfuse traces")
            self.langfuse.flush()

   

def create_tool_runner(
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None
) -> LangfuseRunner:
    """Create a new LangfuseRunner instance with prompt management"""
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