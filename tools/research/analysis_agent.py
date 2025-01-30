import logging
from datetime import datetime
from typing import Dict, Optional, Type, Any, List
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from utils.model_wrapper import model_wrapper
from utils.token_tracking import TokenUsageTracker
from prompt import Prompt
import os

class AnalysisAgentInput(BaseModel):
    query: str = Field(description="Analysis query to process")
    dataset: str = Field(description="Name of the CSV file to analyze")

class AnalysisMetrics(BaseModel):
    numerical_accuracy: float = Field(default=0.0)
    query_understanding: float = Field(default=0.0)
    data_validation: float = Field(default=0.0)
    reasoning_transparency: float = Field(default=0.0)
    calculations: Dict = Field(default_factory=dict)
    used_correct_columns: bool = Field(default=False)
    used_correct_analysis: bool = Field(default=False)
    used_correct_grouping: bool = Field(default=False)
    handled_missing_data: bool = Field(default=False)
    handled_outliers: bool = Field(default=False)
    handled_datatypes: bool = Field(default=False)
    handled_format_issues: bool = Field(default=False)
    explained_steps: bool = Field(default=False)
    stated_assumptions: bool = Field(default=False)
    mentioned_limitations: bool = Field(default=False)
    clear_methodology: bool = Field(default=False)

class AnalysisResult(BaseModel):
    """Structured output for analysis results"""
    analysis: str
    metrics: AnalysisMetrics
    usage: Dict[str, Any] = Field(default_factory=dict)

ANALYZE_DATA_PROMPT = Prompt(
    id="analyze-data",
    content="""Analyze the following dataset based on the query:
    
    Query: {{query}}
    
    Dataset Info:
    {{dataset_info}}
    
    Provide:
    1. Key insights and findings
    2. Statistical analysis
    3. Data quality checks
    4. Methodology explanation
    5. Limitations and assumptions
    
    Return results in a clear, structured format."""
)

class AnalysisAgent(BaseTool):
    name: str = "analysis-agent"
    description: str = "Analyzes datasets using statistical methods"
    args_schema: Type[BaseModel] = AnalysisAgentInput
    custom_prompt: Optional[Prompt] = Field(default=None)
    token_tracker: TokenUsageTracker = Field(default_factory=TokenUsageTracker)
    data_folder: str = Field(default="./data")  # Add this line


    def __init__(
        self,
        custom_prompt: Optional[Prompt] = None,
        data_folder: str = "./data"
    ):
        super().__init__()
        self.custom_prompt = custom_prompt or ANALYZE_DATA_PROMPT
        self.token_tracker = TokenUsageTracker()
        self.data_folder = data_folder
        
        # Ensure data folder exists
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

    def get_available_datasets(self):
        
        # Print the full path to verify
        print(f"Looking for datasets in: {os.path.abspath(self.data_folder)}")
        
        # List all CSV files in the data folder
        try:
            datasets = [
                f for f in os.listdir(self.data_folder) 
                if f.endswith('.csv')
            ]
            
            print(f"Found datasets: {datasets}")
            return datasets
        except Exception as e:
            print(f"Error finding datasets: {e}")
            return []

    def load_and_validate_data(self, file_name: str) -> pd.DataFrame:
        """Load and perform initial validation of the dataset."""
        try:
            file_path = os.path.join(self.data_folder, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file not found: {file_path}")

            df = pd.read_csv(file_path)
            validation_results = {
                'missing_data': df.isnull().sum().to_dict(),
                'datatypes': df.dtypes.to_dict(),
                'row_count': len(df),
                'column_count': len(df.columns)
            }
            logging.info(f"Data validation results: {validation_results}")
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def perform_statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform comprehensive statistical analysis on the dataset."""
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            analysis_results = {
                'descriptive_stats': df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {},
                'correlations': df[numeric_cols].corr().to_dict() if len(numeric_cols) > 0 else {},
                'missing_values': df.isnull().sum().to_dict(),
                'unique_counts': {col: df[col].nunique() for col in df.columns},
                'categorical_summaries': {
                    col: df[col].value_counts().to_dict() 
                    for col in categorical_cols
                } if len(categorical_cols) > 0 else {}
            }
            
            # Add basic outlier detection for numeric columns
            if len(numeric_cols) > 0:
                outliers = {}
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers[col] = len(df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))])
                analysis_results['outliers'] = outliers
            
            return analysis_results
        except Exception as e:
            logging.error(f"Error in statistical analysis: {str(e)}")
            raise

    def invoke_analysis(
        self,
        input: Dict[str, str],
        custom_prompt: Optional[Prompt] = None
    ) -> AnalysisResult:
        """Execute the analysis with comprehensive error handling and token tracking."""
        logging.info(f"Starting analysis for query: {input.get('query', 'No query')}")
        
        try:
            if not input or 'query' not in input or 'dataset' not in input:
                raise ValueError("Query and dataset must be provided")

            dataset_file = input['dataset']
            available_datasets = self.get_available_datasets()
            
            if dataset_file not in available_datasets:
                raise ValueError(f"Invalid dataset: {dataset_file}. Available datasets: {available_datasets}")

            # Load and validate data
            df = self.load_and_validate_data(dataset_file)

            # Perform statistical analysis
            analysis_results = self.perform_statistical_analysis(df)

            # Create analysis metrics
            metrics = AnalysisMetrics(
                numerical_accuracy=1.0,
                query_understanding=1.0,
                data_validation=1.0,
                reasoning_transparency=1.0,
                handled_missing_data=True,
                handled_outliers=True,
                handled_datatypes=True,
                handled_format_issues=True,
                explained_steps=True,
                stated_assumptions=True,
                mentioned_limitations=True,
                clear_methodology=True
            )

            # Generate analysis text
            dataset_info = {
                'filename': dataset_file,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'analysis_results': analysis_results
            }

            prompt_to_use = custom_prompt or self.custom_prompt
            system_prompt = prompt_to_use.compile(
                query=input['query'],
                dataset_info=str(dataset_info)
            )

            analysis_text = model_wrapper(
                system_prompt=system_prompt,
                prompt=prompt_to_use,
                user_prompt=input['query'],
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
                token_tracker=self.token_tracker
            )

            return AnalysisResult(
                analysis=analysis_text,
                metrics=metrics,
                usage={
                    'prompt_tokens': self.token_tracker._total_prompt_tokens,
                    'completion_tokens': self.token_tracker._total_completion_tokens,
                    'total_tokens': self.token_tracker._total_tokens,
                    'model': 'llama3-70b-8192'
                }
            )

        except Exception as e:
            logging.error(f"Error in analysis: {str(e)}")
            raise

    def _run(self, **kwargs) -> AnalysisResult:
        """Execute the analysis tool with the provided parameters."""
        return self.invoke_analysis(input=kwargs)

def run_tool(tool_name: str, query: str, dataset: str = None, tool=None):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s - %(filename)s:%(lineno)d',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('research_tool.log', mode='a')
        ]
    )
    logger = logging.getLogger(__name__)
    
    start_time = datetime.now()
    db = ContentDB("./data/content.db")    
    logger.info(f"Starting tool execution - Tool: {tool_name}")
    logger.info(f"Query received: {query}")
    
    trace = QueryTrace(query)
    trace.data.update({
        "tool": tool_name,
        "tools_used": [tool_name],
        "processing_steps": [],
        "content_new": 0,
        "content_reused": 0
    })

    try:
        if tool_name == "Analysis Agent":
            if tool is None:
                tool = AnalysisAgent(data_folder="./data")
            
            # Get available datasets
            available_datasets = tool.get_available_datasets()
            if not available_datasets:
                raise ValueError("No CSV datasets found in data folder")
            
            if dataset not in available_datasets:
                raise ValueError(f"Invalid dataset: {dataset}. Available datasets: {available_datasets}")
            
            trace.add_prompt_usage("analysis_agent", "analysis", "")
            result = tool.invoke_analysis(input={"query": query, "dataset": dataset})
            
            if result:
                try:
                    evaluation_data = {
                        'query': query,
                        'timestamp': datetime.now().isoformat(),
                        'analysis': result.analysis,
                        'metrics': {}
                    }

                    # Run analysis-specific evaluations
                    if analysis_evaluator:
                        analysis_metrics = analysis_evaluator.evaluate_analysis(result, query)
                        trace.data['analysis_metrics'] = analysis_metrics
                        
                        evaluation_data.update({
                            'numerical_accuracy': analysis_metrics.get('numerical_accuracy', {}).get('score', 0.0),
                            'query_understanding': analysis_metrics.get('query_understanding', {}).get('score', 0.0),
                            'data_validation': analysis_metrics.get('data_validation', {}).get('score', 0.0),
                            'reasoning_transparency': analysis_metrics.get('reasoning_transparency', {}).get('score', 0.0),
                            'overall_score': analysis_metrics.get('overall_score', 0.0),
                            'metrics_details': json.dumps(analysis_metrics),
                            'calculation_examples': json.dumps(analysis_metrics.get('numerical_accuracy', {}).get('details', {}).get('calculation_examples', [])),
                            'term_coverage': analysis_metrics.get('query_understanding', {}).get('details', {}).get('term_coverage', 0.0),
                            'analytical_elements': json.dumps(analysis_metrics.get('query_understanding', {}).get('details', {})),
                            'validation_checks': json.dumps(analysis_metrics.get('data_validation', {}).get('details', {})),
                            'explanation_patterns': json.dumps(analysis_metrics.get('reasoning_transparency', {}).get('details', {}))
                        })
                        
                    # Store complete analysis evaluation
                    db.store_analysis_evaluation(evaluation_data)
                    
                    trace.data.update({
                        "processing_steps": ["Analysis completed successfully"],
                        "analysis_metrics": evaluation_data
                    })
                    
                except Exception as eval_error:
                    logger.error(f"Analysis evaluation failed: {eval_error}")
                    trace.data['evaluation_error'] = str(eval_error)

        else:
            error_msg = f"Tool {tool_name} not found"
            logger.error(error_msg)
            trace.data.update({
                "processing_steps": [f"Error: {error_msg}"],
                "error": error_msg,
                "success": False
            })
            db.close()
            return None, trace

        # Finalize execution
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trace.data.update({
            "duration": duration,
            "success": True if result else False,
            "end_time": end_time.isoformat()
        })
        
        try:
            token_stats = trace.token_tracker.get_usage_stats()
            logger.info(f"Final token usage stats: {token_stats}")
            
            if token_stats['tokens']['total'] > 0:
                usage_msg = f"Total tokens used: {token_stats['tokens']['total']}"
                logger.info(usage_msg)
                trace.data["processing_steps"].append(usage_msg)
        except Exception as token_error:
            logger.warning(f"Could not retrieve token stats: {token_error}")
        
        logger.info(f"{tool_name} completed successfully")
        trace.data["processing_steps"].append(f"{tool_name} completed successfully")
        
        try:
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_save_error:
            logger.error(f"Failed to save trace: {trace_save_error}")
        
        db.close()
        return result, trace
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error running {tool_name}: {error_msg}", exc_info=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trace.data.update({
            "end_time": end_time.isoformat(),
            "duration": duration,
            "error": error_msg,
            "success": False,
            "processing_steps": [f"Execution failed: {error_msg}"]
        })
        
        try:
            tracer = CustomTracer()
            tracer.save_trace(trace)
        except Exception as trace_save_error:
            logger.error(f"Failed to save error trace: {trace_save_error}")
        
        db.close()
        return None, trace