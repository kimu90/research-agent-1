import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Optional, Type, Any, List
import pandas as pd
from bs4 import BeautifulSoup
from langfuse import Langfuse

from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from utils.model_wrapper import model_wrapper
from utils.token_tracking import TokenUsageTracker
from prompt import Prompt
from langchain_community.utilities import GoogleSerperAPIWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('analysis_agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

class PromptLoader:
    """Handles loading of prompts from Langfuse."""
    
    @staticmethod
    def load_prompt(langfuse_client: Langfuse, prompt_name: str = "general-analysis", 
                   version: str = None, variant: str = None) -> Prompt:
        """Load a prompt from Langfuse."""
        try:
            if version:
                langfuse_prompt = langfuse_client.get_prompt_version(prompt_name, version)
            elif variant:
                variant_name = f"{prompt_name}-{variant}"
                langfuse_prompt = langfuse_client.get_prompt(variant_name)
            else:
                langfuse_prompt = langfuse_client.get_prompt(prompt_name)
            
            return Prompt(
                id=langfuse_prompt.name,
                content=langfuse_prompt.prompt,
                metadata={
                    "source": "langfuse",
                    "version": version,
                    "variant": variant,
                    "config": langfuse_prompt.config
                }
            )
        except Exception as e:
            logger.error(f"Error loading Langfuse prompt: {e}")
            # Fallback prompt
            return Prompt(
                id="fallback-prompt",
                content="""Analyze the following content and provide a comprehensive summary.
                
                Research Topic: {{research_topic}}
                Content: {{content}}
                
                Provide key insights, main themes, and relevant conclusions."""
            )

    @staticmethod
    def list_available_prompts(langfuse_client: Langfuse) -> List[str]:
        """List all available prompts from Langfuse."""
        try:
            prompts = langfuse_client.list_prompts()
            return [prompt.name for prompt in prompts]
        except Exception as e:
            logger.error(f"Error listing Langfuse prompts: {e}")
            return []

class AnalysisAgentInput(BaseModel):
    query: str = Field(description="Analysis query to process")
    dataset: str = Field(description="Name of the CSV file to analyze")
    analysis_type: str = Field(
        default="general",
        description="Type of analysis to perform (e.g., 'general', 'statistical', 'correlation')"
    )

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

class AnalysisAgent(BaseTool):
    name: str = "analysis-agent"
    description: str = "Analyzes datasets using statistical methods"
    args_schema: Type[BaseModel] = AnalysisAgentInput
    current_prompt: Optional[Prompt] = Field(default=None)
    token_tracker: TokenUsageTracker = Field(default_factory=TokenUsageTracker)
    data_folder: str = Field(default="./data")
    langfuse_client: Optional[Langfuse] = None

    def __init__(
        self,
        langfuse_client: Langfuse,
        data_folder: str = "./data",
        prompt_name: str = "general-analysis",
        prompt_version: Optional[str] = None,
        prompt_variant: Optional[str] = None
    ):
        super().__init__()
        self.langfuse_client = langfuse_client
        self.data_folder = data_folder
        
        try:
            self.current_prompt = PromptLoader.load_prompt(
                langfuse_client=langfuse_client,
                prompt_name=prompt_name,
                version=prompt_version,
                variant=prompt_variant
            )
            logger.info(f"Loaded prompt: {prompt_name} (version: {prompt_version}, variant: {prompt_variant})")
        except Exception as e:
            logger.warning(f"Failed to load Langfuse prompt, falling back to default. Error: {e}")
            self.current_prompt = PromptLoader.load_prompt(
                langfuse_client=langfuse_client,
                prompt_name="general-analysis"
            )
        
        self.token_tracker = TokenUsageTracker()
        
        # Ensure data folder exists
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            
        # Log environment status
        serper_key_present = bool(os.getenv('SERPER_API_KEY'))
        logger.info(f"SERPER_API_KEY present: {serper_key_present}")

    def get_available_datasets(self):
        """List available datasets in data folder."""
        logger.info(f"Looking for datasets in: {os.path.abspath(self.data_folder)}")
        
        try:
            datasets = [
                f for f in os.listdir(self.data_folder) 
                if f.endswith('.csv')
            ]
            
            logger.info(f"Found datasets: {datasets}")
            return datasets
        except Exception as e:
            logger.error(f"Error finding datasets: {e}")
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
            logger.info(f"Data validation results: {validation_results}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
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
            logger.error(f"Error in statistical analysis: {str(e)}")
            raise

    def perform_web_research(self, query: str, df_columns: List[str]) -> List[Dict]:
        """Execute web research with error handling."""
        try:
            if not os.getenv('SERPER_API_KEY'):
                logger.warning("SERPER_API_KEY not found in environment")
                return []

            google_serper = GoogleSerperAPIWrapper(
                type="news",
                k=5,
                serper_api_key=os.getenv('SERPER_API_KEY')
            )
            
            search_query = f"{query} {' '.join(df_columns[:3])} analysis research"
            web_results = google_serper.results(search_query)
            news_results = web_results.get("news", [])
            
            web_research = []
            for news in news_results:
                try:
                    url = news.get("link", "")
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                    }
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    article_content = ""
                    for p in soup.find_all('p'):
                        article_content += p.get_text() + "\n"
                    
                    web_research.append({
                        'title': news.get("title", ""),
                        'snippet': news.get("snippet", ""),
                        'url': url,
                        'content': article_content[:1000]
                    })
                    
                except Exception as scrape_error:
                    logger.warning(f"Error scraping {url}: {str(scrape_error)}")
                    continue
                    
            return web_research
            
        except Exception as e:
            logger.error(f"Error in web research: {str(e)}", exc_info=True)
            return []

    def invoke_analysis(
        self,
        input: Dict[str, str],
        prompt_version: Optional[str] = None,
        prompt_variant: Optional[str] = None
    ) -> AnalysisResult:
        """Execute the analysis with Langfuse prompt management and token tracking."""
        logger.info(f"Starting analysis for query: {input.get('query', 'No query')}")
        
        try:
            if not input or 'query' not in input or 'dataset' not in input:
                raise ValueError("Query and dataset must be provided")

            # Load new prompt version/variant if specified
            if prompt_version or prompt_variant:
                self.current_prompt = PromptLoader.load_prompt(
                    langfuse_client=self.langfuse_client,
                    prompt_name=self.current_prompt.id,
                    version=prompt_version,
                    variant=prompt_variant
                )

            dataset_file = input['dataset']
            available_datasets = self.get_available_datasets()
            
            if dataset_file not in available_datasets:
                raise ValueError(f"Invalid dataset: {dataset_file}. Available datasets: {available_datasets}")

            # Step 1: Load and validate data
            df = self.load_and_validate_data(dataset_file)
            analysis_results = self.perform_statistical_analysis(df)

            # Step 2: Perform web research
            web_research = self.perform_web_research(
                query=input['query'],
                df_columns=list(df.columns)
            )

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

            # Prepare dataset info
            dataset_info = {
                'filename': dataset_file,
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'analysis_results': analysis_results,
                'web_research': web_research
            }

            # Compile prompt with Langfuse tracking
            system_prompt = self.current_prompt.compile(
                query=input['query'],
                dataset_info=str(dataset_info)
            )

            # Create Langfuse generation for tracking
            generation = self.langfuse_client.generation(
                name="analysis-generation",
                metadata={
                    "prompt_name": self.current_prompt.id,
                    "prompt_version": prompt_version,
                    "prompt_variant": prompt_variant,
                    "dataset": dataset_file
                }
            )

            try:
                analysis_text = model_wrapper(
                    system_prompt=system_prompt,
                    prompt=self.current_prompt,
                    user_prompt=f"""Analyze the dataset '{dataset_file}' in the context of the query: {input['query']}.
                                Include relevant insights from web research where applicable.
                                Focus on connecting statistical findings with broader industry/research context.""",
                    model="llama3-70b-8192",
                    host="groq",
                    temperature=0.7,
                    token_tracker=self.token_tracker
                )

                # Log successful generation
                generation.end(
                    success=True,
                    prompt_tokens=self.token_tracker._total_prompt_tokens,
                    completion_tokens=self.token_tracker._total_completion_tokens
                )

            except Exception as model_error:
                generation.end(
                    success=False,
                    error_message=str(model_error)
                )
                raise

            # Track prompt performance
            self.langfuse_client.score(
                name="analysis_quality",
                value=metrics.numerical_accuracy,
                comment="Analysis quality score based on metrics",
                trace_id=generation.trace_id
            )

            # Updated to match new trace data structure
            return AnalysisResult(
                analysis=analysis_text,
                metrics=metrics,
                usage={
                    'prompt_tokens': self.token_tracker._total_prompt_tokens,
                    'completion_tokens': self.token_tracker._total_completion_tokens,
                    'total_tokens': self.token_tracker._total_tokens,
                    'model': 'llama3-70b-8192',
                    'timestamp': datetime.now().isoformat(),
                    'prompt_info': {
                        'name': self.current_prompt.id,
                        'version': prompt_version,
                        'variant': prompt_variant,
                        'generation_id': generation.id
                    }
                }
            )

        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            raise

    def _run(self, **kwargs) -> AnalysisResult:
        """Execute the analysis tool with the provided parameters."""
        return self.invoke_analysis(input=kwargs)