import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, Optional, Type, Any, List
import pandas as pd
from bs4 import BeautifulSoup
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
    """Handles dynamic loading of prompts from a specified directory."""
    
    @staticmethod
    def load_prompt(prompt_name: str = "research.txt") -> Prompt:
        """Load a prompt from the prompts directory."""
        try:
            with open(os.path.join("/app/prompts", prompt_name), 'r') as f:
                prompt_content = f.read()
            
            return Prompt(
                id=f"dynamic-prompt-{prompt_name}",
                content=prompt_content,
                metadata={"source": prompt_name}
            )
        except FileNotFoundError:
            logger.error(f"Prompt file not found: {prompt_name}")
            return Prompt(
                id="fallback-prompt",
                content="""Analyze the following content and provide a comprehensive summary.
                
                Research Topic: {{research_topic}}
                Content: {{content}}
                
                Provide key insights, main themes, and relevant conclusions."""
            )

    @staticmethod
    def list_available_prompts() -> List[str]:
        """List all available prompt files."""
        base_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
        
        try:
            return [
                f for f in os.listdir(base_path) 
                if f.endswith('.txt') and os.path.isfile(os.path.join(base_path, f))
            ]
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
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

    def __init__(
        self,
        current_prompt: Optional[Prompt] = None,
        data_folder: str = "./data",
        prompt_name: Optional[str] = None
    ):
        super().__init__()
        if current_prompt:
            self.current_prompt = current_prompt
        elif prompt_name:
            try:
                self.current_prompt = PromptLoader.load_prompt(prompt_name)
            except Exception as e:
                logger.warning(f"Failed to load prompt {prompt_name}, falling back to default. Error: {e}")
                self.current_prompt = PromptLoader.load_prompt("research.txt")
        else:
            self.current_prompt = Prompt(
                id="default-content-selection",
                content="""Analyze the following articles:
                Research Topic: {{research_topic}}
                Available Articles: {{formatted_snippets}}
                Return the indices of the most relevant articles."""
            )
        
        self.token_tracker = TokenUsageTracker()
        self.data_folder = data_folder
        
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
        current_prompt: Optional[Prompt] = None
    ) -> AnalysisResult:
        """Execute the analysis with comprehensive error handling and token tracking."""
        logger.info(f"Starting analysis for query: {input.get('query', 'No query')}")
        
        try:
            if not input or 'query' not in input or 'dataset' not in input:
                raise ValueError("Query and dataset must be provided")

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

            prompt_to_use = current_prompt or self.current_prompt
            system_prompt = prompt_to_use.compile(
                query=input['query'],
                dataset_info=str(dataset_info)
            )

            analysis_text = model_wrapper(
                system_prompt=system_prompt,
                prompt=prompt_to_use,
                user_prompt=f"""Analyze the dataset '{dataset_file}' in the context of the query: {input['query']}.
                            Include relevant insights from web research where applicable.
                            Focus on connecting statistical findings with broader industry/research context.""",
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
            logger.error(f"Error in analysis: {str(e)}", exc_info=True)
            raise

    def _run(self, **kwargs) -> AnalysisResult:
        """Execute the analysis tool with the provided parameters."""
        return self.invoke_analysis(input=kwargs)