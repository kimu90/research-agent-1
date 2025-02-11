import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from tools.research.common.model_schemas import ResearchToolOutput, ContentItem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('research_agent.log')
    ]
)

logger = logging.getLogger(__name__)

# Suppress noisy logs
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
    SERPER_AVAILABLE = True
    logger.info("Successfully imported GoogleSerperAPIWrapper")
except ImportError:
    SERPER_AVAILABLE = False
    logger.error("Failed to import GoogleSerperAPIWrapper. Web search will be disabled.")

class PromptLoader:
    """Handles dynamic loading of prompts from a specified directory."""
    
    @staticmethod
    def load_prompt(prompt_name: str = "research.txt") -> str:
        """Load a prompt from the prompts directory."""
        logger.info(f"Loading prompt: {prompt_name}")
        
        base_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
        prompt_path = os.path.join(base_path, prompt_name)
        
        try:
            with open(prompt_path, 'r') as f:
                content = f.read()
            logger.info(f"Successfully loaded prompt from {prompt_path}")
            return content
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_path}")
            
            default_prompt_path = os.path.join(base_path, "research.txt")
            try:
                with open(default_prompt_path, 'r') as f:
                    content = f.read()
                logger.info("Successfully loaded default research prompt")
                return content
            except FileNotFoundError:
                logger.error("Default research prompt not found, using fallback")
                return """Default Research Prompt
                
                Perform comprehensive research on the given topic:
                1. Gather relevant information
                2. Analyze key points
                3. Provide a summary of findings"""

    @staticmethod
    def list_available_prompts() -> List[str]:
        """List all available prompt files."""
        base_path = os.path.join(os.path.dirname(__file__), "..", "prompts")
        
        try:
            prompts = [
                f for f in os.listdir(base_path) 
                if f.endswith('.txt') and os.path.isfile(os.path.join(base_path, f))
            ]
            logger.info(f"Found {len(prompts)} available prompts")
            return prompts
        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}")
            return []

class Question(BaseModel):
    """Represents an individual research question."""
    id: str = Field(
        ...,
        description="Unique identifier for each question"
    )
    text: str = Field(..., description="The text of the question")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of question IDs this depends on"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "dependencies": self.dependencies,
        }

class ResearchOutline(BaseModel):
    """Represents a research outline consisting of questions."""
    questions: List[Question] = Field(
        ...,
        description="List of questions",
        min_items=1,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {"questions": [q.to_dict() for q in self.questions]}

class ResearchAgent:
    """Orchestrates the research process using tools and prompts."""

    def __init__(
        self, 
        tools: List[BaseTool], 
        prompt_name: str = "research.txt",
        include_summary: bool = True
    ):
        """Initialize ResearchAgent."""
        logger.info(f"Initializing ResearchAgent with prompt: {prompt_name}")
        
        self.tools = tools
        self.include_summary = include_summary
        
        # Check environment variables
        if not os.getenv('SERPER_API_KEY'):
            logger.warning("SERPER_API_KEY not found in environment variables")
        
        # Load prompt
        prompt_content = PromptLoader.load_prompt(prompt_name)
        self.research_prompt = {
            "content": prompt_content,
            "name": prompt_name
        }
        
        logger.info("ResearchAgent initialized successfully")

    def perform_web_search(self, query: str) -> List[Dict]:
        """Execute web search with error handling."""
        logger.info(f"Starting web search for query: {query}")
        
        if not SERPER_AVAILABLE:
            logger.error("Web search unavailable - GoogleSerperAPIWrapper not imported")
            return []
            
        if not os.getenv('SERPER_API_KEY'):
            logger.error("Web search unavailable - SERPER_API_KEY not found")
            return []
            
        try:
            google_serper = GoogleSerperAPIWrapper(
                type="news", 
                k=10, 
                serper_api_key=os.getenv('SERPER_API_KEY')
            )
            
            response = google_serper.results(query=query)
            results = response.get("news", [])
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}", exc_info=True)
            return []

    def generate_summary(self, query: str, content: List[ContentItem]) -> str:
        """Generate research summary."""
        logger.info("Generating research summary")
        
        try:
            formatted_content = "\n\n".join([
                f"### {item.title}\n{item.snippet}"
                for item in content
            ])
            
            system_prompt = self.research_prompt["content"].replace(
                "{{query}}", 
                query
            ).replace(
                "{{content}}", 
                formatted_content
            )
            
            summary = model_wrapper(
                system_prompt=system_prompt,
                user_prompt=f"Summarize the research on: '{query}'",
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7
            )
            
            logger.info("Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}", exc_info=True)
            return "Unable to generate summary."

    def invoke(self, input: Dict[str, str]) -> ResearchToolOutput:
        """Execute research process."""
        start_time = datetime.now()
        logger.info(f"Starting research at {start_time}")
        logger.info(f"Input query: {input.get('query', 'No query')}")
        
        # Create default output
        default_output = ResearchToolOutput(
            content=[],
            summary="No results found.",
            usage={
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'model': 'unknown'
            }
        )
        
        try:
            # Validate input
            if not input or 'query' not in input:
                logger.warning("No query provided")
                return default_output
            
            # Perform web search
            news_results = self.perform_web_search(input["query"])
            if not news_results:
                logger.warning("No search results found")
                return default_output
            
            # Process results
            content = []
            for idx, news in enumerate(news_results, 1):
                logger.debug(f"Processing result {idx}/{len(news_results)}")
                content.append(
                    ContentItem(
                        url=news.get("link", ""),
                        title=news.get("title", "") + " - " + news.get("date", ""),
                        snippet=news.get("snippet", ""),
                        content=""
                    )
                )
            
            # Generate summary if enabled
            summary = ""
            if self.include_summary:
                summary = self.generate_summary(input["query"], content)
            
            # Prepare output
            output = ResearchToolOutput(
                content=content,
                summary=summary
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Research completed in {duration:.2f} seconds")
            
            return output
            
        except Exception as e:
            logger.error(f"Research failed: {str(e)}", exc_info=True)
            default_output.summary = f"An error occurred: {str(e)}"
            return default_output

def list_research_prompts() -> List[str]:
    """List available research prompts."""
    return PromptLoader.list_available_prompts()

if __name__ == "__main__":
    # Test logging configuration
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")