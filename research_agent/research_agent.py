# research_agent/research_agent.py

import os
import json
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from typing import List, Dict, Any, Optional

from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from datetime import datetime

from tools.research.common.model_schemas import ResearchToolOutput, ContentItem
from langchain_community.utilities import GoogleSerperAPIWrapper

class PromptLoader:
    """
    Handles dynamic loading of prompts from a specified directory.
    """
    @staticmethod
    def load_prompt(prompt_name: str = "research.txt") -> str:
        """
        Load a prompt from the prompts directory.
        
        Args:
            prompt_name: Name of the prompt file to load (default: research.txt)
        
        Returns:
            str: Loaded prompt content
        """
        # Determine the base path for prompts
        base_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "prompts"
        )
        
        # Construct full path to the prompt file
        prompt_path = os.path.join(base_path, prompt_name)
        
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {prompt_path}")
            
            # Fallback to default prompt if specific prompt is not found
            default_prompt_path = os.path.join(base_path, "research.txt")
            try:
                with open(default_prompt_path, 'r') as f:
                    return f.read()
            except FileNotFoundError:
                logging.error("Default research prompt not found!")
                return """Default Research Prompt
                
                Perform comprehensive research on the given topic:
                1. Gather relevant information
                2. Analyze key points
                3. Provide a summary of findings"""

    @staticmethod
    def list_available_prompts() -> List[str]:
        """
        List all available prompt files in the prompts directory.
        
        Returns:
            List of prompt file names
        """
        base_path = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "prompts"
        )
        
        try:
            return [
                f for f in os.listdir(base_path) 
                if f.endswith('.txt') and os.path.isfile(os.path.join(base_path, f))
            ]
        except Exception as e:
            logging.error(f"Error listing prompts: {e}")
            return []

class Question(BaseModel):
    """Represents an individual research question."""
    id: str = Field(
        ...,
        description="Unique identifier for each question, reflecting position and dependency structure",
    )
    text: str = Field(..., description="The text of the question")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of IDs that this question depends on. Empty array indicates no dependencies",
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
        description="List of main questions and subquestions",
        min_items=1,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {"questions": [question.to_dict() for question in self.questions]}

class ResearchAgent:
    """
    Orchestrates the research process using tools and dynamically loaded prompts.
    """

    def __init__(
        self, 
        tools: List[BaseTool], 
        prompt_name: str = "research.txt",
        include_summary: bool = True
    ):
        """
        Initialize ResearchAgent with required tools and prompt.
        
        Args:
            tools: List of LangChain tools for research tasks
            prompt_name: Name of the prompt file to load
            include_summary: Whether to generate a summary
        """
        self.tools = tools
        self.include_summary = include_summary
        
        # Dynamically load prompt
        prompt_content = PromptLoader.load_prompt(prompt_name)
        
        # Create prompt object
        self.research_prompt = {
            "content": prompt_content,
            "name": prompt_name
        }

    def invoke(
        self, 
        input: Dict[str, str]
    ) -> ResearchToolOutput:
        """
        Execute comprehensive research process
        
        Args:
            input: Dictionary containing the search query
        
        Returns:
            ResearchToolOutput containing search results and usage information
        """
        # Logging and initial setup
        logging.info(f"Starting research for query: {input.get('query', 'No query')}")
        
        # Create a default output structure
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
                logging.warning("No query provided")
                return default_output
            
            # Initialize search tools
            google_serper = GoogleSerperAPIWrapper(
                type="news", 
                k=10, 
                serper_api_key=os.getenv('SERPER_API_KEY')
            )
            
            # Perform initial news search
            try:
                response = google_serper.results(query=input["query"])
                news_results = response.get("news", [])
            except Exception as search_error:
                logging.error(f"Search API error: {str(search_error)}")
                return default_output
            
            # Process news results
            content = []
            for news in news_results:
                content.append(
                    ContentItem(
                        url=news.get("link", ""),
                        title=news.get("title", "") + " - " + news.get("date", ""),
                        snippet=news.get("snippet", ""),
                        content=""  # Placeholder for full content
                    )
                )
            
            # Generate summary if enabled
            summary = ""
            if self.include_summary:
                try:
                    # Prepare summary using the dynamically loaded prompt
                    formatted_content = "\n\n".join([
                        f"### {item.title}\n{item.snippet}"
                        for item in content
                    ])
                    
                    # Use the dynamically loaded prompt content
                    system_prompt = self.research_prompt["content"].replace(
                        "{{query}}", 
                        input["query"]
                    ).replace(
                        "{{content}}", 
                        formatted_content
                    )
                    
                    # Generate summary
                    summary = model_wrapper(
                        system_prompt=system_prompt,
                        user_prompt=f"Summarize the research on: '{input['query']}'",
                        model="llama3-70b-8192",
                        host="groq",
                        temperature=0.7
                    )
                    logging.info("Generated summary of research")
                except Exception as summary_error:
                    logging.error(f"Summary generation error: {str(summary_error)}")
                    summary = "Unable to generate summary."
            
            # Prepare final output
            output = ResearchToolOutput(
                content=content, 
                summary=summary
            )
            
            return output
        
        except Exception as e:
            # Catch-all error handling
            logging.error(f"Unexpected error in invoke method: {str(e)}")
            
            # Return a default output with error information
            default_output.summary = f"An error occurred: {str(e)}"
            return default_output

# Additional utility function for listing available prompts
def list_research_prompts():
    """
    List available research prompts.
    
    Returns:
        List of available prompt file names
    """
    return PromptLoader.list_available_prompts()