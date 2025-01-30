import os, logging, requests, time
from typing import Type, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from prompt import Prompt
from .common.model_schemas import ContentItem, ResearchToolOutput
from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from utils.token_tracking import TokenUsageTracker

load_dotenv()
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class PromptLoader:
    """
    Handles dynamic loading of prompts from a specified directory.
    """
    @staticmethod
    def load_prompt(prompt_name: str = "research.txt") -> Prompt:
        """
        Load a prompt from the prompts directory.
        
        Args:
            prompt_name: Name of the prompt file to load (default: research.txt)
        
        Returns:
            Prompt: Loaded prompt object
        """
        try:
            with open(os.path.join("/app/prompts", prompt_name), 'r') as f:
                prompt_content = f.read()
            
            return Prompt(
                id=f"dynamic-prompt-{prompt_name}",
                content=prompt_content,
                metadata={"source": prompt_name}
            )
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {prompt_name}")
            return Prompt(
                id="fallback-prompt",
                content="""Analyze the following content and provide a comprehensive summary.
                
                Research Topic: {{research_topic}}
                
                Content:
                {{content}}
                
                Provide key insights, main themes, and relevant conclusions."""
            )
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

class GeneralAgentInput(BaseModel):
    query: str = Field(description="Search anything General")

class GeneralAgent(BaseTool):
    name: str = "general-agent"
    description: str = "Invoke when user wants to search for news."
    args_schema: Type[BaseModel] = GeneralAgentInput
    include_summary: bool = False
    custom_prompt: Optional[Prompt] = Field(default=None)
    token_tracker: TokenUsageTracker = Field(default_factory=TokenUsageTracker)
    current_prompt: Optional[Prompt] = Field(default=None)  # Add this line

    def __init__(
        self, 
        include_summary: bool = False, 
        custom_prompt: Optional[Prompt] = None,
        prompt_name: Optional[str] = None
    ):
        super().__init__()
        self.include_summary = include_summary
        # Determine which prompt to use
        if custom_prompt:
            # Custom prompt takes highest precedence
            self.current_prompt = custom_prompt
        elif prompt_name:
            # Load prompt from file if prompt name is provided
            try:
                self.current_prompt = PromptLoader.load_prompt(prompt_name)
            except Exception as e:
                logging.warning(f"Failed to load prompt {prompt_name}, falling back to default. Error: {e}")
                self.current_prompt = PromptLoader.load_prompt("research.txt")
        else:
            # Use default content selection prompt if no custom prompt or name is provided
            self.current_prompt = Prompt(
                id="default-content-selection",
                content="""Analyze the following news articles and select the most relevant ones:
                Research Topic: {{research_topic}}
                
                Available Articles:
                {{formatted_snippets}}
                
                Return the indices of the most relevant articles."""
            )
        
        # Initialize token tracker
        self.token_tracker = TokenUsageTracker()

    def decide_what_to_use(self, content: List[dict], research_topic: str) -> List[dict]:
        try:
            logging.info(f"Processing {len(content)} articles for topic: {research_topic}")
            
            formatted_snippets = "\n".join([f"{i}: {doc['title']}: {doc['snippets'][0]}" for i, doc in enumerate(content)])
            
            # Use the current prompt for content selection
            system_prompt = self.current_prompt.compile(
                research_topic=research_topic, 
                formatted_snippets=formatted_snippets
            )
            
            class ModelResponse(BaseModel):
                sources: List[int]  # Expect indices (integers)

            response = json_model_wrapper(
                system_prompt=system_prompt,
                user_prompt="Pick the snippets you want to include in the summary.",
                prompt=self.current_prompt,
                base_model=ModelResponse,
                model="gpt-3.5-turbo",
                temperature=0,
                token_tracker=self.token_tracker
            )
            
            logging.info(f"Received response: {response}")
            
            if response is None or not hasattr(response, 'sources'):
                logging.warning("No valid response received, using all articles")
                return content
                
            # Ensure that the 'sources' are indices (integers), and extract them correctly
            indices = []
            for source in response.sources:
                if isinstance(source, dict) and 'index' in source:
                    indices.append(source['index'])
                elif isinstance(source, int):
                    indices.append(source)
            
            # Filter valid indices
            indices = [i for i in indices if i < len(content)]
            
            if not indices:
                logging.warning("No valid indices found, using all articles")
                return content
                
            logging.info(f"Selected {len(indices)} articles from {len(content)} total results")
            return [content[i] for i in indices]
            
        except Exception as e:
            logging.error(f"Error in decide_what_to_use: {str(e)}")
            return content

    def scrape_pages(self, urls: List[str]) -> List[Document]:
        logging.info(f"Starting to scrape {len(urls)} news pages")
        docs = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }

        for url in urls:
            try:
                time.sleep(2)
                domain = urlparse(url).netloc
                logging.info(f"Attempting to scrape content from {domain}")
                response = requests.get(url, headers=headers, timeout=15, verify=True)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                for tag in soup.find_all(['script', 'style', 'meta', 'noscript', 'iframe', 'comment']):
                    tag.decompose()
                
                content_strategies = [
                    lambda: soup.find('article'),
                    lambda: soup.find(['main', 'div'], class_=re.compile(r'(content|article|post|story)')),
                    lambda: soup.find_all('p'),
                ]
                
                content = ""
                for strategy in content_strategies:
                    result = strategy()
                    if result:
                        if isinstance(result, list):
                            content = '\n'.join([p.get_text(strip=True) for p in result if len(p.get_text(strip=True).split()) > 5])
                        else:
                            content = result.get_text(separator='\n', strip=True)
                        if len(content.strip()) > 200:
                            break
                
                content = re.sub(r'\n{3,}', '\n\n', content)
                content = re.sub(r' {2,}', ' ', content)
                
                if len(content.strip()) > 200:
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "title": soup.title.string if soup.title else "No title",
                            "length": len(content)
                        }
                    ))
                    logging.info(f"Successfully scraped {url} ({len(content)} chars)")
                else:
                    logging.warning(f"Retrieved content too short from {url}")
                
            except requests.exceptions.RequestException as e:
                logging.error(f"Error scraping {url}: {str(e)}")
            except Exception as e:
                logging.error(f"Unexpected error scraping {url}: {str(e)}")

        logging.info(f"Successfully scraped {len(docs)} pages out of {len(urls)} attempted")
        return docs

    def _run(self, **kwargs) -> ResearchToolOutput:
        logging.info(f"Starting news search for query: {kwargs['query']}")
        
        google_serper = GoogleSerperAPIWrapper(type="news", k=10, serper_api_key=SERPER_API_KEY)
        response = google_serper.results(query=kwargs["query"])
        news_results = response.get("news", [])

        for news in news_results:
            for field in ["snippet", "date", "source", "title", "link", "imageUrl"]:
                if field not in news:
                    news[field] = ""
        
        selected_results = self.decide_what_to_use(content=news_results, research_topic=kwargs["query"])

        webpage_urls = [result["link"] for result in selected_results]
        webpages = self.scrape_pages(webpage_urls)

        content = []
        for news in news_results:
            webpage = next((doc.page_content for doc in webpages if doc.metadata.get("source") == news["link"]), "")
            title = news.get("title", "") + " - " + news.get("date", "")
            content.append(ContentItem(
                url=news["link"],
                title=title,
                snippet=news.get("text", ""),
                content=webpage,
            ))

        summary = ""
        if self.include_summary:
            formatted_content = "\n\n".join([f"### {item}" for item in content])
            
            # Use the current prompt for summary generation
            summary_prompt = Prompt(
                id="dynamic-summary-prompt",
                content="""Analyze and summarize the following search results:
                
                Query: {{user_prompt}}
                
                Search Results:
                {{search_results_str}}
                
                Provide a comprehensive summary grouped by themes and include relevant links."""
            )
            
            system_prompt = summary_prompt.compile(
                search_results_str=formatted_content, 
                user_prompt=kwargs["query"]
            )

            summary = model_wrapper(
                system_prompt=system_prompt,
                prompt=summary_prompt,
                user_prompt=f"Summarize and group the search results based on this: '{kwargs['query']}'. Include links, dates, and snippets from the search results.",
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
                token_tracker=self.token_tracker
            )
            logging.info("Generated summary of news articles")

        return ResearchToolOutput(content=content, summary=summary)
