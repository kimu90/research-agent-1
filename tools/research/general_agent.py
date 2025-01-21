import os
import logging
import requests
from typing import Type, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import instructor
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
# Langchain and external imports
from langchain.tools import BaseTool
from langchain.docstore.document import Document
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import WebBaseLoader

# Pydantic and custom imports
from pydantic import BaseModel, Field
from prompts import Prompt
from .common.model_schemas import ContentItem, ResearchToolOutput
from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from utils.token_tracking import TokenUsageTracker  # Add this import

# Load environment variables
load_dotenv()

# API Keys
BROWSERLESS_API_KEY = os.getenv('BROWSERLESS_API_KEY')
SERPER_API_KEY = os.getenv('SERPER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

class GeneralAgentInput(BaseModel):
    query: str = Field(description="Search anything General")

# Define prompts at module level with proper initialization
SELECT_CONTENT_PROMPT = Prompt(
    id="research-agent-select-content",
    content="""Analyze the following news articles and select the most relevant ones:
    Research Topic: {{research_topic}}
    
    Available Articles:
    {{formatted_snippets}}
    
    Return the indices of the most relevant articles.""",
    metadata={"type": "content_selection"}
)

SUMMARIZE_RESULTS_PROMPT = Prompt(
    id="summarize-search-results",
    content="""Analyze and summarize the following search results:
    
    Query: {{user_prompt}}
    
    Search Results:
    {{search_results_str}}
    
    Provide a comprehensive summary grouped by themes and include relevant links.""",
    metadata={"type": "summarization"}
)

class GeneralAgent(BaseTool):
    name: str = "general-agent"
    description: str = "Invoke when user wants to search for news."
    args_schema: Type[BaseModel] = GeneralAgentInput
    include_summary: bool = False
    custom_prompt: Optional[Prompt] = Field(default=None)
    token_tracker: TokenUsageTracker = Field(default_factory=TokenUsageTracker)  # Add token tracker

    def __init__(self, include_summary: bool = False, custom_prompt: Optional[Prompt] = None):
        super().__init__()
        self.include_summary = include_summary
        self.custom_prompt = custom_prompt or SELECT_CONTENT_PROMPT
        self.token_tracker = TokenUsageTracker()  # Initialize token tracker

    def scrape_pages(self, urls: List[str]) -> List[Document]:
        """Scrape content from provided URLs using requests and BeautifulSoup"""
        logging.info(f"Starting to scrape {len(urls)} news pages")
        docs = []
        
        # Common headers to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }

        for url in urls:
            try:
                # Add delay between requests to be respectful
                time.sleep(2)
                
                # Get the domain for logging
                domain = urlparse(url).netloc
                logging.info(f"Attempting to scrape content from {domain}")
                
                # Enhanced request with more robust error handling
                response = requests.get(
                    url, 
                    headers=headers, 
                    timeout=15,  # Increased timeout
                    verify=True  # Keep SSL verification
                )
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for tag in soup.find_all(['script', 'style', 'meta', 'noscript', 'iframe', 'comment']):
                    tag.decompose()
                
                # Extract main content with multiple strategies
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
                            # For paragraph lists, filter and join meaningful text
                            content = '\n'.join([
                                p.get_text(strip=True) 
                                for p in result 
                                if len(p.get_text(strip=True).split()) > 5
                            ])
                        else:
                            content = result.get_text(separator='\n', strip=True)
                        
                        # Break if we get meaningful content
                        if len(content.strip()) > 200:
                            break
                
                # Clean up the content
                content = re.sub(r'\n{3,}', '\n\n', content)
                content = re.sub(r' {2,}', ' ', content)
                
                # Only add if we got meaningful content
                if len(content.strip()) > 200:  # Increased minimum content length
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
                continue
            except Exception as e:
                logging.error(f"Unexpected error scraping {url}: {str(e)}")
                continue

        logging.info(f"Successfully scraped {len(docs)} pages out of {len(urls)} attempted")
        return docs

    def decide_what_to_use(
        self, content: List[dict], research_topic: str
    ) -> List[dict]:
        """Decide which news articles to include based on relevance"""
        try:
            logging.info(f"Processing {len(content)} articles for topic: {research_topic}")
            
            formatted_snippets = ""
            for i, doc in enumerate(content):
                formatted_snippets += f"{i}: {doc['title']}: {doc['snippet']}\n"
            
            logging.info("Formatted snippets created")
            
            prompt_to_use = self.custom_prompt or SELECT_CONTENT_PROMPT
            
            system_prompt = prompt_to_use.compile(
                research_topic=research_topic, 
                formatted_snippets=formatted_snippets
            )
            
            logging.info("Compiled system prompt")

            class ModelResponse(BaseModel):
                snippet_indeces: List[int]

            # Add token tracking for the selection phase
            response = json_model_wrapper(
                system_prompt=system_prompt,
                user_prompt="Pick the snippets you want to include in the summary.",
                prompt=prompt_to_use,
                base_model=ModelResponse,
                model="gpt-3.5-turbo",
                temperature=0,
                token_tracker=self.token_tracker  # Pass token tracker
            )
            
            logging.info(f"Received response: {response}")
            
            if response is None or not hasattr(response, 'snippet_indeces'):
                logging.warning("No valid response received, using all articles")
                return content
                
            indices = [i for i in response.snippet_indeces if i < len(content)]
            
            if not indices:
                logging.warning("No valid indices found, using all articles")
                return content
                
            logging.info(f"Selected {len(indices)} articles from {len(content)} total results")
            return [content[i] for i in indices]
            
        except Exception as e:
            logging.error(f"Error in decide_what_to_use: {str(e)}")
            return content

    def _run(self, **kwargs) -> ResearchToolOutput:
        """Execute the news search tool"""
        logging.info(f"Starting news search for query: {kwargs['query']}")
        
        google_serper = GoogleSerperAPIWrapper(
            type="news", 
            k=10, 
            serper_api_key=SERPER_API_KEY
        )
        response = google_serper.results(query=kwargs["query"])
        news_results = response.get("news", [])

        for news in news_results:
            for field in ["snippet", "date", "source", "title", "link", "imageUrl"]:
                if field not in news:
                    news[field] = ""
        
        selected_results = self.decide_what_to_use(
            content=news_results, 
            research_topic=kwargs["query"]
        )

        webpage_urls = [result["link"] for result in selected_results]
        webpages = self.scrape_pages(webpage_urls)

        content = []
        for news in news_results:
            webpage = next(
                (
                    doc.page_content
                    for doc in webpages
                    if doc.metadata.get("source") == news["link"]
                ),
                "",
            )
            title = news.get("title", "") + " - " + news.get("date", "")
            content.append(
                ContentItem(
                    url=news["link"],
                    title=title,
                    snippet=news.get("text", ""),
                    content=webpage,
                )
            )

        summary = ""
        if self.include_summary:
            formatted_content = "\n\n".join([f"### {item}" for item in content])
            system_prompt = SUMMARIZE_RESULTS_PROMPT.compile(
                search_results_str=formatted_content, 
                user_prompt=kwargs["query"]
            )

            # Add token tracking for the summarization phase
            summary = model_wrapper(
                system_prompt=system_prompt,
                prompt=SUMMARIZE_RESULTS_PROMPT,
                user_prompt=f"Summarize and group the search results based on this: '{kwargs['query']}'. Include links, dates, and snippets from the search results.",
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
                token_tracker=self.token_tracker  # Pass token tracker
            )
            logging.info("Generated summary of news articles")

        return ResearchToolOutput(content=content, summary=summary)