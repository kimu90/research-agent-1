import os
import logging
import requests
from typing import Type, List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import instructor
import time

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
    custom_prompt: Optional[Prompt] = Field(default=None)  # Add this line to declare the field

    def __init__(self, include_summary: bool = False, custom_prompt: Optional[Prompt] = None):
        super().__init__()
        self.include_summary = include_summary
        self.custom_prompt = custom_prompt or SELECT_CONTENT_PROMPT  # Use default if none provided

    def scrape_pages(self, urls: List[str]) -> List[Document]:
        """Scrape content from provided URLs using Browserless API"""
        logging.info(f"Starting to scrape {len(urls)} news pages")
        docs = []

        for url in urls:
            try:
                # Use Browserless API to fetch page content
                response = requests.get(
                    f"https://api.browserless.io/content",
                    params={"token": BROWSERLESS_API_KEY},
                    json={"url": url}
                )
                response.raise_for_status()
                
                page_content = response.text
                
                # Clean up the content
                while "\n\n" in page_content:
                    page_content = page_content.replace("\n\n", "\n")
                while "  " in page_content:
                    page_content = page_content.replace("  ", " ")
                
                docs.append(Document(page_content=page_content, metadata={"source": url}))
                logging.info(f"Successfully scraped {url}")
            
            except Exception as error:
                logging.error(f"Error scraping {url}: {error}")
        
        logging.info(f"Successfully scraped {len(docs)} news pages")
        return docs

    def decide_what_to_use(
        self, content: List[dict], research_topic: str
    ) -> List[dict]:
        """Decide which news articles to include based on relevance"""
        try:
            # Add logging to see what's coming in
            logging.info(f"Processing {len(content)} articles for topic: {research_topic}")
            
            formatted_snippets = ""
            for i, doc in enumerate(content):
                formatted_snippets += f"{i}: {doc['title']}: {doc['snippet']}\n"
            
            logging.info("Formatted snippets created")
            
            # Use custom prompt if available, otherwise use default
            prompt_to_use = self.custom_prompt or SELECT_CONTENT_PROMPT
            
            system_prompt = prompt_to_use.compile(
                research_topic=research_topic, 
                formatted_snippets=formatted_snippets
            )
            
            logging.info("Compiled system prompt")

            class ModelResponse(BaseModel):
                snippet_indeces: List[int]

            # Add error handling for json_model_wrapper
            response = json_model_wrapper(
                system_prompt=system_prompt,
                user_prompt="Pick the snippets you want to include in the summary.",
                prompt=prompt_to_use,
                base_model=ModelResponse,
                model="gpt-3.5-turbo",
                temperature=0
            )
            
            logging.info(f"Received response: {response}")
            
            # Add validation and fallback
            if response is None or not hasattr(response, 'snippet_indeces'):
                logging.warning("No valid response received, using all articles")
                return content  # Return all content as fallback
                
            indices = [i for i in response.snippet_indeces if i < len(content)]
            
            if not indices:  # If no valid indices, return all content
                logging.warning("No valid indices found, using all articles")
                return content
                
            logging.info(f"Selected {len(indices)} articles from {len(content)} total results")
            return [content[i] for i in indices]
            
        except Exception as e:
            logging.error(f"Error in decide_what_to_use: {str(e)}")
            # Return all content as fallback in case of error
            return content

    def _run(self, **kwargs) -> ResearchToolOutput:
        """Execute the news search tool"""
        logging.info(f"Starting news search for query: {kwargs['query']}")
        
        # Execute Serper search
        google_serper = GoogleSerperAPIWrapper(
            type="news", 
            k=10, 
            serper_api_key=SERPER_API_KEY
        )
        response = google_serper.results(query=kwargs["query"])
        news_results = response.get("news", [])

        # Normalize results data
        for news in news_results:
            for field in ["snippet", "date", "source", "title", "link", "imageUrl"]:
                if field not in news:
                    news[field] = ""
        
        # Select relevant articles
        selected_results = self.decide_what_to_use(
            content=news_results, 
            research_topic=kwargs["query"]
        )

        # Scrape full content
        webpage_urls = [result["link"] for result in selected_results]
        webpages = self.scrape_pages(webpage_urls)

        # Process content
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

        # Generate summary if requested
        summary = ""
        if self.include_summary:
            formatted_content = "\n\n".join([f"### {item}" for item in content])
            system_prompt = SUMMARIZE_RESULTS_PROMPT.compile(
                search_results_str=formatted_content, 
                user_prompt=kwargs["query"]
            )

            summary = model_wrapper(
                system_prompt=system_prompt,
                prompt=SUMMARIZE_RESULTS_PROMPT,
                user_prompt=f"Summarize and group the search results based on this: '{kwargs['query']}'. Include links, dates, and snippets from the search results.",
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
            )
            logging.info("Generated summary of news articles")

        return ResearchToolOutput(content=content, summary=summary)