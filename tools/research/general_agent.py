from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool

from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from typing import Type, List, Optional
from prompts import Prompt

import logging
import os

class GeneralAgentInput(BaseModel):
    query: str = Field(description="Search anything General")

class GeneralAgent(BaseTool):
    name: str = "general-agent"
    description: str = "Invoke when user wants to search for news."
    args_schema: Type[BaseModel] = GeneralAgentInput
    include_summary: bool = False

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def scrape_pages(self, urls: List[str]):
        """
        Scrape content from provided URLs using WebBaseLoader
        """
        logging.info(f"Starting to scrape {len(urls)} news pages")
        loader = WebBaseLoader(
            urls,
            proxies={
                scheme: f"http://{os.getenv('ZYTE_API_KEY')}:@api.zyte.com:8011"
                for scheme in ("http", "https")
            },
        )
        loader.requests_per_second = 5
        try:
            docs = loader.aload()
            for doc in docs:
                while "\n\n" in doc.page_content:
                    doc.page_content = doc.page_content.replace("\n\n", "\n")
                while "  " in doc.page_content:
                    doc.page_content = doc.page_content.replace("  ", " ")
            logging.info(f"Successfully scraped {len(docs)} news pages")
        except Exception as error:
            logging.error(f"Error scraping news content: {error}")
            docs = []
        return docs

    def decide_what_to_use(
        self, content: List[dict], research_topic: str
    ) -> List[dict]:
        """
        Decide which news articles to include based on relevance
        """
        select_content = Prompt("research-agent-select-content")
        formatted_snippets = ""
        for i, doc in enumerate(content):
            formatted_snippets += f"{i}: {doc['title']}: {doc['snippet']}\n"

        system_prompt = select_content.compile(
            research_topic=research_topic, 
            formatted_snippets=formatted_snippets
        )

        class ModelResponse(BaseModel):
            snippet_indeces: List[int]

        response: ModelResponse = json_model_wrapper(
            system_prompt=system_prompt,
            user_prompt="Pick the snippets you want to include in the summary.",
            prompt=select_content,
            base_model=ModelResponse,
        )

        indices = [i for i in response.snippet_indeces if i < len(content)]
        logging.info(f"Selected {len(indices)} articles from {len(content)} total results")
        return [content[i] for i in indices]

    def _run(self, **kwargs) -> ResearchToolOutput:
        """
        Execute the news search tool
        """
        logging.info(f"Starting news search for query: {kwargs['query']}")
        
        # Execute Google Serper search
        google_serper = GoogleSerperAPIWrapper(type="news", k=10)
        response = google_serper.results(query=kwargs["query"])
        news_results = response["news"]

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
            summarize_search_results = Prompt("summarize-search-results")
            formatted_content = "\n\n".join([f"### {item}" for item in content])
            system_prompt = summarize_search_results.compile(
                search_results_str=formatted_content, 
                user_prompt=kwargs["query"]
            )

            summary = model_wrapper(
                system_prompt=system_prompt,
                prompt=summarize_search_results,
                user_prompt=f"Summarize and group the search results based on this: '{kwargs['query']}'. Include links, dates, and snippets from the search results.",
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
            )
            logging.info("Generated summary of news articles")

        return ResearchToolOutput(content=content, summary=summary)