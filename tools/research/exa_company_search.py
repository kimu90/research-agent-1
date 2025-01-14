from .common.model_schemas import ContentItem, ResearchToolOutput
from .base_tool import ResearchTool
from langchain.tools import BaseTool

from langchain_community.document_loaders import WebBaseLoader
from utils.model_wrapper import model_wrapper
from pydantic import BaseModel, Field
from typing import Type, List
from prompts import Prompt

import logging
import requests
import os

class ExaCompanySearchInput(BaseModel):
    query: str = Field(description="Search query for companies")

class ExaCompanySearch(BaseTool):
    name: str = "exa-company-search"
    description: str = "Invoke when the user wants to search one or multiple companies. This tool only finds companies that might fit the user request and returns only company urls and landingpage summaries. No other data. This tool cannot compare companies or find similar companies."
    args_schema: Type[BaseModel] = ExaCompanySearchInput
    include_summary: bool = False

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def scrape_pages(self, urls: List[str]):
        """
        Scrape content from provided URLs using WebBaseLoader
        """
        logging.info(f"Starting to scrape {len(urls)} pages")
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
            logging.info(f"Successfully scraped {len(docs)} pages")
        except Exception as error:
            logging.error(f"Error scraping content: {error}")
            docs = []
        return docs

    def _run(self, **kwargs) -> ResearchToolOutput:
        """
        Execute the company search tool
        """
        logging.info(f"Executing company search for query: {kwargs['query']}")
        
        # Prepare API request
        url = "https://api.exa.ai/search"
        payload = {
            "category": "company",
            "query": kwargs["query"],
            "contents": {"text": {"includeHtmlTags": False}},
            "numResults": 3,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": os.getenv("EXA_API_KEY"),
        }

        # Execute search
        response = requests.post(url, json=payload, headers=headers)
        urls = [result["url"] for result in response.json()["results"]]
        
        # Scrape additional content
        webpages = self.scrape_pages(urls)

        # Process results
        content = []
        for result in response.json()["results"]:
            webpage = next(
                (
                    doc.page_content
                    for doc in webpages
                    if doc.metadata.get("source") == result["url"]
                ),
                "",
            )
            title = result.get("title", "") + " - " + result.get("publishedDate", "")
            content.append(
                ContentItem(
                    url=result["url"],
                    title=title,
                    snippet=result.get("text", ""),
                    content=webpage,
                    source="Exa AI",
                )
            )

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
                user_prompt=kwargs["query"],
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
            )
            logging.info("Generated summary from search results")

        return ResearchToolOutput(content=content, summary=summary)