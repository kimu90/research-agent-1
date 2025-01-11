from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool
from utils.model_wrapper import model_wrapper
from langchain.pydantic_v1 import BaseModel, Field
from prompts import Prompt
from typing import Type, Optional

import requests
import logging
import os

class YouComSearchInput(BaseModel):
    query: str = Field(description="Search query for You.com")

class YouComSearch(BaseTool):
    name: str = "you-com-search"
    description: str = "Invoke when the user asks a general question. It works like Google Search. Don't use this to search for companies."
    args_schema: Type[BaseModel] = YouComSearchInput
    include_summary: bool = False

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def you_com_search(self, query: str) -> dict:
        """
        Perform a search query using You.com API 
        
        Args:
            query: The search query string
        
        Returns:
            Search results as a JSON object
        """
        logging.info(f"Performing You.com search for: {query}")
        headers = {"X-API-Key": os.environ["YOUCOM_API_KEY"]}
        params = {"query": query}

        response = requests.get(
            f"https://api.ydc-index.io/rag?query={query}",
            params=params,
            headers=headers,
        )

        if response.status_code == 200:
            logging.info("You.com search successful")
            return response.json()
        else:
            logging.error(f"You.com search failed with status code: {response.status_code}")
            return {"hits": []}

    def _run(self, **kwargs) -> ResearchToolOutput:
        """
        Execute the You.com search tool
        """
        logging.info(f"Starting You.com search for query: {kwargs['query']}")

        # Get search results
        result = self.you_com_search(kwargs["query"])

        # Process results
        content = []
        for hit in result.get("hits", []):
            if isinstance(hit["ai_snippets"], str):
                hit["ai_snippets"] = [hit["ai_snippets"]]

            content.append(
                ContentItem(
                    url=hit["url"],
                    title=hit["title"],
                    snippet=hit["snippet"],
                    content="\n".join(hit["ai_snippets"]),
                )
            )
        logging.info(f"Processed {len(content)} search results")

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
                user_prompt=kwargs["query"],
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
            )
            logging.info("Generated summary of search results")

        return ResearchToolOutput(content=content, summary=summary)