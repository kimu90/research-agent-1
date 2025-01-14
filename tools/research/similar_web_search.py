from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool

from utils.model_wrapper import model_wrapper
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from prompts import Prompt
from typing import Type, Optional

import requests
import logging
import os

class SimilarWebSearchInput(BaseModel):
    entity_name: str = Field(description="Entity name to search for on SimilarWeb.")
    instructions: Optional[str] = Field(
        default=None, 
        description="Instructions on how to process the raw text."
    )

class SimilarWebSearch(BaseTool):
    name: str = "similar-web-search"
    description: str = "Search for a website on SimilarWeb and generate a detailed report."
    args_schema: Type[BaseModel] = SimilarWebSearchInput
    user_prompt: Optional[str] = None
    include_summary: bool = False

    def __init__(
        self,
        include_summary: bool = False,
        user_prompt: str = "",
    ):
        super().__init__()
        self.include_summary = include_summary
        self.user_prompt = user_prompt

    def brave_search(self, query: str, count: int) -> dict:
        """
        Perform a search query using Brave Search API.

        Args:
            query: The search query string
            count: Number of results to return

        Returns:
            Search results as a JSON object
        """
        logging.info(f"Performing Brave search for: {query}")
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": os.getenv("BRAVE_SEARCH_API_KEY"),
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }
        params = {"q": query, "count": count}

        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            logging.info("Brave search successful")
            return response.json()
        else:
            logging.error(f"Brave search failed with status code: {response.status_code}")
            return {
                "error": "Failed to fetch search results",
                "status_code": response.status_code,
            }

    def _run(self, **kwargs) -> ResearchToolOutput:
        """
        Execute the SimilarWeb search tool
        """
        entity_name = kwargs.get("entity_name")
        instructions = kwargs.get("instructions")
        
        logging.info(f"Starting SimilarWeb search for entity: {entity_name}")

        # Get domain from Brave search
        search_results = self.brave_search(entity_name + " website", count=1)
        result = search_results["web"]["results"][0]
        domain = result["url"].split("/")[2]

        # Fetch SimilarWeb data
        url = f"https://www.similarweb.com/website/{domain}/#overview"
        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(os.getenv("ZYTE_API_KEY"), ""),
            json={"url": url, "browserHtml": True},
        )

        text = ""
        if response.status_code == 200:
            logging.info("Successfully fetched SimilarWeb data")
            soup = BeautifulSoup(response.json().get("browserHtml", ""), "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            # Generate snippet
            generate_paragraph = Prompt("summarize-text-into-three-paragraphs")
            snippet = model_wrapper(
                system_prompt=generate_paragraph.compile(text=text),
                prompt=generate_paragraph,
                user_prompt=f"Generate a snippet based on the given text:\n{text}",
                model="gpt-3.5-turbo-1106",
                temperature=0.7,
            )

            content = [
                ContentItem(
                    url=url,
                    title=f"SimilarWeb data for {entity_name}",
                    snippet=snippet,
                    content=text,
                    source="SimilarWeb",
                )
            ]
            logging.info("Generated snippet from SimilarWeb data")
        else:
            logging.error(f"Failed to fetch SimilarWeb data: {response.status_code}")
            content = []

        # Generate summary if requested
        summary = ""
        if self.include_summary and len(content) > 0:
            summarize_similarweb = Prompt("summarize-similarweb-search-result")
            system_prompt = summarize_similarweb.compile(
                text=text, 
                instructions=instructions, 
                user_prompt=self.user_prompt
            )
            summary = model_wrapper(
                system_prompt=system_prompt,
                user_prompt="Generate a detailed report based on the given text.",
                prompt=summarize_similarweb,
                temperature=0.7,
            )
            logging.info("Generated summary report")

        return ResearchToolOutput(content=content, summary=summary)