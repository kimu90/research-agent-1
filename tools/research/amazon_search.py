# tools/research/amazon_search.py

from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List
import requests
import logging
import os

class AmazonSearchInput(BaseModel):
    query: str = Field(description="Search query for Amazon products")

class AmazonSearch(BaseTool):
    name: str = "amazon-search"
    description: str = "Search for products on Amazon and get details including prices, ratings and reviews."
    args_schema: Type[BaseModel] = AmazonSearchInput
    include_summary: bool = Field(default=False)
    rainforest_api_key: str = Field(default_factory=lambda: os.getenv("RAINFOREST_API_KEY"))

    def amazon_product_search(self, query: str) -> List[dict]:
        """Perform a search query using Rainforest API"""
        logging.info(f"Performing Amazon search for: {query}")
        
        params = {
            'api_key': self.rainforest_api_key,
            'type': 'search',
            'amazon_domain': 'amazon.com',
            'search_term': query,
            'sort_by': 'price_high_to_low'
        }

        try:
            response = requests.get('https://api.rainforestapi.com/request', params=params)
            response.raise_for_status()
            return response.json().get('search_results', [])
        except Exception as e:
            logging.error(f"Error in Amazon search: {str(e)}")
            return []

    def format_summary(self, results: List[dict], query: str) -> str:
        """Format the results into the required markdown summary"""
        if not results:
            return """## Commercial Products: FALSE

## Summary:
No commercial products related to the search query were found.

## Commercial Products Details:
No products found.

## References:
None available."""

        # Format product details
        product_details = []
        references = []
        for i, item in enumerate(results):
            asin = item.get('asin', 'N/A')
            price = item.get('price', {}).get('value', 'N/A')
            title = item.get('title', 'N/A')
            
            if all(x != 'N/A' for x in [asin, price, title]):
                product_details.append(f"* {title} (ASIN: {asin}) - ${price}")
                references.append(f"    {i+1}. Amazon product {asin}: https://www.amazon.com/dp/{asin}")

        # Create summary
        summary = "## Commercial Products: TRUE\n\n"
        summary += "## Summary:\n"
        summary += f"A search for '{query}' revealed several commercial products available on Amazon. "
        summary += f"The search returned {len(results)} products with varying price points and specifications. "
        summary += "These products demonstrate commercial applications and market availability.\n\n"
        
        # Add price analysis if possible
        try:
            prices = [float(i.get('price', {}).get('value', 0)) for i in results if i.get('price', {}).get('value')]
            if prices:
                avg_price = sum(prices) / len(prices)
                summary += f"The average price point for these products is ${avg_price:.2f}, "
                summary += f"ranging from ${min(prices):.2f} to ${max(prices):.2f}.\n\n"
        except Exception as e:
            logging.error(f"Error calculating price statistics: {str(e)}")

        # Add product details
        summary += "## Commercial Products Details:\n"
        summary += "\n".join(product_details)
        summary += "\n\n## References:\n"
        summary += "\n".join(references)

        return summary

    def _run(self, **kwargs) -> ResearchToolOutput:
        """Execute the Amazon search tool"""
        logging.info(f"Starting Amazon search for query: {kwargs['query']}")
        
        # Get search results
        results = self.amazon_product_search(kwargs["query"])
        
        # Process results for ContentItem list
        content = []
        for item in results:
            content_text = f"""
            Price: ${item.get('price', {}).get('value', 'N/A')}
            Rating: {item.get('rating', 'N/A')}
            Total Reviews: {item.get('ratings_total', 'N/A')}
            ASIN: {item.get('asin', 'N/A')}
            Features: {' | '.join(item.get('features', ['No features listed']))}
            """
            
            content.append(
                ContentItem(
                    url=f"https://www.amazon.com/dp/{item.get('asin', '')}",
                    title=item.get('title', ''),
                    snippet=item.get('snippet', ''),
                    content=content_text,
                    source="Amazon"
                )
            )
        
        # Generate formatted summary
        summary = self.format_summary(results, kwargs["query"])
        
        return ResearchToolOutput(content=content, summary=summary)

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async version not implemented")