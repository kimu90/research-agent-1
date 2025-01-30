from typing import Optional, Any, Type, Dict
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import logging
import json
from urllib.parse import quote_plus
from .common.model_schemas import ContentItem, ResearchToolOutput
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time
import re

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

class WebScraperInput(BaseModel):
    """Input for WebScraper"""
    url: str = Field(description="URL to scrape")
    search_term: Optional[str] = Field(default="", description="Search term to look for")

class WebScraper(BaseTool):
    name: str = Field(default="web_scraper")
    description: str = Field(default="A tool for scraping web content from URLs and performing Google searches")
    args_schema: Type[BaseModel] = WebScraperInput
    
    def __init__(self):
        super().__init__()
        self.chrome_options = Options()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')

    def _setup_driver(self):
        return webdriver.Chrome(options=self.chrome_options)

    def search(self, query: str) -> str:
        """
        Perform a Google search and return formatted results
        """
        try:
            driver = self._setup_driver()
            search_url = f"https://www.google.com/search?q={quote_plus(query)}"
            driver.get(search_url)
            
            # Wait for search results to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "g"))
            )
            
            # Parse the page with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            search_results = []
            
            # Find all search result divs
            for result in soup.find_all('div', class_='g'):
                title_element = result.find('h3')
                link_element = result.find('a')
                snippet_element = result.find('div', class_='VwiC3b')
                
                if title_element and link_element and snippet_element:
                    title = title_element.text
                    link = link_element['href']
                    snippet = snippet_element.text
                    
                    if link.startswith('http'):
                        search_results.append({
                            "title": title,
                            "url": link,
                            "snippet": snippet
                        })
            
            driver.quit()
            return json.dumps(search_results[:5], ensure_ascii=False)
            
        except Exception as e:
            logging.error(f"Error in Google search: {str(e)}")
            return json.dumps([])
        
    def scrape_url(self, url: str) -> Optional[str]:
        """
        Scrape content from a given URL
        """
        try:
            driver = self._setup_driver()
            driver.get(url)
            
            # Wait for body to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Get page content
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            driver.quit()
            return text
            
        except Exception as e:
            logging.error(f"Error scraping URL {url}: {str(e)}")
            return None

    def _run(self, url: str, search_term: str = "") -> ResearchToolOutput:
        """
        Execute the web scraping tool
        """
        try:
            content = []
            
            if url.startswith('http'):
                # Scrape specific URL
                scraped_content = self.scrape_url(url)
                if scraped_content:
                    content.append(
                        ContentItem(
                            url=url,
                            title=url.split('/')[-1],
                            snippet=scraped_content[:200] + "...",
                            content=scraped_content,
                            source="Web Scraper"
                        )
                    )
            else:
                # Treat input as search query
                search_results = json.loads(self.search(url))
                for result in search_results:
                    scraped_content = self.scrape_url(result['url'])
                    if scraped_content:
                        content.append(
                            ContentItem(
                                url=result['url'],
                                title=result['title'],
                                snippet=result['snippet'],
                                content=scraped_content,
                                source="Web Scraper"
                            )
                        )
            
            return ResearchToolOutput(
                content=content,
                summary=f"Scraped {len(content)} pages successfully"
            )
            
        except Exception as e:
            logging.error(f"Error in web scraper: {str(e)}")
            return ResearchToolOutput(
                content=[],
                summary=f"Error scraping content: {str(e)}"
            )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async version not implemented")