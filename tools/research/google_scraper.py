# tools/research/google_scraper.py

import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.tools import BaseTool
from .common.model_schemas import ScrapeWebsiteInput

brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

class WebScraper(BaseTool):
    name = "scrape_google_search_results"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def summary(self, objective, content):
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
        )
        docs = text_splitter.create_documents([content])
        map_prompt = """
        Write a summary of the following text for {objective}. The text is Scraped data from a website so 
        will have a lot of usless information that doesnt relate to this topic, links, other news stories etc.. 
        Only summarise the relevant Info and try to keep as much factual information Intact:
        "{text}"
        SUMMARY:
        """
        map_prompt_template = PromptTemplate(
            template=map_prompt, input_variables=["text", "objective"]
        )
        summary_chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=map_prompt_template,
            verbose=True,
        )
        output = summary_chain.run(input_documents=docs, objective=objective)
        return output

    @staticmethod
    def search(query: str):
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query})
        headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.text

    def scrape_website(self, objective: str, url: str):
        headers = {
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
        }
        data = {"url": url}
        data_json = json.dumps(data)
        post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            if len(text) > 10000:
                output = self.summary(objective, text)
                return output
            else:
                return text
        else:
            print(f"HTTP request failed with status code {response.status_code}")
    
    def _run(self, objective: str, url: str):
        return self.scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")