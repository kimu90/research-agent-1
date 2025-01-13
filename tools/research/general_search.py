# tools/research/general_search.py

from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field
from typing import Type, List
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage
from .google_scraper import WebScraper
import logging

class GeneralSearchInput(BaseModel):
    query: str = Field(description="Search query for general research")

class GeneralSearch(BaseTool):
    name: str = "general-search"
    description: str = "Comprehensive research tool for general topics and analysis"
    args_schema: Type[BaseModel] = GeneralSearchInput
    include_summary: bool = Field(default=False)
    web_scraper: WebScraper = Field(default_factory=WebScraper)  # Add this line

    GENERAL_PROMPT = """Utilizing the provided query, your task is to conduct detailed research and produce a factual and data-backed report. Your goal is to gather comprehensive information and provide a thorough analysis based on the data collected. 

Please complete the objective with the following guidelines:
1. Conduct extensive research to gather as much information as possible about the topic.
2. After initial research, evaluate if further searches are needed based on the data collected.
3. Ensure that all information provided is factual and based on collected data.
4. In the final output, include a markdown header '## Research Summary:' followed by a comprehensive summary of the findings.
5. At the end of the summary, include a markdown header '## References:' with all sources used to back up your research.

Ensure the final output is clear, accurate, and fully substantiated by the gathered data."""

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def execute_research(self, query: str) -> str:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=1000)
        memory = ConversationSummaryBufferMemory(
            memory_key="memory",
            return_messages=True,
            llm=llm,
            max_token_limit=1000
        )

        system_message = SystemMessage(content=self.GENERAL_PROMPT)
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_message,
        }

        tools = [
            Tool(
                name="Search",
                func=self.web_scraper.search,
                description="Useful for when you need to answer questions about current events and data"
            ),
            self.web_scraper
        ]

        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            agent_kwargs=agent_kwargs,
            memory=memory,
        )

        result = agent({"input": query})
        return result['output']

    def _run(self, **kwargs) -> ResearchToolOutput:
        """
        Execute the general research tool
        """
        logging.info(f"Starting general research for query: {kwargs['query']}")
        
        try:
            # Get research results
            research_output = self.execute_research(kwargs["query"])
            
            # Create a single ContentItem for the research results
            content = [
                ContentItem(
                    url="",
                    title="General Research Results",
                    snippet="Comprehensive research findings",
                    content=research_output,
                    source="General Research Agent"
                )
            ]
            
            return ResearchToolOutput(
                content=content,
                summary=research_output
            )
            
        except Exception as e:
            logging.error(f"Error in general research: {str(e)}")
            return ResearchToolOutput(
                content=[],
                summary=f"Error performing research: {str(e)}"
            )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async version not implemented")