# tools/research/marine_search.py

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

class MarineSearchInput(BaseModel):
    query: str = Field(description="Search query for marine research")

class MarineSearch(BaseTool):
    name: str = "marine-search"
    description: str = "Advanced marine science research tool"
    args_schema: Type[BaseModel] = MarineSearchInput
    include_summary: bool = Field(default=False)
    web_scraper: WebScraper = Field(default_factory=WebScraper)  # Add this line

    MARINE_PROMPT = """You are a world-class marine scientific researcher, who can do detailed research on marine natural products, compounds, genetics, and produce fact-based results.

Please make sure you complete the research with the following rules:
1. You should do enough research to gather as much information as possible about the objective.
2. You should search & scrape based on the data you have gathered to increase research quality.
3. You should check for products, prices, and other relevant information when gathering data.
4. If asked to search for products, look for enzymes, luciferases, biotechnology, chemical compounds, natural products, drugs, cosmetics, pharmaceuticals, medicines, nutraceuticals, supplements, fisheries, foods, and other useful things.
5. Include details of known distribution of organisms, habitat, and ecology in the summary.
6. After scraping and searching, evaluate if further research is needed to increase quality.
7. List up to ten products, prices, and other relevant information found.
8. Only write facts & data that you have gathered.
9. Write in the third person.
10. Begin with '## Organism: scientific name' if applicable.
11. Include '## Vernacular: common name' if applicable.
12. Start with '## Commercial Products: TRUE/FALSE'.
13. Follow with '## Summary:'.
14. If TRUE, add '## Commercial Products Details:'.
15. End with '## References:' and numbered list."""

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def execute_research(self, query: str) -> str:
        llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        memory = ConversationSummaryBufferMemory(
            memory_key="memory",
            return_messages=True,
            llm=llm,
            max_token_limit=1000
        )

        system_message = SystemMessage(content=self.MARINE_PROMPT)
        agent_kwargs = {
            "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
            "system_message": system_message,
        }

        tools = [
            Tool(
                name="Search",
                func=self.web_scraper.search,
                description="Useful for marine research and gathering scientific data"
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
        Execute the marine research tool
        """
        logging.info(f"Starting marine research for query: {kwargs['query']}")
        
        try:
            # Get research results
            research_output = self.execute_research(kwargs["query"])
            
            # Create a single ContentItem for the research results
            content = [
                ContentItem(
                    url="",
                    title="Marine Research Results",
                    snippet="Marine science research findings",
                    content=research_output,
                    source="Marine Research Agent"
                )
            ]
            
            return ResearchToolOutput(
                content=content,
                summary=research_output
            )
            
        except Exception as e:
            logging.error(f"Error in marine research: {str(e)}")
            return ResearchToolOutput(
                content=[],
                summary=f"Error performing research: {str(e)}"
            )

    def _arun(self, *args, **kwargs):
        raise NotImplementedError("Async version not implemented")