# research_agent/research_agent.py

from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from .research_task_scheduler import TaskScheduler
from .research_task import ResearchTask, TaskResult
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from datetime import datetime
from typing import List, Dict, Any, Optional
from prompts import Prompt
from .tracers import CustomTracer, QueryTrace

# Correct import for model schemas
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem

# Separate import for GoogleSerperAPIWrapper
from langchain_community.utilities import GoogleSerperAPIWrapper

import json
import logging
import os
class Question(BaseModel):
    """Represents an individual research question."""
    id: str = Field(
        ...,
        description="Unique identifier for each question, reflecting position and dependency structure",
    )
    text: str = Field(..., description="The text of the question")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of IDs that this question depends on. Empty array indicates no dependencies",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "dependencies": self.dependencies,
        }

class ResearchOutline(BaseModel):
    """Represents a research outline consisting of questions."""
    questions: List[Question] = Field(
        ...,
        description="List of main questions and subquestions",
        min_items=1,
    )

    def to_dict(self) -> Dict[str, Any]:
        return {"questions": [question.to_dict() for question in self.questions]}

class ResearchContext:
    """Handles research context and messaging."""
    def __init__(self):
        self.messages = []

    def new_message(self):
        return ResearchMessage(self)

class ResearchMessage:
    """Handles individual research messages."""
    def __init__(self, context):
        self.context = context
        self.content = []
        self.message_id = None

    def add(self, type, text="", **kwargs):
        if type == "text":
            self.content.append(text)
        logging.info(text)
        return self

    def replace(self, message_id, type, text=""):
        # Placeholder for message replacement
        return self

    def notify(self):
        # Placeholder for notification handling
        pass

class ResearchAgent:
    """
    Orchestrates the research process using tools and prompts.
    Handles outline generation, task planning, execution, and report generation.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize ResearchAgent with required tools.
        
        Args:
            tools: List of LangChain tools for research tasks
        """
        self.tools = tools
        self.tracer = CustomTracer()

    def invoke(
        self, 
        input: Dict[str, str], 
        custom_prompt: Optional[Prompt] = None
    ) -> ResearchToolOutput:
        """
        Execute the news search tool with comprehensive error handling and token tracking
        
        Args:
            input: Dictionary containing the search query
            custom_prompt: Optional custom prompt to override default
        
        Returns:
            ResearchToolOutput containing search results and usage information
        """
        # Logging and initial setup
        logging.info(f"Starting news search for query: {input.get('query', 'No query')}")
        
        # Create a default output structure
        default_output = ResearchToolOutput(
            content=[],
            summary="No results found.",
            usage={
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0,
                'model': 'unknown'
            }
        )
        
        try:
            # Validate input
            if not input or 'query' not in input:
                logging.warning("No query provided")
                return default_output
            
            # Use custom prompt if provided, otherwise use default
            current_prompt = custom_prompt or self.custom_prompt
            
            # Initialize search tools
            google_serper = GoogleSerperAPIWrapper(
                type="news", 
                k=10, 
                serper_api_key=SERPER_API_KEY
            )
            
            # Perform initial news search
            try:
                response = google_serper.results(query=input["query"])
                news_results = response.get("news", [])
            except Exception as search_error:
                logging.error(f"Search API error: {str(search_error)}")
                return default_output
            
            # Ensure all required fields exist in news results
            for news in news_results:
                for field in ["snippet", "date", "source", "title", "link", "imageUrl"]:
                    if field not in news:
                        news[field] = ""
            
            # Select relevant results
            try:
                selected_results = self.decide_what_to_use(
                    content=news_results, 
                    research_topic=input["query"]
                )
            except Exception as selection_error:
                logging.error(f"Result selection error: {str(selection_error)}")
                selected_results = news_results
            
            # Scrape webpage content
            webpage_urls = [result["link"] for result in selected_results]
            try:
                webpages = self.scrape_pages(webpage_urls)
            except Exception as scrape_error:
                logging.error(f"Web scraping error: {str(scrape_error)}")
                webpages = []
            
            # Prepare content items
            content = []
            for news in selected_results:
                # Find corresponding webpage content
                webpage = next(
                    (
                        doc.page_content
                        for doc in webpages
                        if doc.metadata.get("source") == news["link"]
                    ),
                    "",
                )
                
                # Create content item
                title = news.get("title", "") + " - " + news.get("date", "")
                content.append(
                    ContentItem(
                        url=news["link"],
                        title=title,
                        snippet=news.get("text", ""),
                        content=webpage,
                    )
                )
            
            # Generate summary if enabled
            summary = ""
            if self.include_summary:
                try:
                    # Prepare summary prompt
                    formatted_content = "\n\n".join([f"### {item}" for item in content])
                    system_prompt = SUMMARIZE_RESULTS_PROMPT.compile(
                        search_results_str=formatted_content, 
                        user_prompt=input["query"]
                    )
                    
                    # Generate summary
                    summary = model_wrapper(
                        system_prompt=system_prompt,
                        prompt=SUMMARIZE_RESULTS_PROMPT,
                        user_prompt=f"Summarize and group the search results based on this: '{input['query']}'. Include links, dates, and snippets from the search results.",
                        model="llama3-70b-8192",
                        host="groq",
                        temperature=0.7,
                        token_tracker=self.token_tracker
                    )
                    logging.info("Generated summary of news articles")
                except Exception as summary_error:
                    logging.error(f"Summary generation error: {str(summary_error)}")
                    summary = "Unable to generate summary."
            
            # Prepare final output with token usage
            output = ResearchToolOutput(
                content=content, 
                summary=summary
            )
            
            # Attach token usage information
            try:
                output.usage = {
                    'prompt_tokens': self.token_tracker._total_prompt_tokens,
                    'completion_tokens': self.token_tracker._total_completion_tokens,
                    'total_tokens': self.token_tracker._total_tokens,
                    'model': 'llama3-70b-8192'
                }
            except Exception as usage_error:
                logging.warning(f"Could not attach token usage: {str(usage_error)}")
                output.usage = {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0,
                    'model': 'unknown'
                }
            
            return output
        
        except Exception as e:
            # Catch-all error handling
            logging.error(f"Unexpected error in invoke method: {str(e)}")
            
            # Return a default output with error information
            default_output.summary = f"An error occurred: {str(e)}"
            return default_output

    def _generate_outline(self, query: str, trace: Optional[QueryTrace] = None) -> str:
        """
        Generate research outline from query.
        
        Args:
            query: Research question/topic
            trace: Optional QueryTrace object for token tracking
            
        Returns:
            str: Generated research outline
        """
        generate_outline = Prompt(
            id="research-agent-generate-outline",
            content="""Generate a detailed research outline for: {{user_prompt}}

            Create a hierarchical structure that:
            1. Breaks down the main topic
            2. Identifies key research areas
            3. Lists specific questions to investigate
            4. Shows relationships between topics

            Format as a clear, numbered outline."""
        )
        system_prompt = generate_outline.compile(user_prompt=query)
        return model_wrapper(
            system_prompt=system_prompt,
            prompt=generate_outline,
            user_prompt=query,
            trace=trace  # Pass trace object for token tracking
        )

    def _convert_outline_to_dag(self, outline: str) -> ResearchOutline:
        """
        Convert text outline to structured DAG.
        
        Args:
            outline: Text outline to convert
            
        Returns:
            ResearchOutline: Structured outline with dependencies
        """
        outline_to_dag = Prompt(
            id="research-agent-outline-to-dag-conversion",
            content="""Convert this outline into a structured DAG:

            Outline:
            {{outline}}

            Generate a structured format that:
            1. Assigns unique IDs to each section
            2. Identifies dependencies between sections
            3. Maintains the hierarchical relationships

            Schema format: {{output_schema}}"""
        )
        system_prompt = outline_to_dag.compile(output_schema="", outline=outline)
        return json_model_wrapper(
            system_prompt=system_prompt,
            user_prompt="Parse the outline into the json schema",
            prompt=outline_to_dag,
            base_model=ResearchOutline
        )

    def _plan_and_execute(
        self, 
        research_outline: ResearchOutline, 
        context: ResearchContext
    ) -> List[TaskResult]:
        """
        Plan and execute research tasks based on outline.
        
        Args:
            research_outline: Structured research outline
            context: Research context for messaging
            
        Returns:
            List[TaskResult]: Results of all research tasks
        """
        task_list = [
            ResearchTask(
                id=question.id,
                research_topic=question.text,
                dependencies=question.dependencies
            )
            for question in research_outline.questions
        ]

        scheduler = TaskScheduler(task_list, self.tools)
        scheduler.execute()
        return scheduler.get_results()

    def _generate_final_report(self, results: List[TaskResult]) -> str:
        """
        Generate final research report from task results.
        
        Args:
            results: List of task results to summarize
            
        Returns:
            str: Formatted final report
        """
        research_section_summarizer = Prompt(
            id="research-section-summarizer",
            content="""Summarize this research section:

            Topic: {{research_topic}}
            Notes: {{section_notes}}

            Provide:
            1. Key findings
            2. Supporting evidence
            3. Important conclusions"""
        )
        final_report = ""

        for task_result in results:
            if task_result.error:
                continue

            if not task_result.content_used:
                final_report += (
                    f"{task_result.id} {task_result.research_topic}\n"
                    "No content found.\n\n"
                )
                continue

            system_prompt = research_section_summarizer.compile(
                research_topic=task_result.research_topic,
                section_notes=task_result.result
            )

            section_summary = model_wrapper(
                system_prompt=system_prompt,
                prompt=research_section_summarizer,
                user_prompt="Generate a summary of the section",
                model="llama3-70b-8192",
                host="groq"
            )

            final_report += (
                f"**{task_result.id} {task_result.research_topic}**\n"
                f"{section_summary}\n\n"
            )

        return final_report

    def _save_final_report(
        self,
        outline: str,
        query: str,
        research_outline: ResearchOutline,
        results: List[TaskResult],
        final_report: str
    ) -> None:
        """
        Save all research artifacts to file.
        
        Args:
            outline: Original text outline
            query: Original research query
            research_outline: Structured research outline
            results: Task results
            final_report: Generated report
        """
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        with open(f"research_{timestamp}.json", "w") as f:
            json.dump(
                {
                    "outline": outline,
                    "query": query,
                    "dag": json.loads(research_outline.model_dump_json()),
                    "results": [result.to_dict() for result in results],
                    "final_report": final_report
                },
                f,
                indent=4
            )

    def _send_message(
        self, 
        context: ResearchContext, 
        text: str, 
        content: str = ""
    ) -> None:
        """
        Send message to research context.
        
        Args:
            context: Research context
            text: Message text
            content: Optional additional content
        """
        if context:
            message = context.new_message()
            msg = message.add("text", text=text)
            if content:
                message.replace(msg.message_id, "text", text=content)
            message.notify()