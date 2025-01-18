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

import json
import logging

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

    def invoke(self, context: Optional[ResearchContext] = None, **kwargs) -> str:
        """
        Execute the research process from query to final report.
        
        Args:
            context: Optional research context for messaging
            **kwargs: Must include 'query' key with research question
            
        Returns:
            str: Final research report
            
        Raises:
            ValueError: If required arguments are missing
            Exception: For other errors during research process
        """
        if 'query' not in kwargs:
            raise ValueError("Research query is required")

        if context is None:
            context = ResearchContext()

        # Initialize trace
        trace = QueryTrace(kwargs["query"])
        trace.data["start_time"] = datetime.now().isoformat()
        
        try:
            # Generate research outline
            self._send_message(context, "Generating outline...")
            self.tracer.log_step(trace, "start_outline_generation")
            outline = self._generate_outline(kwargs["query"], trace)  # Pass trace object
            trace.data["outline"] = outline
            self._send_message(context, "Generating outline... done.", outline)

            # Convert outline to DAG
            self.tracer.log_step(trace, "start_dag_conversion")
            research_outline = self._convert_outline_to_dag(outline, trace)  # Pass trace object
            trace.data["dag"] = research_outline.to_dict()
            self._send_message(context, "Planning tasks... done.")

            # Execute research tasks
            self.tracer.log_step(trace, "start_task_execution")
            results = self._plan_and_execute(research_outline, context)
            trace.data["results"] = [r.to_dict() for r in results]

            # Generate final report
            self.tracer.log_step(trace, "start_report_generation")
            final_report = self._generate_final_report(results)
            trace.data["final_report"] = final_report
            self._send_message(context, "Generating final report...", final_report)

            # Include token usage statistics in trace
            if hasattr(trace, 'token_tracker'):
                token_stats = trace.token_tracker.get_usage_stats()
                trace.data["token_usage"] = token_stats
                logging.info(f"Total tokens used: {token_stats['total_usage']['total_tokens']}")

            # Save research results
            self._save_final_report(
                outline=outline,
                query=kwargs["query"],
                research_outline=research_outline,
                results=results,
                final_report=final_report
            )

            return final_report

        except Exception as e:
            error_msg = f"Error during research: {str(e)}"
            trace.data["error"] = error_msg
            logging.error(error_msg)
            self._send_message(context, error_msg)
            raise

        finally:
            # Complete trace
            trace.data["end_time"] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(trace.data["start_time"])
            end_time = datetime.fromisoformat(trace.data["end_time"])
            trace.data["duration"] = (end_time - start_time).total_seconds()
            self.tracer.save_trace(trace)

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