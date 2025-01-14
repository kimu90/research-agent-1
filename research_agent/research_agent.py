from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from .research_task_scheduler import TaskScheduler
from .research_task import ResearchTask, TaskResult
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from datetime import datetime
from typing import List, Dict, Any, Optional
from prompts import Prompt
from .tracers import CustomTracer, QueryTrace  # Add this import

import json
import logging

class Question(BaseModel):
    """
    Represents an individual research question.
    """
    id: str = Field(
        ...,
        description="A unique identifier for each question, reflecting its position and dependency structure.",
    )
    text: str = Field(..., description="The text of the question.")
    dependencies: List[str] = Field(
        default_factory=list,
        description="A list of IDs that this question depends on. An empty array indicates no dependencies.",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "dependencies": self.dependencies,
        }

class ResearchOutline(BaseModel):
    """
    Represents a research outline consisting of a list of questions.
    """
    questions: List[Question] = Field(
        ...,
        description="A list of main questions and subquestions.",
        min_items=1,
    )

    def to_dict(self) -> Dict["str", Any]:
        return {"questions": [question.to_dict() for question in self.questions]}

class ResearchContext:
    """
    A simple replacement for Eezo's Context class
    """
    def __init__(self):
        self.messages = []

    def new_message(self):
        return ResearchMessage(self)

class ResearchMessage:
    """
    A simple replacement for Eezo's Message class
    """
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
        # In a real implementation, this might send a notification
        pass

class ResearchAgent:
    """
    Orchestrates the research process using various tools and prompts.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initializes the ResearchAgent with a list of tools.
        """
        self.tools = tools
        self.tracer = CustomTracer()  # Add tracer initialization

    
    def invoke(self, context: Optional[ResearchContext] = None, **kwargs) -> str:  # Return final_report
        if context is None:
            context = ResearchContext()

        # Initialize trace for this query
        trace = QueryTrace(kwargs["query"])
        
        try:
            self._send_message(context, "Generating outline...")
            
            # Track outline generation
            self.tracer.log_step(trace, "start_outline_generation")
            outline: str = self._generate_outline(kwargs["query"])
            trace.data["outline"] = outline
            self._send_message(context, "Generating outline... done.", outline)
            
            # Track DAG conversion
            self.tracer.log_step(trace, "start_dag_conversion")
            research_outline: ResearchOutline = self._convert_outline_to_dag(outline)
            trace.data["dag"] = research_outline.to_dict()
            self._send_message(context, "Planning tasks... done.")
            
            # Track task execution
            self.tracer.log_step(trace, "start_task_execution")
            results: List[TaskResult] = self._plan_and_execute(
                research_outline, context
            )
            trace.data["results"] = [r.to_dict() for r in results]
            
            # Track final report generation
            self.tracer.log_step(trace, "start_report_generation")
            final_report = self._generate_final_report(results)
            trace.data["final_report"] = final_report
            self._send_message(context, "Generating final report...", final_report)
            
            self._save_final_report(
                outline, kwargs["query"], research_outline, results, final_report
            )
            
            return final_report  # Return the final report

        except Exception as e:
            error_msg = f"Error during research: {str(e)}"
            trace.data["error"] = error_msg
            logging.error(error_msg)
            self._send_message(context, error_msg)
            raise

        finally:
            # Complete the trace
            trace.data["end_time"] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(trace.data["start_time"])
            end_time = datetime.fromisoformat(trace.data["end_time"])
            trace.data["duration"] = (end_time - start_time).total_seconds()
            
            # Save the trace
            self.tracer.save_trace(trace)

    # Rest of the methods remain the same
    def _generate_outline(self, query: str) -> str:
        generate_outline = Prompt("research-agent-generate-outline")
        system_prompt = generate_outline.compile(user_prompt=query)
        return model_wrapper(
            system_prompt=system_prompt,
            prompt=generate_outline,
            user_prompt=query,
        )

    def _convert_outline_to_dag(self, outline: str) -> ResearchOutline:
        outline_to_dag = Prompt("research-agent-outline-to-dag-conversion")
        system_prompt = outline_to_dag.compile(output_schema="", outline=outline)
        return json_model_wrapper(
            system_prompt=system_prompt,
            user_prompt="Parse the outline into the json schema",
            prompt=outline_to_dag,
            base_model=ResearchOutline,
        )

    def _plan_and_execute(
        self, research_outline: ResearchOutline, context: ResearchContext
    ) -> List[TaskResult]:
        task_list = []
        for question in research_outline.questions:
            task_list.append(
                ResearchTask(
                    id=question.id,
                    research_topic=question.text,
                    dependencies=question.dependencies,
                )
            )

        scheduler = TaskScheduler(task_list, self.tools)
        scheduler.execute()
        return scheduler.get_results()

    def _generate_final_report(self, results: List[TaskResult]) -> str:
        research_section_summarizer = Prompt("research-section-summarizer")
        final_report = ""
        for task_result in results:
            if task_result.error != "":
                continue
            if len(task_result.content_used) == 0:
                final_report += f"{task_result.id} {task_result.research_topic}\nNo content found.\n\n"
            else:
                system_prompt = research_section_summarizer.compile(
                    research_topic=task_result.research_topic,
                    section_notes=task_result.result,
                )
                section_summary = model_wrapper(
                    system_prompt=system_prompt,
                    prompt=research_section_summarizer,
                    user_prompt="Generate a summary of the section",
                    model="llama3-70b-8192",
                    host="groq",
                )
                final_report += f"**{task_result.id} {task_result.research_topic}**\n{section_summary}\n\n"
        return final_report

    def _save_final_report(
        self,
        outline: str,
        query: str,
        research_outline: ResearchOutline,
        results: List[TaskResult],
        final_report: str,
    ) -> None:
        human_readable_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(f"research_{human_readable_timestamp}.json", "w") as f:
            json.dump(
                {
                    "outline": outline,
                    "query": query,
                    "dag": json.loads(research_outline.model_dump_json()),
                    "results": [result.to_dict() for result in results],
                    "final_report": final_report,
                },
                f,
                indent=4,
            )

    def _send_message(
        self, context: ResearchContext, text: str, content: str = ""
    ) -> None:
        """
        Sends a message to the context, optionally including additional content.
        """
        if context:
            m = context.new_message()
            c = m.add("text", text=text)
            if content:
                m.replace(c.message_id, "text", text=content)
            m.notify()