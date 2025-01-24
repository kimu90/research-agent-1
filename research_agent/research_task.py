from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from utils.db import ContentDB
from tools.research.common.model_schemas import ContentItem
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from typing import List, Dict, Any, Optional, Type
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from prompts import Prompt
import openai
import logging
import uuid
import json
import os

oc = openai.Client()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskResult(BaseModel):
    """Result of a research task"""
    result: Optional[str] = ""
    content_used: Optional[List[str]] = []
    content_urls: Optional[List[str]] = []
    research_topic: Optional[str] = ""
    id: str
    error: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "content_used": self.content_used,
            "content_urls": self.content_urls,
            "research_topic": self.research_topic,
            "id": self.id,
            "error": self.error,
        }

class ResearchContext:
    """Context for managing research messages and progress"""
    def __init__(self):
        self.messages = []
        self.task_progress = {}

    def new_message(self):
        return ResearchMessage(self)

    def update_progress(self, task_id: str, progress: float, status: str):
        """Update task progress"""
        self.task_progress[task_id] = {
            "progress": progress,
            "status": status
        }

class ResearchMessage:
    """Handles research progress messages"""
    def __init__(self, context):
        self.context = context
        self.content = []
        self.message_id = str(uuid.uuid4())

    def add(self, type, text="", **kwargs):
        if type == "text":
            self.content.append(text)
            logger.info(text)
        return self

    def notify(self):
        """Send notification about task progress"""
        if self.content:
            logger.info(f"Message {self.message_id}: {self.content[-1]}")

# Initialize prompts with proper attributes and templates
SELECT_CONTENT_PROMPT = Prompt(
    id="research-agent-select-content",
    content="""Analyze the following content and select the most relevant pieces for researching: {{topic}}
    
    Consider:
    1. Relevance to the topic
    2. Credibility of sources
    3. Recency of information
    4. Depth of coverage
    
    Content items available:
    {{content_items}}
    
    Return selected content IDs as a comma-separated list.""",
    metadata={"type": "content_selection"}
)

EXTRACT_NOTES_PROMPT = Prompt(
    id="research-agent-extract-notes-from-webpages",
    content="""Extract key information from the following content about: {{topic}}
    
    Focus on:
    1. Main findings and conclusions
    2. Supporting evidence
    3. Methodologies used
    4. Expert opinions
    5. Current trends
    
    Content to analyze:
    {{content}}
    
    Provide a detailed summary with specific examples and citations.""",
    metadata={"type": "note_extraction"}
)

ASSESS_INFO_PROMPT = Prompt(
    id="research-agent-assessing-information-sufficiency",
    content="""Evaluate if the following information sufficiently addresses: {{topic}}

    Current information:
    {{current_info}}
    
    Assessment criteria:
    1. Comprehensiveness
    2. Depth of analysis
    3. Source diversity
    4. Evidence quality
    5. Gap identification
    
    Return YES if sufficient, NO with explanation if not.""",
    metadata={"type": "assessment"}
)

SYNTHESIZE_PROMPT = Prompt(
    id="research-agent-synthesize-information",
    content="""Synthesize the following research notes into a coherent analysis about: {{topic}}

    Research notes:
    {{notes}}
    
    Previous findings:
    {{previous_findings}}
    
    Create a comprehensive synthesis that:
    1. Identifies key themes
    2. Highlights connections
    3. Notes contradictions
    4. Suggests implications
    5. Acknowledges limitations""",
    metadata={"type": "synthesis"}
)

class ResearchTask:
    """Handles individual research tasks with comprehensive content processing"""
    
    def __init__(
        self,
        id: str,
        research_topic: str,
        dependencies: List[str],
        context: Optional[ResearchContext] = None,
        min_content_items: int = 3,
        max_content_items: int = 10
    ):
        self.id = id
        self.research_topic = research_topic
        self.dependencies = dependencies
        self.context = context or ResearchContext()
        self.min_content_items = min_content_items
        self.max_content_items = max_content_items
        self.llm = ChatOpenAI(model="gpt-4")

    def execute(
        self,
        db: ContentDB,
        state: Dict[str, TaskResult],
        tools: List[BaseTool],
    ) -> TaskResult:
        """
        Execute the research task with comprehensive processing.
        
        Args:
            db: Content database for research
            state: Current state of all task results
            tools: Available research tools
            
        Returns:
            TaskResult: Complete research task result
        """
        try:
            # Initialize progress
            self.context.update_progress(self.id, 0.0, "Started")
            message = self.context.new_message()
            message.add("text", text=f"**Researching {self.id}** - {self.research_topic}\n\n")
            message.notify()

            # Get dependency results
            relevant_state = {dep: state[dep] for dep in self.dependencies}
            previous_findings = self._compile_previous_findings(relevant_state)
            
            # Get content excluding previously used
            used_content_ids = []
            for result in relevant_state.values():
                used_content_ids.extend(result.content_used)

            # Select relevant content
            self.context.update_progress(self.id, 0.2, "Selecting content")
            selected_content = self._select_content(db, used_content_ids)
            
            if not selected_content:
                return self._create_error_result("No relevant content found")

            # Process URLs if present
            self.context.update_progress(self.id, 0.4, "Processing web content")
            web_content = self._process_web_content(selected_content)
            
            # Extract information
            self.context.update_progress(self.id, 0.6, "Extracting information")
            extracted_notes = self._extract_notes(selected_content, web_content)
            
            # Assess information sufficiency
            self.context.update_progress(self.id, 0.8, "Assessing information")
            if not self._assess_information(extracted_notes):
                # Try to get more content if needed
                additional_content = self._select_content(db, used_content_ids + [c.id for c in selected_content])
                if additional_content:
                    additional_notes = self._extract_notes(additional_content, self._process_web_content(additional_content))
                    extracted_notes = self._synthesize_information(extracted_notes, additional_notes, previous_findings)
                    selected_content.extend(additional_content)

            # Finalize result
            self.context.update_progress(self.id, 1.0, "Completed")
            content_urls = [item.url for item in selected_content if hasattr(item, 'url')]
            content_ids = [str(item.id) for item in selected_content if hasattr(item, 'id')]

            return TaskResult(
                result=extracted_notes,
                content_used=content_ids,
                content_urls=content_urls,
                research_topic=self.research_topic,
                id=self.id,
                error=""
            )

        except Exception as e:
            error_msg = f"Error in task {self.id}: {str(e)}"
            logger.error(error_msg)
            return self._create_error_result(error_msg)

    def _compile_previous_findings(self, state: Dict[str, TaskResult]) -> str:
        """Compile findings from dependent tasks"""
        findings = []
        for task_id, result in state.items():
            if result.result:
                findings.append(f"From {task_id}: {result.result}")
        return "\n\n".join(findings)

    def _select_content(self, db: ContentDB, exclude_ids: List[str]) -> List[ContentItem]:
        """
        Select relevant content items from the database
        """
        try:
            # Get available content
            available_content = db.get_content(exclude_ids=exclude_ids)
            if not available_content:
                return []

            # Prepare content for selection
            content_items = "\n".join([
                f"ID: {item.id}\nTitle: {item.title}\nSummary: {item.summary}\n"
                for item in available_content
            ])

            # Get model selection
            system_prompt = SELECT_CONTENT_PROMPT.compile(
                topic=self.research_topic,
                content_items=content_items
            )
            
            response = model_wrapper(
                system_prompt=system_prompt,
                prompt=SELECT_CONTENT_PROMPT,
                user_prompt="Select the most relevant content IDs"
            )

            # Process selected IDs
            selected_ids = [id.strip() for id in response.split(',')]
            selected_content = [
                item for item in available_content 
                if str(item.id) in selected_ids
            ]

            # Ensure minimum content
            if len(selected_content) < self.min_content_items:
                additional_needed = self.min_content_items - len(selected_content)
                remaining_content = [
                    item for item in available_content 
                    if item not in selected_content
                ][:additional_needed]
                selected_content.extend(remaining_content)

            return selected_content[:self.max_content_items]

        except Exception as e:
            logger.error(f"Error selecting content: {str(e)}")
            return []

    def _process_web_content(self, content_items: List[ContentItem]) -> Dict[str, str]:
        """
        Process web content from URLs in content items
        """
        web_content = {}
        for item in content_items:
            if hasattr(item, 'url') and item.url:
                try:
                    loader = WebBaseLoader(item.url)
                    docs = loader.load()
                    web_content[str(item.id)] = "\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    logger.warning(f"Error loading URL {item.url}: {str(e)}")
        return web_content

    def _extract_notes(self, content_items: List[ContentItem], web_content: Dict[str, str]) -> str:
        """
        Extract and combine notes from all content sources
        """
        try:
            # Prepare content for extraction
            combined_content = []
            for item in content_items:
                item_content = f"Title: {item.title}\n"
                if hasattr(item, 'content'):
                    item_content += f"Content: {item.content}\n"
                if str(item.id) in web_content:
                    item_content += f"Web Content: {web_content[str(item.id)]}\n"
                combined_content.append(item_content)

            system_prompt = EXTRACT_NOTES_PROMPT.compile(
                topic=self.research_topic,
                content="\n---\n".join(combined_content)
            )

            return model_wrapper(
                system_prompt=system_prompt,
                prompt=EXTRACT_NOTES_PROMPT,
                user_prompt="Extract key information from the content"
            )

        except Exception as e:
            logger.error(f"Error extracting notes: {str(e)}")
            return ""

    def _assess_information(self, current_info: str) -> bool:
        """
        Assess if the gathered information is sufficient
        """
        try:
            system_prompt = ASSESS_INFO_PROMPT.compile(
                topic=self.research_topic,
                current_info=current_info
            )

            response = model_wrapper(
                system_prompt=system_prompt,
                prompt=ASSESS_INFO_PROMPT,
                user_prompt="Assess information sufficiency"
            )

            return response.strip().upper().startswith("YES")

        except Exception as e:
            logger.error(f"Error assessing information: {str(e)}")
            return True  # Default to True to prevent infinite loops

    def _synthesize_information(self, original_notes: str, additional_notes: str, previous_findings: str) -> str:
        """
        Synthesize multiple sources of information
        """
        try:
            system_prompt = SYNTHESIZE_PROMPT.compile(
                topic=self.research_topic,
                notes=f"{original_notes}\n\nAdditional notes:\n{additional_notes}",
                previous_findings=previous_findings
            )

            return model_wrapper(
                system_prompt=system_prompt,
                prompt=SYNTHESIZE_PROMPT,
                user_prompt="Synthesize all information"
            )

        except Exception as e:
            logger.error(f"Error synthesizing information: {str(e)}")
            return original_notes  # Fall back to original notes on error

    def _create_error_result(self, error_msg: str) -> TaskResult:
        """
        Create a TaskResult for error conditions
        """
        return TaskResult(
            result="",
            content_used=[],
            content_urls=[],
            research_topic=self.research_topic,
            id=self.id,
            error=error_msg
        )