from utils.model_wrapper import model_wrapper
from utils.json_model_wrapper import json_model_wrapper
from .db import ContentDB

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

class TaskResult(BaseModel):
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

    def add(self, type, text="", **kwargs):
        if type == "text":
            self.content.append(text)
        logging.info(text)

    def notify(self):
        # In a real implementation, this might send a notification
        pass

class ResearchTask:
    def __init__(
        self,
        id: str,
        research_topic: str,
        dependencies: List[str],
        context: ResearchContext,
    ):
        self.id = id
        self.research_topic = research_topic
        self.dependencies = dependencies
        self.context = context

    # Rest of the methods remain the same as in the original implementation
    # Just replace `self.eezo_context` with `self.context`
    # And replace `m = self.eezo_context.new_message()` with `m = self.context.new_message()`

    # The execute method and other methods would remain mostly unchanged
    def execute(
        self,
        db: ContentDB,
        state: Dict[str, TaskResult],
        tools: List[BaseTool],
    ) -> TaskResult:
        logging.info(f"Executing task {self.id}:")
        relevant_state = {dep: state[dep] for dep in self.dependencies}

        # Get the content used by previous tasks.
        content_ids = [item.content_used for item in relevant_state.values()]
        content_ids = [item for sublist in content_ids for item in sublist]

        m = self.context.new_message()
        m.add("text", text=f"**Researching {self.id}** - {self.research_topic}\n\n")
        m.notify()

        # Rest of the method remains the same...
        # You would implement the same logic as in the original method

        # This is a placeholder return - you'd replace with actual logic
        return TaskResult(
            result="",
            content_used=[],
            content_urls=[],
            research_topic=self.research_topic,
            id=self.id,
            error=""
        )

# Prompts can be defined as global variables or class attributes
select_content = Prompt("research-agent-select-content")
extract_notes = Prompt("research-agent-extract-notes-from-webpages")
assessing_information_sufficiency = Prompt("research-agent-assessing-information-sufficiency")