# tools/research/common/model_schemas.py

from pydantic import BaseModel
from typing import List, Optional

class ContentItem(BaseModel):
    """
    Represents a single content item.
    """
    url: str
    title: str
    snippet: str
    content: str
    source: Optional[str] = ""
    id: Optional[str] = ""

    def __str__(self):
        return f"{self.title}\n{self.url}\n{self.snippet}"

    def to_dict(self):
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "source": self.source,
            "id": self.id,
        }

class ResearchToolOutput(BaseModel):
    """
    Represents the output of a research tool.
    """
    content: List[ContentItem]
    summary: str

class ScrapeWebsiteInput(BaseModel):
    """
    Input schema for website scraping
    """
    objective: str = "Extract relevant information"
    url: str