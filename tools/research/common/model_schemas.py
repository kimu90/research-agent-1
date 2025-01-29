# tools/research/common/model_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class ContentItem(BaseModel):
    """
    Represents a single content item from research.
    """
    url: str
    title: str
    snippet: str = ""
    content: str = ""
    source: Optional[str] = ""
    id: Optional[str] = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

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
            "metadata": self.metadata
        }

class ResearchToolOutput(BaseModel):
    """
    Represents the output of a research tool with comprehensive tracking.
    """
    content: List[ContentItem] = Field(default_factory=list)
    summary: str = ""
    usage: Dict[str, Any] = Field(default_factory=lambda: {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0,
        'model': 'unknown',
        'cost': 0.0
    })
    raw_data: Optional[List[Dict[str, Any]]] = None
    
    def total_content_items(self) -> int:
        """
        Returns the total number of content items.
        """
        return len(self.content)

    def get_unique_sources(self) -> List[str]:
        """
        Returns a list of unique sources.
        """
        return list(set(item.source for item in self.content if item.source))

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the output to a dictionary representation.
        """
        return {
            "content": [item.to_dict() for item in self.content],
            "summary": self.summary,
            "usage": self.usage,
            "total_items": self.total_content_items(),
            "sources": self.get_unique_sources()
        }

class ScrapeWebsiteInput(BaseModel):
    """
    Input schema for website scraping with enhanced options.
    """
    objective: str = Field(
        default="Extract relevant information", 
        description="The specific goal of the web scraping"
    )
    url: str
    max_content_length: Optional[int] = Field(
        default=10000, 
        description="Maximum length of content to extract"
    )
    extract_metadata: bool = Field(
        default=True, 
        description="Whether to extract additional metadata"
    )
    keywords: Optional[List[str]] = Field(
        default=None, 
        description="Keywords to focus extraction on"
    )

class ResearchQuery(BaseModel):
    """
    Represents a structured research query with additional metadata.
    """
    query: str
    categories: Optional[List[str]] = None
    complexity: Optional[str] = Field(
        default=None, 
        description="Complexity level of the research query"
    )
    language: Optional[str] = Field(
        default="en", 
        description="Language of the research query"
    )
    
    def normalize(self) -> str:
        """
        Normalizes the query by removing extra whitespace and converting to lowercase.
        """
        return ' '.join(self.query.split()).lower()

class AnalysisMetrics(BaseModel):
    """
    Metrics specific to analysis outputs.
    """
    numerical_accuracy: float = Field(default=0.0)
    query_understanding: float = Field(default=0.0)
    data_validation: float = Field(default=0.0)
    reasoning_transparency: float = Field(default=0.0)
    calculations: Dict = Field(default_factory=dict)
    used_correct_columns: bool = Field(default=False)
    used_correct_analysis: bool = Field(default=False)
    used_correct_grouping: bool = Field(default=False)
    handled_missing_data: bool = Field(default=False)
    handled_outliers: bool = Field(default=False)
    handled_datatypes: bool = Field(default=False)
    handled_format_issues: bool = Field(default=False)
    explained_steps: bool = Field(default=False)
    stated_assumptions: bool = Field(default=False)
    mentioned_limitations: bool = Field(default=False)
    clear_methodology: bool = Field(default=False)

class AnalysisResult(BaseModel):
    """
    Output specific to analysis operations.
    """
    analysis: str
    metrics: AnalysisMetrics
    usage: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis,
            "metrics": self.metrics.dict(),
            "usage": self.usage
        }