from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import re

from tools import GeneralAgent, PromptLoader
from research_components.research import run_tool

api_router = APIRouter()

# Models
class ResearchRequest(BaseModel):
    """Request model for research endpoint."""
    query: str
    tool_name: Optional[str] = "General Agent"
    prompt_name: Optional[str] = "research.txt"

class ResearchResponse(BaseModel):
    """Response model for research endpoint."""
    summary: str
    content: List[Dict[str, Any]]
    trace_data: Dict[str, Any]
    prompt_used: str

class PromptListResponse(BaseModel):
    """Response model for listing available prompts."""
    prompts: List[str]

# Helper functions
def clean_text(text: str) -> str:
    """Clean response text by removing markdown and excess whitespace."""
    text = re.sub(r"\\*|##|\*", "", text)  # Remove markdown formatting
    text = re.sub(r"\n\s\n", "\n", text)   # Remove excessive newlines
    return text.strip()

# Routes
@api_router.post("/", response_model=ResearchResponse)
async def generate_summary(request: ResearchRequest):
    """Generate research summary based on query."""
    try:
        tool = GeneralAgent(
            include_summary=True,
            prompt_name=request.prompt_name
        )

        result, trace = run_tool(
            tool_name=request.tool_name,
            query=request.query,
            tool=tool
        )

        if not result:
            raise HTTPException(status_code=400, detail="Research failed")

        cleaned_content = [
            {
                "title": clean_text(item.title),
                "url": item.url,
                "snippet": clean_text(item.snippet),
                "content": clean_text(item.content)
            }
            for item in result.content
        ]

        return {
            "summary": clean_text(result.summary),
            "content": cleaned_content,
            "trace_data": trace.data,
            "prompt_used": request.prompt_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/prompts", response_model=PromptListResponse)
async def get_available_prompts():
    """Retrieve list of available research prompts."""
    try:
        prompts = PromptLoader.list_available_prompts()
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing prompts: {str(e)}"
        )