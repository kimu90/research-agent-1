from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from research_components.langfuse_runner import create_tool_runner
import re

api_router = APIRouter()
tool_runner = create_tool_runner()

# Existing models can remain the same
class ResearchRequest(BaseModel):
    query: str
    tool_name: Optional[str] = "General Agent"
    prompt_name: Optional[str] = "research.txt"

class ResearchResponse(BaseModel):
    summary: str
    content: List[Dict[str, Any]]
    trace_data: Dict[str, Any]
    prompt_used: str

def clean_text(text: str) -> str:
    text = re.sub(r"\*\*|##|\*", "", text)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()

@api_router.post("/", response_model=ResearchResponse)
async def generate_summary(request: ResearchRequest):
    try:
        result, trace = await tool_runner.run_tool(
            tool_name=request.tool_name,
            query=request.query,
            prompt_name=request.prompt_name
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
            "trace_data": trace,
            "prompt_used": request.prompt_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))