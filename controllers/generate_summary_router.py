from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from tools import GeneralAgent, PromptLoader
from research_components.research import run_tool

api_router = APIRouter()

class ResearchRequest(BaseModel):
    """Request model for research endpoint."""
    query: str
    tool_name: Optional[str] = "General Agent"
    prompt_name: Optional[str] = "research.txt"

class ResearchResponse(BaseModel):
    """Response model for research endpoint."""
    summary: str
    content: List[dict]
    trace_data: dict
    prompt_used: str

class PromptListResponse(BaseModel):
    """Response model for listing available prompts."""
    prompts: List[str]
@api_router.post("/", response_model=ResearchResponse)
async def generate_summary(request: ResearchRequest):
    try:
        tool = GeneralAgent(
            include_summary=True,
            prompt_name=request.prompt_name
        )
        
        # Run research
        result, trace = run_tool(
            tool_name=request.tool_name,
            query=request.query,
            tool=tool
        )
        
        if result:
            return {
                "summary": result.summary,
                "content": [
                    {
                        "title": item.title,
                        "url": item.url,
                        "snippet": item.snippet,
                        "content": item.content
                    } for item in result.content
                ],
                "trace_data": trace.data,
                "prompt_used": request.prompt_name
            }
        else:
            raise HTTPException(status_code=400, detail="Research failed")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/prompts", response_model=PromptListResponse)
async def get_available_prompts():
    """
    Retrieve list of available research prompts.
    """
    try:
        prompts = PromptLoader.list_available_prompts()
        return {"prompts": prompts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing prompts: {str(e)}")