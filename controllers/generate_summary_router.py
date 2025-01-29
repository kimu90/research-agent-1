from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from tools import GeneralAgent
from research_components.research import run_tool

api_router = APIRouter()

class ResearchRequest(BaseModel):
    query: str
    tool_name: Optional[str] = "General Agent"

class ResearchResponse(BaseModel):
    summary: str
    content: list
    trace_data: dict

@api_router.post("/", response_model=ResearchResponse)
async def generate_summary(request: ResearchRequest):
    try:
        # Initialize tool
        tool = GeneralAgent(include_summary=True)
        
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
                "trace_data": trace.data
            }
        else:
            raise HTTPException(status_code=400, detail="Research failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
