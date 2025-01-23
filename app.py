from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from tools import GeneralAgent
from research_components.research import run_tool
from research_components.database import get_db_connection
app = FastAPI()
class ResearchRequest(BaseModel):
    query: str
    tool_name: Optional[str] = "General Agent"
class ResearchResponse(BaseModel):
    summary: str
    content: list
    trace_data: dict
@app.post("/research/", response_model=ResearchResponse)
async def perform_research(request: ResearchRequest):
    try:
        # Initialize tool
        tool = GeneralAgent(include_summary=True)

        # Run research
        result, trace = run_tool(
            tool_name=request.tool_name, 
            query=request.query, 
            tool=tool
        )

        # Prepare response
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