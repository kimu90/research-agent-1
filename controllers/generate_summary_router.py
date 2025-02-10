from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from research_components.langfuse_runner import create_tool_runner
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI router and models
api_router = APIRouter()
tool_runner = create_tool_runner()

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
    logger.info(f"Received research request: {request}")
    try:
        print(f"Running tool: {request.tool_name} with query: {request.query}")
        result, trace = tool_runner.run_tool(
            tool_name=request.tool_name,
            query=request.query,
            prompt_name=request.prompt_name
        )
        
        if not result:
            logger.error("Research failed: No result returned")
            raise HTTPException(status_code=400, detail="Research failed")
        
        print(f"Research completed. Summary length: {len(result.summary)}")
        logger.info(f"Research completed successfully for query: {request.query}")
        
        cleaned_content = [
            {
                "title": clean_text(item.title),
                "url": item.url,
                "snippet": clean_text(item.snippet),
                "content": clean_text(item.content)
            }
            for item in result.content
        ]
        
        print(f"Cleaned content. Number of items: {len(cleaned_content)}")
        
        response = {
            "summary": clean_text(result.summary),
            "content": cleaned_content,
            "trace_data": trace,
            "prompt_used": request.prompt_name
        }
        logger.info(f"Returning response for query: {request.query}")
        return response
    except Exception as e:
        logger.exception(f"Error occurred during research: {str(e)}")
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))