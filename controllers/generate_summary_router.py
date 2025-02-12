import logging
import re
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from research_components.langfuse_runner import create_tool_runner

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console and file handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("research_router.log")

# Create formatters
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
console_formatter = logging.Formatter(log_format)
file_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Router and tool runner setup
api_router = APIRouter()
tool_runner = create_tool_runner()

# Request and Response Models
class ResearchRequest(BaseModel):
    query: str
    tool_name: Optional[str] = "General Agent"
    prompt_name: Optional[str] = "research.txt"

class ResearchResponse(BaseModel):
    summary: str
    content: List[Dict[str, Any]]
    trace_data: Dict[str, Any]
    prompt_used: str
    usage: Dict[str, Any]

def clean_text(text: str) -> str:
    """
    Clean and format text by removing unnecessary formatting.
    
    Args:
    - text: Input text to be cleaned
    
    Returns:
    - Cleaned text
    """
    try:
        # Remove markdown-like formatting
        text = re.sub(r"\*\*|##|\*", "", text)
        # Remove excessive newlines
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text  # Return original text if cleaning fails

@api_router.post("/", response_model=ResearchResponse)
async def generate_summary(request: ResearchRequest):
    """
    Generate a research summary based on the given query.
    
    Args:
    - request: Research request with query and optional parameters
    
    Returns:
    - Detailed research response
    """
    logger.info(f"Received research request: {request}")
    
    try:
        # Run the research tool
        logger.debug(f"Running tool: {request.tool_name} with query: {request.query}")
        result, trace = tool_runner.run_tool(
            tool_name=request.tool_name,
            query=request.query,
            prompt_name=request.prompt_name
        )
        
        # Validate result
        if not result:
            logger.error("Research failed: No result returned")
            raise HTTPException(status_code=400, detail="Research failed to produce results")
        
        # Clean and process content
        cleaned_content = [
            {
                "title": clean_text(item.title),
                "url": item.url,
                "snippet": clean_text(item.snippet),
                "content": clean_text(item.content)
            }
            for item in result.content
        ]
        
        # Prepare response
        response = {
            "summary": clean_text(result.summary),
            "content": cleaned_content,
            "trace_data": trace,
            "prompt_used": request.prompt_name,
            "usage": result.usage if hasattr(result, 'usage') else {}
        }
        
        logger.info(f"Research completed successfully for query: {request.query}")
        logger.debug(f"Response summary length: {len(response['summary'])}")
        logger.debug(f"Number of content items: {len(cleaned_content)}")
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error occurred during research: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Research generation failed: {str(e)}")