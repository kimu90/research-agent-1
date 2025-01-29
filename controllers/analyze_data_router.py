from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from tools import AnalysisAgent
from research_components.research import run_tool

api_router = APIRouter()

class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoints"""
    query: str
    tool_name: Optional[str] = "Analysis Agent"
    dataset_path: Optional[str] = None

class AnalyzeResponse(BaseModel):
    """Response model for analysis results"""
    analysis: str
    trace_data: Dict[str, Any]
    metrics: Dict[str, Any]
    usage: Dict[str, Any]

@api_router.post("/", response_model=AnalyzeResponse)
async def generate_analysis(request: AnalyzeRequest):
    """Generate analysis based on the provided query"""
    try:
        # Initialize analysis agent
        tool = AnalysisAgent()
        
        # Run analysis
        result, trace = run_tool(
            tool_name=request.tool_name,
            query=request.query,
            tool=tool
        )
        
        if result:
            return {
                "analysis": result.analysis,
                "trace_data": trace.data,
                "metrics": result.metrics.dict(),
                "usage": result.usage
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Analysis failed to produce valid results"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis generation failed: {str(e)}"
        )

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "analysis-api"}