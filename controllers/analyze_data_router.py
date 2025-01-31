from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from tools import AnalysisAgent, PromptLoader
from research_components.research import run_tool

api_router = APIRouter()

class AnalyzeRequest(BaseModel):
    """Enhanced request model for analysis endpoints"""
    query: str = Field(..., min_length=2, max_length=200)
    tool_name: Optional[str] = "Analysis Agent"
    dataset: str
    prompt_name: Optional[str] = "research.txt"
    analysis_type: str = Field(
        default="general",
        description="Type of analysis to perform (e.g., 'general', 'statistical', 'correlation')"
    )
class AnalyzeResponse(BaseModel):
    """Comprehensive response model for analysis results"""
    analysis: str
    trace_data: Dict[str, Any]
    metrics: Dict[str, Any]
    usage: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    prompt_used: str

class PromptListResponse(BaseModel):
    """Response model for listing available prompts."""
    prompts: List[str]


@api_router.post("/analyze-data", response_model=AnalyzeResponse)
async def generate_analysis(request: AnalyzeRequest):
    """
    Generate analysis based on the provided query and dataset
    
    Supports different analysis types and provides comprehensive results
    """
    try:
        # Initialize analysis agent
        tool = AnalysisAgent(
            data_folder="./data",
            prompt_name=request.prompt_name
        )

        
        # Validate dataset availability
        available_datasets = tool.get_available_datasets()
        if request.dataset not in available_datasets:
            raise HTTPException(
                status_code=400,
                detail=f"Dataset not found. Available datasets: {available_datasets}"
            )
        
        # Run analysis with additional parameters
        result, trace = run_tool(
            tool_name=request.tool_name,
            query=request.query,
            dataset=request.dataset,
            analysis_type=request.analysis_type,
            tool=tool
        )
        
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Analysis failed to produce valid results"
            )
        
        return {
            "analysis": result.analysis,
            "trace_data": trace.data,
            "metrics": result.metrics.dict(),
            "usage": result.usage,
            "prompt_used": request.prompt_name,
            "metadata": {
                "dataset": request.dataset,
                "analysis_type": request.analysis_type,
                "timestamp": trace.timestamp
            }
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis generation failed: {str(e)}"
        )


@api_router.get("/health")
async def health_check():
    """
    Simple health check endpoint
    
    Returns service status and basic information
    """
    return {
        "status": "healthy",
        "service": "species-analysis-api",
        "version": "1.0.0",
        "available_analysis_types": ["basic", "trends", "geographic"]
    }

@api_router.get("/dataset/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    """
    Retrieve detailed information about a specific dataset
    
    Provides comprehensive dataset metadata
    """
    try:
        tool = AnalysisAgent(data_folder="./data")
        
        # Verify dataset exists
        available_datasets = tool.get_available_datasets()
        if dataset_name not in available_datasets:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found. Available: {available_datasets}"
            )
        
        # Load and analyze dataset
        df = tool.load_and_validate_data(dataset_name)
        
        return {
            "filename": dataset_name,
            "columns": list(df.columns),
            "rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset info: {str(e)}"
        )