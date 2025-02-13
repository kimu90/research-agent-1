import os
import logging
import re
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd

from research_components.langfuse_runner import create_tool_runner
from tools.research.analysis_agent import AnalysisAgent

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("analysis_router.log")

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

# Configuration
DATA_FOLDER = os.getenv("DATASETS_FOLDER", "./data")

class CustomJSONResponse(JSONResponse):
    def __init__(self, content: Any, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self.headers["Content-Type"] = "application/json; charset=utf-8"
        self.headers["Cache-Control"] = "no-store"
        self.headers["X-Content-Type-Options"] = "nosniff"

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    tool_name: Optional[str] = "Analysis Agent"
    dataset: str
    prompt_name: Optional[str] = "research.txt"
    analysis_type: str = Field(default="general")
    run_evaluation: bool = Field(default=True)

class AnalyzeResponse(BaseModel):
    analysis: str
    trace_data: Dict[str, Any]
    usage: Dict[str, Any]
    prompt_used: str
    evaluation_results: Optional[Dict[str, Any]] = None

class DatasetInfoResponse(BaseModel):
    filename: str
    columns: List[str]
    rows: int
    missing_values: Dict[str, int]
    dtypes: Dict[str, str]

def clean_text(text: str) -> str:
    """Clean and format text by removing unnecessary formatting."""
    try:
        text = re.sub(r"\*\*|##|\*", "", text)
        text = re.sub(r"\n\s*\n", "\n", text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

@api_router.post("/", response_model=AnalyzeResponse)
async def generate_analysis(request: AnalyzeRequest):
    """Generate a data analysis based on the given request."""
    logger.info("=== Starting new analysis request ===")
    logger.debug(f"Request parameters: {request.dict()}")
    
    try:
        result, trace_data = tool_runner.run_tool(
            tool_name=request.tool_name,
            query=request.query,
            dataset=request.dataset,
            analysis_type=request.analysis_type,
            prompt_name=request.prompt_name,
            run_evaluation=request.run_evaluation
        )
        
        if not result:
            logger.error("Analysis produced no valid results")
            raise HTTPException(
                status_code=400,
                detail="Analysis failed to produce valid results"
            )
        
        cleaned_analysis = clean_text(result.analysis)
        
        response_data = {
            "analysis": cleaned_analysis,
            "trace_data": trace_data,
            "usage": result.usage,
            "prompt_used": request.prompt_name,
            "evaluation_results": trace_data.get("evaluation_results")
        }
        
        logger.info("Analysis request completed successfully")
        logger.debug(f"Response usage data: {result.usage}")
        
        return CustomJSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis generation failed: {str(e)}")

@api_router.get("/datasets", response_model=List[str])
async def get_datasets():
    """Retrieve a list of available datasets."""
    logger.info("Fetching available datasets")
    
    try:
        tool = AnalysisAgent(data_folder=DATA_FOLDER)
        datasets = tool.get_available_datasets()
        logger.info(f"Successfully retrieved {len(datasets)} datasets")
        logger.debug(f"Available datasets: {datasets}")
        
        return CustomJSONResponse(content=datasets)
        
    except Exception as e:
        logger.error(f"Failed to retrieve datasets: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving datasets: {str(e)}"
        )

@api_router.get("/dataset/{dataset_name}/info", response_model=DatasetInfoResponse)
async def get_dataset_info(dataset_name: str):
    """Retrieve detailed information about a specific dataset."""
    logger.info(f"Fetching info for dataset: {dataset_name}")
    
    try:
        tool = AnalysisAgent(data_folder=DATA_FOLDER)
        
        if dataset_name not in tool.get_available_datasets():
            logger.warning(f"Dataset not found: {dataset_name}")
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        df = tool.load_and_validate_data(dataset_name)
        
        dataset_info = {
            "filename": dataset_name,
            "columns": list(df.columns),
            "rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
        }
        
        logger.info("Dataset info retrieved successfully")
        logger.debug(f"Dataset info: {dataset_info}")
        
        return CustomJSONResponse(content=dataset_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve dataset info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset info: {str(e)}"
        )

@api_router.get("/health")
async def health_check():
    """Perform a health check on the analysis service."""
    try:
        tool = AnalysisAgent(data_folder=DATA_FOLDER)
        datasets = tool.get_available_datasets()
        
        response_data = {
            "status": "healthy",
            "service": "species-analysis-api",
            "version": "1.0.0",
            "data_folder": DATA_FOLDER,
            "total_datasets": len(datasets),
            "available_analysis_types": ["general", "detailed", "comparative"]
        }
        
        return CustomJSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return CustomJSONResponse(
            content={
                "status": "degraded",
                "error": str(e),
                "service": "species-analysis-api",
                "version": "1.0.0"
            }
        )