import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from research_components.langfuse_runner import create_tool_runner
from tools.research.analysis_agent import AnalysisAgent
import re

# Set up logging
logger = logging.getLogger("research_api")
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("research_api.log")

# Create formatters and add it to handlers
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
console_formatter = logging.Formatter(log_format)
file_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

api_router = APIRouter()
tool_runner = create_tool_runner()

class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=200)
    tool_name: Optional[str] = "Analysis Agent"
    dataset: str
    prompt_name: Optional[str] = "research.txt"
    analysis_type: str = Field(default="general")

class AnalyzeResponse(BaseModel):
    analysis: str
    trace_data: Dict[str, Any]
    usage: Dict[str, Any]
    prompt_used: str

@api_router.post("/analyze-data", response_model=AnalyzeResponse)
def generate_analysis(request: AnalyzeRequest):
    logger.info("=== Starting new analysis request ===")
    logger.debug(f"Request parameters: {request.dict()}")
    
    try:
        logger.info(f"Executing analysis with tool: {request.tool_name}")
        logger.debug(f"Analysis parameters - Dataset: {request.dataset}, Type: {request.analysis_type}")
        
        result, trace = tool_runner.run_tool(
            tool_name=request.tool_name,
            query=request.query,
            dataset=request.dataset,
            analysis_type=request.analysis_type,
            prompt_name=request.prompt_name
        )
        
        if not result:
            logger.error("Analysis produced no valid results")
            raise HTTPException(
                status_code=400,
                detail="Analysis failed to produce valid results"
            )
        
        logger.debug("Analysis completed successfully, cleaning output text")
        
        def clean_text(text):
            logger.debug("Cleaning analysis text")
            text = re.sub(r"\*\*|##|\*", "", text)
            text = re.sub(r"\n\s*\n", "\n", text)
            return text.strip()
        
        cleaned_analysis = clean_text(result.analysis)
        logger.debug(f"Text cleaned, final length: {len(cleaned_analysis)}")
        
        response_data = {
            "analysis": cleaned_analysis,
            "trace_data": trace,
            "usage": result.usage,
            "prompt_used": request.prompt_name
        }
        
        logger.info("Analysis request completed successfully")
        logger.debug(f"Response usage data: {result.usage}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Analysis generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis generation failed: {str(e)}")

@api_router.get("/datasets")
def get_datasets():
    logger.info("Fetching available datasets")
    
    try:
        tool = AnalysisAgent(data_folder="./data")
        logger.debug("AnalysisAgent initialized")
        
        datasets = tool.get_available_datasets()
        logger.info(f"Successfully retrieved {len(datasets)} datasets")
        logger.debug(f"Available datasets: {datasets}")
        
        return datasets
        
    except Exception as e:
        logger.error(f"Failed to retrieve datasets: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving datasets: {str(e)}"
        )

@api_router.get("/dataset/{dataset_name}/info")
def get_dataset_info(dataset_name: str):
    logger.info(f"Fetching info for dataset: {dataset_name}")
    
    try:
        tool = AnalysisAgent(data_folder="./data")
        logger.debug("AnalysisAgent initialized")
        
        available_datasets = tool.get_available_datasets()
        logger.debug(f"Available datasets: {available_datasets}")
        
        if dataset_name not in available_datasets:
            logger.warning(f"Dataset not found: {dataset_name}")
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        logger.info(f"Loading dataset: {dataset_name}")
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
        
        return dataset_info
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Failed to retrieve dataset info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset info: {str(e)}"
        )