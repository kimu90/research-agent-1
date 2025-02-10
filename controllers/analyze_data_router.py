from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from research_components.langfuse_runner import create_tool_runner
import re

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
async def generate_analysis(request: AnalyzeRequest):
    try:
        result, trace = await tool_runner.run_tool(
            tool_name=request.tool_name,
            query=request.query,
            dataset=request.dataset,
            analysis_type=request.analysis_type,
            prompt_name=request.prompt_name
        )
        
        if not result:
            raise HTTPException(
                status_code=400,
                detail="Analysis failed to produce valid results"
            )

        def clean_text(text):
            text = re.sub(r"\*\*|##|\*", "", text)
            text = re.sub(r"\n\s*\n", "\n", text)
            return text.strip()

        cleaned_analysis = clean_text(result.analysis)

        return {
            "analysis": cleaned_analysis,
            "trace_data": trace,
            "usage": result.usage,
            "prompt_used": request.prompt_name
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis generation failed: {str(e)}")

@api_router.get("/datasets")
async def get_datasets():
    try:
        tool = AnalysisAgent(data_folder="./data")
        datasets = tool.get_available_datasets()
        return datasets
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving datasets: {str(e)}"
        )

@api_router.get("/dataset/{dataset_name}/info")
async def get_dataset_info(dataset_name: str):
    try:
        tool = AnalysisAgent(data_folder="./data")
        available_datasets = tool.get_available_datasets()
        
        if dataset_name not in available_datasets:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        df = tool.load_and_validate_data(dataset_name)
        
        return {
            "filename": dataset_name,
            "columns": list(df.columns),
            "rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset info: {str(e)}"
        )