from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict
import os

# Import your existing routers
from controllers.generate_summary_router import api_router as summary_router
from controllers.analyze_data_router import api_router as analysis_router
from controllers.report_router import api_router as report_router
from controllers.datasets_router import api_router as datasets_router
from controllers.prompts_router import api_router as prompts_router

# Pydantic models for type checking
class PromptFolderResponse(BaseModel):
    folders: List[str]

class PromptResponse(BaseModel):
    prompts: List[Dict[str, str]]

# Create FastAPI app instance
app = FastAPI(
    title="Species Research Platform",
    version="0.1.0",
    description="Advanced platform for species research and prompt-based exploration",
    contact={
        "name": "Research Team",
        "email": "research@speciesexplorer.org",
        "url": "https://speciesexplorer.org"
    }
)

# Configure CORS with more specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",  # Local development
        "https://speciesexplorer.org",  # Production domain
        "http://127.0.0.1:8000",  # Local development
        "*"  # Adjust in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/prompts", StaticFiles(directory="prompts"), name="prompts")
app.mount("/data", StaticFiles(directory="data"), name="data")

# Include the API routers
app.include_router(summary_router, prefix="/api/generate-summary", tags=["Summary"])
app.include_router(analysis_router, prefix="/api/analyze-data", tags=["Analysis"])
app.include_router(report_router, prefix="/api/generate-report", tags=["Report"])
app.include_router(datasets_router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(prompts_router, prefix="/api/prompts", tags=["Prompts"])

# Endpoint to list prompt folders
@app.get("/api/prompt-folders", response_model=PromptFolderResponse)
async def get_prompt_folders():
    """
    Retrieve list of available prompt folders.
    """
    try:
        prompts_dir = "prompts"  # Base prompts directory
        if not os.path.exists(prompts_dir):
            return {"folders": []}
        
        # List only directories in the prompts folder
        folders = [
            f for f in os.listdir(prompts_dir) 
            if os.path.isdir(os.path.join(prompts_dir, f))
        ]
        
        return {"folders": folders}
    
    except Exception as e:
        print(f"Error retrieving prompt folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to list prompts in a specific folder
@app.get("/api/prompts", response_model=PromptResponse)
async def get_prompts_in_folder(folder: str = None):
    """
    Retrieve list of available prompt templates for a specific folder.
    """
    try:
        # If no folder specified, return an error
        if not folder:
            raise HTTPException(status_code=400, detail="Folder parameter is required")

        prompts_dir = os.path.join("prompts", folder)
        if not os.path.exists(prompts_dir):
            raise HTTPException(status_code=404, detail=f"Folder {folder} not found")
        
        # List all .txt files in the specified folder
        prompt_files = [
            {
                "id": filename, 
                "content": f"Prompt: {filename}", 
                "filename": filename
            } for filename in os.listdir(prompts_dir)
            if filename.endswith('.txt')
        ]
        
        return {"prompts": prompt_files}
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to list datasets
@app.get("/api/datasets")
async def get_datasets():
    """
    Retrieve list of available datasets.
    """
    try:
        data_dir = "data"
        if not os.path.exists(data_dir):
            raise HTTPException(status_code=404, detail="Datasets directory not found")
        
        # List CSV files in the data directory
        datasets = [
            f for f in os.listdir(data_dir)
            if f.endswith('.csv')
        ]
        
        return datasets
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrieving datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve the main application page
@app.get("/", response_class=HTMLResponse)
async def read_index():
    """
    Serve the main application HTML page.
    """
    try:
        with open("templates/index.html", "r") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Application Not Found</h1>", status_code=404)

# Health check endpoint
@app.get("/health")
def health_check():
    """
    Basic health check endpoint.
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "components": {
            "summary": "operational",
            "analysis": "operational",
            "reports": "operational",
            "prompts": "operational"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)