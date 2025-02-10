from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from langfuse import Langfuse
import os
import os
from typing import Optional
from dotenv import load_dotenv

from controllers.generate_summary_router import api_router as summary_router
from controllers.analyze_data_router import api_router as analysis_router
from controllers.report_router import api_router as report_router
from controllers.datasets_router import api_router as datasets_router
from controllers.prompts_router import api_router as prompts_router

# Load environment variables
load_dotenv()

# Langfuse configuration
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY")
)

# Lifespan management for Langfuse
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events can be added here if needed
    yield
    # Ensure all events are flushed to Langfuse on shutdown
    langfuse.flush()

# App Configuration
app = FastAPI(
    title="Species Research Platform",
    version="0.0.1",
    lifespan=lifespan,
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
        "url": "https://your-url.com"
    }
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/prompts", StaticFiles(directory="prompts"), name="prompts")

# API Routes
app.include_router(summary_router, prefix="/api/generate-summary", tags=["Summary"])
app.include_router(analysis_router, prefix="/api", tags=["Analysis"])
app.include_router(report_router, prefix="/api/generate-report", tags=["Report"])
app.include_router(datasets_router, prefix="/api", tags=["Datasets"])
app.include_router(prompts_router, prefix="/api", tags=["Prompts"])

# Base Routes
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "species-research-platform",
        "version": "0.0.1",
        "tracing": "Langfuse enabled"
    }

# Run Configuration
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)