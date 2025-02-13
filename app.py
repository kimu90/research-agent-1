from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging

from langfuse import Langfuse
from init_langfuse import CreateNewEntities

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Async initialization function for Langfuse
async def initialize_langfuse():
    try:
        initializer = CreateNewEntities()
        await initializer.main()
        logger.info("Langfuse initialization completed successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")

# Lifespan management for Langfuse
@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_langfuse()
    
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY")
    )
    
    yield
    
    langfuse.flush()

# App Configuration
app = FastAPI(
    title="Species Research Platform",
    version="0.0.1",
    lifespan=lifespan,
    contact={
        "name": os.getenv("CONTACT_NAME", "Your Name"),
        "email": os.getenv("CONTACT_EMAIL", "your.email@example.com"),
        "url": os.getenv("CONTACT_URL", "https://your-url.com")
    }
)

# Import and Include Routers
from controllers.generate_summary_router import api_router as summary_router
from controllers.analyze_data_router import api_router as analysis_router
from controllers.report_router import api_router as report_router
from controllers.datasets_router import api_router as datasets_router
from controllers.prompts_router import api_router as prompts_router

# API Routes
app.include_router(summary_router, prefix="/api/generate-summary", tags=["Summary"])
app.include_router(analysis_router, prefix="/api/analyze-data", tags=["Analysis"])
app.include_router(report_router, prefix="/api/generate-report", tags=["Report"])
app.include_router(datasets_router, prefix="/api", tags=["Datasets"])
app.include_router(prompts_router, prefix="/api", tags=["Prompts"])

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'"
    return response

# Static Files - mount after API routes
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/prompts", StaticFiles(directory="prompts"), name="prompts")

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