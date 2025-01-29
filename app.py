# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from controllers.generate_summary_router import api_router as summary_router
from controllers.analyze_data_router import api_router as analysis_router
from controllers.report_router import api_router as report_router

# Create FastAPI app instance
app = FastAPI(
    title="Species Research Platform",
    version="0.0.1",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
        "url": "https://your-url.com"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the API routers
app.include_router(summary_router, prefix="/api/generate-summary", tags=["Summary"])
app.include_router(analysis_router, prefix="/api/analyze-data", tags=["Analysis"])
app.include_router(report_router, prefix="/api/generate-report", tags=["Report"])

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html") as f:
        return f.read()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)