from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import os

api_router = APIRouter()
PROMPTS_DIR = "/app/prompts"

class Prompt(BaseModel):
    id: str
    content: str
    metadata: dict = {}

class PromptResponse(BaseModel):
    folders: List[str]
    prompts: List[Prompt]

@api_router.get("/prompts", response_model=PromptResponse)
async def get_prompts():
    try:
        structure = {
            "folders": set(),
            "prompts": []
        }

        for root, dirs, files in os.walk(PROMPTS_DIR):
            rel_path = os.path.relpath(root, PROMPTS_DIR)
            if rel_path != '.' and not any(folder in rel_path for folder in structure["folders"]):
                structure["folders"].add(rel_path.split('/')[0])

            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(rel_path, file)
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                    structure["prompts"].append(
                        Prompt(
                            id=file_path,
                            content=content,
                            metadata={"folder": rel_path.split('/')[0]}
                        )
                    )

        structure["folders"] = sorted(list(structure["folders"]))
        return structure

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "species-analysis-api",
        "version": "1.0.0",
        "available_analysis_types": ["basic", "trends", "geographic"]
    }