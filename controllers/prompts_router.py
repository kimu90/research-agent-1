from fastapi import APIRouter, HTTPException
import os
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from typing import List
from pydantic import BaseModel

api_router = APIRouter()

class Prompt(BaseModel):
    id: str
    content: str
    metadata: dict = {}




@api_router.get("/prompt-folders", response_model=List[str])
async def get_prompt_folders():
    """
    Retrieve list of available prompt folders
    """
    prompts_dir = "/app/prompts"  # Base prompts directory
    if not os.path.exists(prompts_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Prompts directory not found: {prompts_dir}"
        )
    
    # List only directories in the prompts folder
    folders = [
        f for f in os.listdir(prompts_dir) 
        if os.path.isdir(os.path.join(prompts_dir, f))
    ]
    
    return folders

@api_router.get("/prompts", response_model=List[Prompt])
async def get_prompts(folder: str):
    """
    Retrieve list of available prompt templates for a specific folder
    """
    prompts_dir = f"/app/prompts/{folder}"
    if not os.path.exists(prompts_dir):
        raise HTTPException(
            status_code=404,
            detail=f"Prompt folder not found: {prompts_dir}"
        )
    
    # List all .txt files in the specified folder
    prompt_files = [
        f for f in os.listdir(prompts_dir)
        if f.endswith('.txt')
    ]
    
    prompts = [
        Prompt(id=filename, content=f"Prompt: {filename}", metadata={})
        for filename in prompt_files
    ]
    
    return prompts