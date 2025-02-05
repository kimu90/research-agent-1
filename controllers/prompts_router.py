from fastapi import APIRouter, HTTPException
import os
import logging
from typing import Dict, List, Any
from pydantic import BaseModel

logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

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
           "folders": set(),  # Use set for unique folders
           "prompts": []
       }
       
       for root, dirs, files in os.walk(PROMPTS_DIR):
           rel_path = os.path.relpath(root, PROMPTS_DIR)
           if rel_path != '.' and not any(folder in rel_path for folder in structure["folders"]):
               structure["folders"].add(rel_path.split('/')[0])  # Only add top-level folder
               
           for file in files:
               if file.endswith('.txt'):
                   file_path = os.path.join(rel_path, file)
                   with open(os.path.join(root, file), 'r') as f:
                       content = f.read()
                   structure["prompts"].append(Prompt(
                       id=file_path,
                       content=content,
                       metadata={"folder": rel_path.split('/')[0]}  # Use top-level folder
                   ))
                   
       structure["folders"] = sorted(list(structure["folders"]))  # Convert set to sorted list
       return structure
       
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))