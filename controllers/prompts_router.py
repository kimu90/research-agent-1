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


@api_router.get("/prompts", response_model=List[Prompt])
async def get_prompts():
    """
    Retrieve list of available prompt templates
    """
    try:
        prompts_dir = "/app/prompts"  # Adjust this path as needed
        if not os.path.exists(prompts_dir):
            raise HTTPException(
                status_code=404,
                detail=f"Prompts directory not found: {prompts_dir}"
            )

        # List all .txt files in the prompts directory
        prompt_files = [
            f for f in os.listdir(prompts_dir)
            if f.endswith('.txt')
        ]

        if not prompt_files:
            raise HTTPException(
                status_code=404,
                detail="No prompt templates found"
            )

        # Convert the list of filenames to a list of Prompt objects
        prompts = [
            Prompt(id=filename, content=f"Prompt: {filename}", metadata={})
            for filename in prompt_files
        ]

        return prompts

    except Exception as e:
        logging.error(f"Error retrieving prompts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving prompts: {str(e)}"
        )
