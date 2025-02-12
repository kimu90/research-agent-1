import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console and file handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("prompts_router.log")

# Create formatters
log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
console_formatter = logging.Formatter(log_format)
file_formatter = logging.Formatter(log_format)
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Router setup
api_router = APIRouter()

# Configuration
PROMPTS_DIR = os.getenv("PROMPTS_FOLDER", "/app/prompts")

# Pydantic Models
class Prompt(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = {}

class PromptResponse(BaseModel):
    folders: List[str]
    prompts: List[Prompt]

@api_router.get("/prompts", response_model=PromptResponse)
async def get_prompts():
    """
    Retrieve all available prompts organized by folders.
    
    Returns:
    - Dictionary with folders and their corresponding prompts
    """
    try:
        logger.info(f"Retrieving prompts from directory: {PROMPTS_DIR}")
        
        # Validate prompts directory exists
        if not os.path.exists(PROMPTS_DIR):
            logger.error(f"Prompts directory not found: {PROMPTS_DIR}")
            raise HTTPException(
                status_code=404,
                detail=f"Prompts directory not found: {PROMPTS_DIR}"
            )
        
        # Initialize structure
        structure = {
            "folders": set(),
            "prompts": []
        }
        
        # Walk through directory
        for root, _, files in os.walk(PROMPTS_DIR):
            # Get relative path
            rel_path = os.path.relpath(root, PROMPTS_DIR)
            
            # Add folders (excluding root)
            if rel_path != '.':
                top_level_folder = rel_path.split('/')[0]
                if top_level_folder not in structure["folders"]:
                    structure["folders"].add(top_level_folder)
            
            # Process text files
            for file in files:
                if file.endswith('.txt'):
                    try:
                        file_path = os.path.join(rel_path, file)
                        full_file_path = os.path.join(root, file)
                        
                        # Read file content
                        with open(full_file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Create prompt entry
                        structure["prompts"].append(
                            Prompt(
                                id=file_path,
                                content=content,
                                metadata={
                                    "folder": top_level_folder if rel_path != '.' else 'root',
                                    "filename": file
                                }
                            )
                        )
                    except Exception as file_error:
                        logger.warning(f"Could not process file {file}: {str(file_error)}")
        
        # Sort folders
        structure["folders"] = sorted(list(structure["folders"]))
        
        logger.info(f"Retrieved {len(structure['prompts'])} prompts from {len(structure['folders'])} folders")
        return structure
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving prompts: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving prompts: {str(e)}"
        )

@api_router.get("/health")
async def health_check():
    """
    Perform a health check on the prompts service.
    
    Returns:
    - Health status information
    """
    try:
        # Count total prompts across all subdirectories
        prompt_count = 0
        folders = []
        
        for root, _, files in os.walk(PROMPTS_DIR):
            # Count text files
            txt_files = [f for f in files if f.endswith('.txt')]
            prompt_count += len(txt_files)
            
            # Track unique folders
            rel_path = os.path.relpath(root, PROMPTS_DIR)
            if rel_path != '.' and rel_path not in folders:
                folders.append(rel_path)
        
        return {
            "status": "healthy",
            "service": "species-prompts-api",
            "version": "1.0.0",
            "prompts_directory": PROMPTS_DIR,
            "total_prompts": prompt_count,
            "total_folders": len(folders),
            "available_categories": ["research", "analysis", "summary"]
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "service": "species-prompts-api",
            "version": "1.0.0"
        }