from fastapi import APIRouter, HTTPException
from typing import List
import os
import traceback
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

api_router = APIRouter()

@api_router.get("/datasets", response_model=List[str])
async def get_datasets():
    """
    Retrieve list of available CSV datasets
    """
    try:
        # Explicitly log and check the data folder
        data_folder = "./data"
        full_path = os.path.abspath(data_folder)
        
        # Log detailed path information
        print(f"Looking for datasets in: {full_path}")
        print(f"Absolute path exists: {os.path.exists(full_path)}")
        print(f"Is directory: {os.path.isdir(full_path)}")
        
        # Verify the directory exists and is accessible
        if not os.path.exists(full_path):
            raise HTTPException(
                status_code=404,
                detail=f"Data directory not found: {full_path}"
            )
        
        if not os.path.isdir(full_path):
            raise HTTPException(
                status_code=500,
                detail=f"Path is not a directory: {full_path}"
            )
        
        # Attempt to list files
        try:
            all_files = os.listdir(full_path)
            print(f"All files in directory: {all_files}")
        except Exception as list_error:
            print(f"Error listing directory: {list_error}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not list directory contents: {str(list_error)}"
            )
        
        # Get CSV files
        datasets = [
            f for f in all_files 
            if f.endswith('.csv')
        ]
        
        print(f"Found CSV datasets: {datasets}")
        
        if not datasets:
            raise HTTPException(
                status_code=404,
                detail="No CSV files found in data directory"
            )
        
        return datasets
    
    except HTTPException:
        # Re-raise HTTP exceptions directly
        raise
    
    except Exception as e:
        # Log the full traceback
        print("Unexpected error in get_datasets:")
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving datasets: {str(e)}"
        )