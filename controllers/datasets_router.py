import os
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console and file handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("datasets_router.log")

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

# Dataset Info Model
class DatasetInfo(BaseModel):
    filename: str
    columns: List[str]
    rows: int
    missing_values: Dict[str, int]
    dtypes: Dict[str, str]

@api_router.get("/datasets", response_model=List[str])
async def get_datasets():
    """
    Retrieve a list of available CSV datasets.
    
    Returns:
    - List of dataset filenames
    """
    try:
        data_folder = os.getenv("DATASETS_FOLDER", "./data")
        full_path = os.path.abspath(data_folder)
        
        logger.info(f"Attempting to list datasets in: {full_path}")
        
        # Validate data folder exists and is a directory
        if not os.path.exists(full_path):
            logger.error(f"Data directory not found: {full_path}")
            raise HTTPException(
                status_code=404,
                detail=f"Data directory not found: {full_path}"
            )
        
        if not os.path.isdir(full_path):
            logger.error(f"Path is not a directory: {full_path}")
            raise HTTPException(
                status_code=500,
                detail=f"Path is not a directory: {full_path}"
            )
        
        # List CSV files
        try:
            datasets = [
                f for f in os.listdir(full_path)
                if f.endswith('.csv')
            ]
        except Exception as list_error:
            logger.error(f"Could not list directory contents: {str(list_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not list directory contents: {str(list_error)}"
            )
        
        # Check if any datasets found
        if not datasets:
            logger.warning("No CSV files found in data directory")
            raise HTTPException(
                status_code=404,
                detail="No CSV files found in data directory"
            )
        
        logger.info(f"Found {len(datasets)} datasets")
        return datasets
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving datasets: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error retrieving datasets: {str(e)}"
        )

@api_router.get("/dataset/{dataset_name}/info", response_model=DatasetInfo)
async def get_dataset_info(dataset_name: str):
    """
    Retrieve detailed information about a specific dataset.
    
    Args:
    - dataset_name: Name of the dataset to retrieve info for
    
    Returns:
    - Detailed dataset information
    """
    try:
        import pandas as pd
        
        data_folder = os.getenv("DATASETS_FOLDER", "./data")
        full_path = os.path.join(os.path.abspath(data_folder), dataset_name)
        
        logger.info(f"Attempting to load dataset: {dataset_name}")
        
        # Validate dataset exists
        if not os.path.exists(full_path):
            logger.error(f"Dataset not found: {dataset_name}")
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{dataset_name}' not found"
            )
        
        # Load dataset
        try:
            df = pd.read_csv(full_path)
        except Exception as load_error:
            logger.error(f"Could not load dataset: {str(load_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not load dataset: {str(load_error)}"
            )
        
        # Prepare dataset info
        dataset_info = {
            "filename": dataset_name,
            "columns": list(df.columns),
            "rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()}
        }
        
        logger.info("Dataset info retrieved successfully")
        return dataset_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve dataset info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving dataset info: {str(e)}"
        )