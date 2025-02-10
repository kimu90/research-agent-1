from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import os
import logging
import traceback

api_router = APIRouter()

@api_router.get("/datasets", response_model=List[str])
async def get_datasets():
   try:
       data_folder = "./data"
       full_path = os.path.abspath(data_folder)

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

       try:
           datasets = [
               f for f in os.listdir(full_path)
               if f.endswith('.csv')
           ]
       except Exception as list_error:
           raise HTTPException(
               status_code=500,
               detail=f"Could not list directory contents: {str(list_error)}"
           )

       if not datasets:
           raise HTTPException(
               status_code=404,
               detail="No CSV files found in data directory"
           )

       return datasets

   except HTTPException:
       raise

   except Exception as e:
       traceback.print_exc()
       raise HTTPException(
           status_code=500,
           detail=f"Unexpected error retrieving datasets: {str(e)}"
       )