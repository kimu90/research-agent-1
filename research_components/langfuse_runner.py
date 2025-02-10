import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from langfuse import Langfuse
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import re

from tools import GeneralAgent, AnalysisAgent

# Enhanced logging configuration
def setup_logging():
    """Configure logging with a custom format and multiple handlers"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # File handler - for persistent logging
    file_handler = logging.FileHandler('langfuse_runner.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler - for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Get root logger and configure
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

class LangfuseRunner:
    def __init__(
        self, 
        public_key: Optional[str] = None, 
        secret_key: Optional[str] = None, 
        host: Optional[str] = None
    ):
        logger.info("Initializing LangfuseRunner")
        
        # Use environment variables if not explicitly provided
        self.public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
        self.secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
        self.host = host or os.getenv('LANGFUSE_HOST')
        
        logger.debug(f"Configuration - Host: {self.host}, Public Key: {'*' * len(self.public_key) if self.public_key else 'None'}")

        try:
            self._validate_config()
        except ValueError as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            raise

        # Initialize Langfuse with configuration
        try:
            langfuse_config = {
                'public_key': self.public_key,
                'secret_key': self.secret_key
            }
            if self.host:
                langfuse_config['host'] = self.host
            
            self.langfuse = Langfuse(**langfuse_config)
            logger.info("Successfully initialized Langfuse client")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {str(e)}")
            raise
        
        # Use default or environment variable for model costs
        self.model_costs = {
            'default': float(os.getenv('DEFAULT_MODEL_COST', 0.001))
        }
        logger.debug(f"Model costs configured: {self.model_costs}")

    def _validate_config(self):
        """Validate the configuration settings"""
        logger.debug("Validating configuration")
        if not self.public_key or not self.secret_key:
            logger.error("Missing required environment variables")
            raise ValueError("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set in .env")

    def run_tool(
        self,
        tool_name: str,
        query: str,
        dataset: Optional[str] = None,
        analysis_type: Optional[str] = None,
        tool: Optional[Any] = None,
        prompt_name: str = "research.txt"
    ) -> Tuple[Any, Dict[str, Any]]:
        logger.info(f"Running tool: {tool_name}")
        logger.debug(f"Query: {query}")
        logger.debug(f"Dataset: {dataset}")
        logger.debug(f"Analysis type: {analysis_type}")
        logger.debug(f"Prompt name: {prompt_name}")

        trace = self.langfuse.trace(
            name=f"{tool_name.lower().replace(' ', '-')}-execution",
            metadata={
                "tool": tool_name,
                "query": query,
                "dataset": dataset,
                "analysis_type": analysis_type,
                "prompt_name": prompt_name,
                "timestamp": datetime.now().isoformat()
            }
        )
        logger.debug("Created Langfuse trace")

        try:
            start_time = datetime.now()
            generation = trace.generation(name=f"{tool_name.lower()}-generation")
            logger.debug("Started generation trace")
            
            if tool_name == "General Agent":
                logger.info("Executing General Agent")
                result = self._run_general_agent(
                    generation, query, tool, prompt_name
                )
            elif tool_name == "Analysis Agent":
                logger.info("Executing Analysis Agent")
                result = self._run_analysis_agent(
                    generation, query, dataset, tool, prompt_name
                )
            else:
                logger.error(f"Invalid tool name: {tool_name}")
                raise ValueError(f"Unknown tool: {tool_name}")

            if result:
                generation.end(success=True)
                logger.info("Tool execution completed successfully")
            
            trace_data = self._prepare_trace_data(
                start_time=start_time,
                success=bool(result),
                tool_name=tool_name,
                prompt_name=prompt_name
            )
            logger.debug(f"Trace data prepared: {trace_data}")
            return result, trace_data

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in run_tool: {error_msg}", exc_info=True)
            trace.error(error_msg)
            generation.end(success=False, error=error_msg)
            return None, {"error": error_msg}

        finally:
            logger.debug("Flushing Langfuse traces")
            self.langfuse.flush()

    def _run_general_agent(self, generation, query: str, tool: Optional[Any], prompt_name: str):
        logger.debug("Initializing General Agent")
        tool = tool or GeneralAgent(include_summary=True, prompt_name=prompt_name)
        
        try:
            result = tool.invoke(input={"query": query})
            
            if result:
                content_count = len(result.content) if result.content else 0
                logger.debug(f"General Agent result - Content count: {content_count}")
                generation.update(
                    metadata={
                        "content_count": content_count,
                        "has_summary": bool(result.summary)
                    }
                )
                logger.info("General Agent execution completed")
            else:
                logger.warning("General Agent returned no results")
                
            return result
        except Exception as e:
            logger.error(f"Error in General Agent execution: {str(e)}", exc_info=True)
            raise

    def _run_analysis_agent(self, generation, query: str, dataset: str, 
                            tool: Optional[Any], prompt_name: str):
        logger.debug("Initializing Analysis Agent")
        tool = tool or AnalysisAgent(data_folder="./data", prompt_name=prompt_name)
        
        try:
            result = tool.invoke_analysis(input={"query": query, "dataset": dataset})
            
            if result:
                analysis_length = len(result.analysis) if result.analysis else 0
                logger.debug(f"Analysis Agent result - Analysis length: {analysis_length}")
                generation.update(
                    metadata={
                        "dataset": dataset,
                        "analysis_length": analysis_length
                    }
                )
                logger.info("Analysis Agent execution completed")
            else:
                logger.warning("Analysis Agent returned no results")
                
            return result
        except Exception as e:
            logger.error(f"Error in Analysis Agent execution: {str(e)}", exc_info=True)
            raise

    def _prepare_trace_data(self, start_time: datetime, success: bool, 
                          tool_name: str, prompt_name: str) -> Dict[str, Any]:
        duration = (datetime.now() - start_time).total_seconds()
        
        trace_data = {
            "duration": duration,
            "success": success,
            "tool": tool_name,
            "prompt_used": prompt_name,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Prepared trace data: {trace_data}")
        return trace_data

def create_tool_runner() -> LangfuseRunner:
    logger.info("Creating new LangfuseRunner instance")
    return LangfuseRunner()