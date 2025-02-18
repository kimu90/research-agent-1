import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from langfuse import Langfuse
from dotenv import load_dotenv
import time
# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console and file handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler("langfuse_prompts_router.log")

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

# Langfuse setup
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Add prompt configurations with clear categories
PROMPT_CONFIGS = {
    "general": {
        "template": """Analyze the dataset with a focus on {{focus_area}}. 
        Consider the key patterns, trends, and insights that emerge from the data. 
        Provide a comprehensive analysis that highlights significant findings.""",
        "config": {"model": "gpt-4", "temperature": 0.7},
        "labels": ["analysis", "general", "basic"],
        "variants": {
            "concise": {
                "template": "Provide a brief analysis of {{dataset}} focusing on {{focus_area}}",
                "labels": ["analysis", "general", "concise"]
            },
            "detailed": {
                "template": "Perform an in-depth analysis of {{dataset}}, examining {{focus_area}}",
                "labels": ["analysis", "general", "detailed"]
            }
        }
    },
    "detailed": {
        "template": """Perform a detailed analysis of {{dataset}} focusing on {{metrics}}. 
        Include statistical measures, outliers, and relationships between variables.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "labels": ["analysis", "detailed", "statistical"],
        "variants": {
            "technical": {
                "template": "Technical analysis of {{dataset}} with focus on {{metrics}}",
                "labels": ["analysis", "detailed", "technical"]
            },
            "business": {
                "template": "Business-focused analysis of {{dataset}} examining {{metrics}}",
                "labels": ["analysis", "detailed", "business"]
            }
        }
    },
    "comparative": {
        "template": """Compare the following aspects of {{dataset}}: {{aspects}}. 
        Analyze similarities, differences, and relationships.""",
        "config": {"model": "gpt-4", "temperature": 0.3},
        "labels": ["analysis", "comparative"],
        "variants": {
            "simple": {
                "template": "Basic comparison of {{aspects}} in {{dataset}}",
                "labels": ["analysis", "comparative", "simple"]
            },
            "complex": {
                "template": "Detailed comparison with statistical analysis of {{aspects}} in {{dataset}}",
                "labels": ["analysis", "comparative", "complex"]
            }
        }
    },
    "species": {
        "template": """Analyze the species data focusing on {{focus_area}}. 
        Consider biological characteristics, habitat preferences, and population trends.""",
        "config": {"model": "gpt-4", "temperature": 0.5},
        "labels": ["analysis", "species", "biological"],
        "variants": {
            "ecological": {
                "template": "Analyze ecological aspects of species in {{dataset}}",
                "labels": ["analysis", "species", "ecological"]
            },
            "conservation": {
                "template": "Analyze conservation status and threats in {{dataset}}",
                "labels": ["analysis", "species", "conservation"]
            }
        }
    }
}

# Pydantic Models
class PromptConfig(BaseModel):
    model: str
    temperature: float
    other_params: Dict[str, Any] = {}

class PromptTemplate(BaseModel):
    name: str
    content: str
    version: Optional[str] = None  # Make version optional with a default of None
    config: PromptConfig
    labels: List[str] = []
    variants: Optional[Dict[str, Any]] = None

class PromptResponse(BaseModel):
    templates: List[PromptTemplate]
    categories: List[str]

def initialize_langfuse():
    """Initialize Langfuse client with error handling"""
    try:
        langfuse = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        # Verify authentication
        if not langfuse.auth_check():
            raise ValueError("Langfuse authentication failed")
        return langfuse
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Langfuse: {str(e)}"
        )

@api_router.get("/prompts", response_model=PromptResponse)
async def get_prompts():
    """
    Retrieve predefined prompts with tracking support.
    """
    request_id = f"req_{int(time.time())}"
    logger.info(f"[{request_id}] Starting GET /prompts request")
    
    try:
        prompts = []
        categories = set()
        
        # Process prompts from PROMPT_CONFIGS
        for name, config in PROMPT_CONFIGS.items():
            # Create template for main prompt
            template = PromptTemplate(
                name=name,
                content=config['template'],
                config=PromptConfig(
                    model=config['config']['model'],
                    temperature=config['config']['temperature']
                ),
                labels=config['labels'],
                variants=config.get('variants', {})
            )
            prompts.append(template)
            
            # Collect categories from labels
            categories.update(config['labels'])
            
            # Collect categories from variants
            if 'variants' in config:
                for variant in config['variants'].values():
                    categories.update(variant.get('labels', []))
        
        # Log summary
        logger.info(f"[{request_id}] Prompt retrieval summary:")
        logger.info(f"- Total prompts: {len(prompts)}")
        logger.info(f"- Categories found: {sorted(list(categories))}")
        
        # Attempt to track prompt retrieval in Langfuse
        try:
            langfuse = Langfuse()
            langfuse.track_prompt_retrieval(
                num_prompts=len(prompts),
                categories=list(categories)
            )
        except Exception as tracking_error:
            logger.warning(f"Langfuse tracking failed: {tracking_error}")
        
        # Prepare and return response
        response = PromptResponse(
            templates=prompts,
            categories=sorted(list(categories))
        )
        
        logger.info(f"[{request_id}] Prompt retrieval completed successfully")
        return response
        
    except Exception as e:
        error_msg = f"Error retrieving prompts: {str(e)}"
        logger.error(f"[{request_id}] {error_msg}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )
@api_router.get("/health")
async def health_check():
    """
    Perform a health check on the prompts service and Langfuse connection.
    
    Returns:
        Health status information
    """
    try:
        # Initialize Langfuse client
        langfuse = initialize_langfuse()
        
        # Verify Langfuse connection
        if not langfuse.auth_check():
            return {
                "status": "degraded",
                "service": "langfuse-prompts-api",
                "version": "1.0.0",
                "error": "Langfuse authentication failed"
            }
            
        # Get total number of prompts
        all_prompts = langfuse.get_prompts()
        prompt_count = len(all_prompts)
        
        # Get unique categories
        categories = set()
        for prompt in all_prompts:
            categories.update(prompt.labels)
        
        return {
            "status": "healthy",
            "service": "langfuse-prompts-api",
            "version": "1.0.0",
            "total_prompts": prompt_count,
            "available_categories": sorted(list(categories)),
            "langfuse_connection": "healthy"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e),
            "service": "langfuse-prompts-api",
            "version": "1.0.0"
        }