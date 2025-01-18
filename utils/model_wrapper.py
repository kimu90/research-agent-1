from pydantic import BaseModel
from openai import OpenAI
from groq import Groq
import instructor
import logging
import time
from typing import Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
groq_client = Groq(api_key=groq_api_key)

def model_wrapper(
    system_prompt: str,
    user_prompt: str,
    prompt: any,
    model: str = "gpt-4",
    temperature: float = 0,
    host: str = "openai",
    trace: Optional["QueryTrace"] = None
) -> str:
    """
    Wrapper for text model responses supporting OpenAI and Groq with performance tracking
    """
    logging.info(f"Start inference with model {model} on host {host}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    start = time.time()
    
    try:
        if host == "openai":
            completion = openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
        elif host == "groq":
            completion = groq_client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
            )
        else:
            raise ValueError(f"Unsupported host: {host}")
        
        result = completion.choices[0].message.content
        duration = time.time() - start
        
        # Log performance metrics
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens
        
        logging.info(f"Inference completed in {duration:.2f}s")
        logging.info(f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
        logging.info(f"Processing speed - {total_tokens/duration:.2f} tokens/second")
        
        # Track token usage if trace is provided
        if trace and hasattr(completion, 'usage'):
            trace.add_token_usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                model_name=model,
                prompt_id=prompt.id if hasattr(prompt, 'id') else None
            )
        
        return result
        
    except Exception as e:
        logging.error(f"Error in model_wrapper: {str(e)}")
        raise