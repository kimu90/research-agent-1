from pydantic import BaseModel
from openai import OpenAI
import instructor
import logging
import time
from typing import Optional


def json_model_wrapper(
    system_prompt: str,
    user_prompt: str,
    prompt: any,
    base_model: [BaseModel],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    trace: Optional["QueryTrace"] = None
) -> Optional[BaseModel]:
    """
    Wrapper for model responses that return structured JSON data
    Uses instructor to enforce response schema with performance tracking
    """
    logging.info(f"Starting structured inference with model {model}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        client = instructor.from_openai(OpenAI())
        
        start = time.time()
        obj, completion = client.chat.completions.create_with_completion(
            model=model,
            response_model=base_model,
            messages=messages,
            temperature=temperature
        )
        duration = time.time() - start
        
        # Log performance metrics
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens
        
        logging.info(f"Completion generated in {duration:.2f}s")
        logging.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
        logging.info(f"Processing rate: {total_tokens/duration:.2f} tokens/sec")
        
        # Log schema compliance
        logging.info(f"Response successfully validated against {base_model.__name__} schema")
        
        # Track token usage if trace is provided
        if trace and hasattr(completion, 'usage'):
            trace.add_token_usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                model_name=model,
                prompt_id=prompt.id if hasattr(prompt, 'id') else None
            )
        
        return obj
        
    except Exception as e:
        logging.error(f"Error in json_model_wrapper: {str(e)}")
        return None