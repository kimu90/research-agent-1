from pydantic import BaseModel
from openai import OpenAI
import instructor
import logging
import time

def json_model_wrapper(
    system_prompt: str,
    user_prompt: str,
    prompt: any,
    base_model: BaseModel,
    model: str = "gpt-3.5-turbo",
    temperature=0,
) -> BaseModel:
    """
    Wrapper for model responses that return structured JSON data
    Uses instructor to enforce response schema
    
    Args:
        system_prompt: The system prompt to send
        user_prompt: The user prompt to send
        prompt: The prompt template (not used but kept for compatibility)
        base_model: The Pydantic model to validate response against
        model: The model to use (defaults to gpt-3.5-turbo)
        temperature: Model temperature (defaults to 0)
        
    Returns:
        BaseModel: The structured response matching the base_model schema
    """
    logging.info(f"Starting structured inference with model {model}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

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

    return obj