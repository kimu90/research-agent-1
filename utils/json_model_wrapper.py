from typing import Optional, Type, Any
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
import time
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def json_model_wrapper(
    system_prompt: str,
    user_prompt: str,
    prompt: Any,
    base_model: Type[BaseModel],
    model: str = "gpt-3.5-turbo",
    temperature: float = 0,
    max_retries: int = 3,
    token_tracker: Optional[Any] = None
) -> Optional[BaseModel]:
    """
    Wrapper for model responses that return structured JSON data
    
    Args:
        system_prompt: The system prompt to guide the model's behavior
        user_prompt: The user's input prompt
        prompt: The original prompt object containing metadata
        base_model: The Pydantic model class to validate the response against
        model: The model to use (default: "gpt-3.5-turbo")
        temperature: Controls randomness in the response (default: 0)
        max_retries: Maximum number of retry attempts for JSON parsing (default: 3)
        token_tracker: Optional token usage tracker object
        
    Returns:
        Optional[BaseModel]: The validated response object or None if validation fails
    """
    # Validate input parameters
    if not system_prompt or not user_prompt:
        logging.error("Missing system or user prompt")
        return None

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as init_error:
        logging.error(f"Failed to initialize OpenAI client: {str(init_error)}")
        return None

    # Prepare messages for API call
    messages = [
        {"role": "system", "content": f"{system_prompt}. Respond with a valid JSON object."},
        {"role": "user", "content": user_prompt}
    ]

    # Track total attempts
    attempts = 0

    while attempts < max_retries:
        try:
            # Start timing the API call
            start_time = time.time()

            # Make API call with JSON mode
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )

            # Extract response content
            response_text = response.choices[0].message.content.strip()
            
            # Log performance metrics
            duration = time.time() - start_time
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens

            logging.info(f"JSON Inference completed in {duration:.2f}s")
            logging.info(f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
            logging.info(f"Processing speed - {total_tokens/duration:.2f} tokens/second")

            # Attempt to parse JSON
            try:
                # First, try to parse as dictionary
                parsed_response = json.loads(response_text)
                
                # Validate against Pydantic model
                try:
                    validated_obj = base_model.model_validate(parsed_response)
                    
                    # Track token usage if tracker is provided
                    if token_tracker and hasattr(token_tracker, 'add_usage'):
                        token_tracker.add_usage(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            model=model,
                            prompt_id=getattr(prompt, 'id', None)
                        )
                    
                    return validated_obj
                
                except ValidationError as val_error:
                    logging.warning(f"Pydantic validation failed: {str(val_error)}")
                    logging.warning(f"Problematic JSON: {response_text}")
                    
                    # Optional: attempt to partially validate or transform
                    try:
                        # Try to create object with partial validation
                        partially_validated = base_model.model_construct(**parsed_response)
                        return partially_validated
                    except Exception as partial_error:
                        logging.error(f"Partial validation failed: {str(partial_error)}")
            
            except json.JSONDecodeError as json_error:
                logging.warning(f"JSON parsing failed (Attempt {attempts + 1}): {str(json_error)}")
                logging.warning(f"Problematic response: {response_text}")
        
        except Exception as api_error:
            logging.error(f"API call failed (Attempt {attempts + 1}): {str(api_error)}")
        
        # Increment attempts and potentially adjust strategy
        attempts += 1
        
        # Optional: add a more specific error prompt on subsequent attempts
        if attempts < max_retries:
            error_guidance = (
                f"The previous JSON response was invalid. "
                f"Please ensure your response is a valid JSON that matches the {base_model.__name__} schema. " 
                "The expected field is 'sources', not 'selected_sources'. Double-check your JSON formatting."
            )
            messages.append({"role": "system", "content": error_guidance})

    # Final fallback if all attempts fail
    logging.error(f"Failed to generate valid JSON response after {max_retries} attempts")
    return None