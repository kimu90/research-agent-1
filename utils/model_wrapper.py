from typing import Optional, Any
import logging
import time
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
import os
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize clients for OpenAI and Groq
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

openai_client = OpenAI(api_key=openai_api_key)
groq_client = Groq(api_key=groq_api_key)

def model_wrapper(
    system_prompt: str,
    user_prompt: str,
    prompt: Any,
    model: str = "gpt-4",
    temperature: float = 0,
    host: str = "openai",
    token_tracker: Optional[Any] = None
) -> str:
    """
    Wrapper for text model responses supporting OpenAI and Groq with performance tracking

    Args:
        system_prompt: The system prompt to guide the model's behavior
        user_prompt: The user's input prompt
        prompt: The original prompt object containing metadata
        model: The model to use (default: "gpt-4")
        temperature: Controls randomness in the response (default: 0)
        host: The API host to use ("openai" or "groq") (default: "openai")
        token_tracker: Optional token usage tracker object

    Returns:
        str: The model's response text
    """
    logging.info(f"Start inference with model {model} on host {host}")
    
    # Prepare messages for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    start = time.time()

    try:
        # Select the appropriate client based on the host
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

        # Extract the result and performance metrics
        result = completion.choices[0].message.content
        duration = time.time() - start

        # Log performance metrics
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_tokens = completion.usage.total_tokens

        logging.info(f"Inference completed in {duration:.2f}s")
        logging.info(f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}")
        logging.info(f"Processing speed - {total_tokens / duration:.2f} tokens/second")

        # Track token usage if token_tracker is provided
        if token_tracker and hasattr(token_tracker, 'add_usage'):
            token_tracker.add_usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                model=model,
                prompt_id=prompt.id if hasattr(prompt, 'id') else None
            )

        return result

    except Exception as e:
        logging.error(f"Error in model_wrapper: {str(e)}", exc_info=True)
        raise
