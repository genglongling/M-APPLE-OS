import os
from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
# from deepseek import Deepseek

def get_llm_client(model_type="google"):
    """
    Initialize and return an LLM client based on the specified model type.
    
    Args:
        model_type (str): The type of LLM client to initialize. Options are:
            - "openai": OpenAI client
            - "anthropic": Anthropic client
            - "google": Google (Gemini) client
            - "deepseek": Deepseek client
            
    Returns:
        The initialized LLM client
        
    Raises:
        ValueError: If an unsupported model type is specified
    """
    if model_type == "openai":
        return OpenAI()
    elif model_type == "anthropic":
        return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_type == "google":
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai
    elif model_type == "deepseek":
        return Deepseek(api_key=os.getenv("DEEPSEEK_API_KEY"))
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 