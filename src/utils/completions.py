import time
import random
from openai import OpenAI, RateLimitError
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def completions_create(client, messages: list, model: str, model_type: str = "google", max_retries: int = 5, base_delay: float = 1.0) -> str:
    """
    Sends a request to the client's `completions.create` method to interact with the language model.
    Includes exponential backoff retry logic for rate limiting.

    Args:
        client: The LLM client object (OpenAI or Google)
        messages (list[dict]): A list of message objects containing chat history for the model.
        model (str): The model to use for generating tool calls and responses.
        model_type (str): The type of model to use ("openai" or "google"). Defaults to "google".
        max_retries (int): Maximum number of retry attempts for rate limiting.
        base_delay (float): Base delay in seconds for exponential backoff.

    Returns:
        str: The content of the model's response.
    """
    # Check if we're using OpenAI or Google
    is_openai = model_type == "openai"
    is_google = model_type == "google"

    if is_openai:
        # Ensure OpenAI API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.3,
                    max_tokens=3000
                )
                return str(response.choices[0].message.content)
            except RateLimitError as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limit hit. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            except Exception as e:
                if "Missing bearer or basic authentication" in str(e):
                    raise ValueError("Invalid or missing OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
                raise e
    elif is_google:
        # Ensure Google API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        for attempt in range(max_retries):
            try:
                # Convert messages to Google's format
                google_messages = []
                for msg in messages:
                    role = "user" if msg["role"] == "user" else "model"
                    google_messages.append({"role": role, "parts": [msg["content"]]})

                # Create a new chat session
                chat = client.GenerativeModel(model).start_chat(history=google_messages)
                
                # Get the last message
                last_message = google_messages[-1]["parts"][0]
                
                # Send the message and get response
                response = chat.send_message(last_message)
                return str(response.text)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                
                # Calculate delay with exponential backoff and jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Error occurred. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    """
    Builds a structured prompt that includes the role and content.

    Args:
        prompt (str): The actual content of the prompt.
        role (str): The role of the speaker (e.g., user, assistant).

    Returns:
        dict: A dictionary representing the structured prompt.
    """
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}


def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the current chat history.
        msg (str): The message to append.
        role (str): The role type (e.g. 'user', 'assistant', 'system')
    """
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        if messages is None:
            messages = []

        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        """Add a message to the queue.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(0)
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """
        super().__init__(messages, total_length)

    def append(self, msg: str):
        """Add a message to the queue. The first messaage will always stay fixed.

        Args:
            msg (str): The message to be added to the queue
        """
        if len(self) == self.total_length:
            self.pop(1)
        super().append(msg)