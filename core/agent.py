"""
sage_agent.py

Agent-side LLM interface for SAGE.
Provides real API calls to various LLM providers (OpenAI, Anthropic, Google AI)
with robust error handling and retry mechanisms.
"""

from __future__ import annotations

import os
import time
import random
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Import token tracker
try:
    from tools.token_tracker import get_tracker
    TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    TOKEN_TRACKING_AVAILABLE = False
    def get_tracker():
        return None

# Suppress warnings
warnings.filterwarnings("ignore")

# Load project-specific environment variables
def load_sage_env():
    """Load environment variables from SAGE project config file."""
    config_file = Path(__file__).parent / "sage_config.env"
    if config_file.exists():
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"Loaded SAGE environment variables from {config_file}")

# Load environment variables when module is imported
load_sage_env()

# Try to import API clients
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    genai = None


@dataclass
class AgentConfig:
    provider: str = "openai"
    model: str = "gpt-5-nano"
    api_key: Optional[str] = None


class SAGEAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def build_prompt(self, system_text: str, user_text: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

    def chat(self, messages: List[Dict[str, str]]) -> str:
        return ask_agent(self.config.model, messages)


def get_content_from_message(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract content from a message, handling both simple string content
    and structured content with text/image_url types.
    """
    parts = []
    if isinstance(message["content"], str):
        parts.append({
            "type": "text",
            "text": message["content"]
        })
        return parts
        
    for content in message["content"]:
        if content["type"] == "text":
            parts.append(content)
        elif content["type"] == "image_url":
            image_url = content["image_url"]["url"]
            if image_url.startswith("data:image"):
                header, base64_data = image_url.split(',', 1)
                if base64_data.startswith('iVBOR'):
                    media_type = 'image/png'
                else:
                    media_type = header.split(':')[1].split(';')[0]
                
                image_content = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data,
                    }
                }
                parts.append(image_content)
    return parts


# Global chat session cache for persistent conversations
_chat_sessions = {}

def ask_agent(model: str, history: List[Dict[str, Any]]) -> str:
    """
    Ask an LLM agent for the next experiment step.
    
    Parameters
    ----------
    model : str
        The LLM model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-latest', 'gemini-1.5-pro')
    history : List[Dict[str, Any]]
        The conversation history/log from the experiment
        
    Returns
    -------
    str
        The agent's response with the next experiment instruction
    """
    max_retries = 5
    count = 0
    system_cache = {}
    
    # Initialize API clients if available
    if OPENAI_AVAILABLE:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
    
    if ANTHROPIC_AVAILABLE:
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    if GOOGLE_AI_AVAILABLE:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    while count < max_retries:
        try:
            # Handle Claude models
            if model.startswith('claude') and ANTHROPIC_AVAILABLE:
                selected_model = 'claude-3-5-sonnet-latest'
                system_content = None
                messages = []

                for msg in history:
                    content = get_content_from_message(msg)
                    if msg["role"] == "system":
                        system_content = content
                    else:
                        role = "user" if msg["role"] == "user" else "assistant"
                        messages.append({
                            "role": role,
                            "content": content
                        })
                
                # Use cached system prompt if available
                if selected_model in system_cache and system_cache[selected_model] == system_content:
                    system_prompt = None
                else:
                    system_prompt = system_content
                    system_cache[selected_model] = system_content
                
                # Prepare the API call parameters
                api_params = {
                    "model": selected_model,
                    "messages": messages,
                    "max_tokens": 4096
                }
                
                if system_prompt:
                    api_params["system"] = system_prompt
                
                response = anthropic_client.messages.create(**api_params)
                
                # Record token usage
                if TOKEN_TRACKING_AVAILABLE:
                    tracker = get_tracker()
                    if tracker:
                        # Anthropic API returns usage in response.usage
                        usage = getattr(response, 'usage', None)
                        if usage:
                            prompt_tokens = getattr(usage, 'input_tokens', 0)
                            completion_tokens = getattr(usage, 'output_tokens', 0)
                            tracker.record_usage(
                                model=selected_model,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens
                            )
                
                return response.content[0].text
                
            # Handle OpenAI models
            elif model in ['gpt-5-nano', 'gpt-5-mini', 'gpt-5', 'gpt-4o-new', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4'] and OPENAI_AVAILABLE:
                if model == 'gpt-4o-new':
                    model = 'gpt-4o-2024-11-20'
                
                # Use new OpenAI client for GPT-5 models
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # GPT-5 models use max_completion_tokens instead of max_tokens
                if model.startswith('gpt-5'):
                    params = {
                        "model": model,
                        "messages": history,
                        "max_completion_tokens": 4096,
                    }
                else:
                    params = {
                        "model": model,
                        "messages": history,
                        "max_tokens": 4096,
                    }
                
                r = client.chat.completions.create(**params)
                
                # Record token usage
                if TOKEN_TRACKING_AVAILABLE:
                    tracker = get_tracker()
                    if tracker:
                        # OpenAI API returns usage in response.usage
                        usage = getattr(r, 'usage', None)
                        if usage:
                            prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                            completion_tokens = getattr(usage, 'completion_tokens', 0)
                            
                            # Check for cached tokens (OpenAI API may provide this in different formats)
                            cached_tokens = 0
                            if hasattr(usage, 'prompt_tokens_details'):
                                # Newer API format
                                details = usage.prompt_tokens_details
                                if isinstance(details, dict):
                                    cached_tokens = details.get('cached_tokens', 0)
                                elif hasattr(details, 'cached_tokens'):
                                    cached_tokens = details.cached_tokens
                            
                            tracker.record_usage(
                                model=model,
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                cached_tokens=cached_tokens
                            )
                
                return r.choices[0].message.content
                
            # Handle Gemini models via Google AI SDK
            elif model.startswith('gemini') and GOOGLE_AI_AVAILABLE:
                # Convert history to Gemini format
                gemini_messages = []
                for msg in history:
                    if msg["role"] == "system":
                        # Gemini doesn't have system messages, prepend to first user message
                        if gemini_messages and gemini_messages[-1]["role"] == "user":
                            gemini_messages[-1]["parts"] = [msg["content"] + "\n\n" + gemini_messages[-1]["parts"][0]]
                        else:
                            gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                    elif msg["role"] == "user":
                        gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                    elif msg["role"] == "assistant":
                        gemini_messages.append({"role": "model", "parts": [msg["content"]]})
                
                # Use Gemini Pro model - correct model names for Google AI
                if "1.5-pro" in model or "pro" in model:
                    model_name = "gemini-pro-latest"
                elif "flash" in model:
                    model_name = "gemini-2.5-flash"
                else:
                    model_name = "gemini-pro-latest"
                # Create session key for persistent chat
                session_key = f"{model_name}_{hash(str(history[:2]))}"  # Use system + user prompt as key
                
                # Get or create persistent chat session
                if session_key not in _chat_sessions:
                    gemini_model = genai.GenerativeModel(model_name)
                    # Start with empty history, we'll send all messages
                    _chat_sessions[session_key] = gemini_model.start_chat(history=[])
                
                chat = _chat_sessions[session_key]
                
                # Send the last message with retry logic
                last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
                
                # Retry logic for Gemini API quota issues
                max_retries = 3
                retry_delay = 60  # 60 seconds
                
                for attempt in range(max_retries):
                    try:
                        # For persistent chat, send the last message
                        response = chat.send_message(last_message)
                        
                        # Record token usage (if available)
                        if TOKEN_TRACKING_AVAILABLE:
                            tracker = get_tracker()
                            if tracker:
                                # Google AI API may provide usage info in response.usage_metadata
                                usage_metadata = getattr(response, 'usage_metadata', None)
                                if usage_metadata:
                                    prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
                                    completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
                                    tracker.record_usage(
                                        model=model_name,
                                        prompt_tokens=prompt_tokens,
                                        completion_tokens=completion_tokens
                                    )
                        
                        return response.text
                    except Exception as e:
                        if "quota" in str(e).lower() or "429" in str(e):
                            if attempt < max_retries - 1:
                                print(f"Gemini API quota exceeded. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}...")
                                import time
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                                continue
                            else:
                                print(f"Gemini API quota exceeded after {max_retries} attempts. Please upgrade to paid plan or use a different LLM.")
                                return "Error: Gemini API quota exceeded. Please upgrade to paid plan or use --agent_llm gpt-4 or claude-3-sonnet."
                        else:
                            raise e
                
            else:
                print(f"❌ ERROR: Unrecognized model name: {model}")
                available_models = ['gpt-5-nano', 'gpt-5-mini', 'gpt-5', 'gpt-4o-new', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4']
                print(f"❌ Available models: {available_models}")
                print(f"❌ OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
                raise ValueError(f"Unrecognized model name: {model}. Available models: {available_models}")

        except Exception as e:
            # Handle OpenAI API errors (both old and new client)
            if "rate_limit" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                if not OPENAI_AVAILABLE:
                    print(f"❌ ERROR: OpenAI not available and rate limit hit: {str(e)}")
                    print(f"❌ OPENAI_AVAILABLE: {OPENAI_AVAILABLE}")
                    raise RuntimeError(f"OpenAI not available and rate limit hit: {str(e)}")
                count += 1
                print(f'OpenAI API rate limit error: {str(e)}')
                wait_time = 60 + 10*random()  # Random wait between 60-70 seconds
                print(f"Attempt {count}/{max_retries}. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                print(f'❌ ERROR: OpenAI model not found: {str(e)}')
                print(f'❌ Requested model: {model}')
                available_models = ['gpt-5-nano', 'gpt-5-mini', 'gpt-5', 'gpt-4o-new', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-4']
                print(f'❌ Available models: {available_models}')
                raise ValueError(f"OpenAI model not found: {str(e)}. Requested: {model}, Available: {available_models}")
            else:
                print(f'❌ ERROR: OpenAI API error: {str(e)}')
                print(f'❌ Error type: {type(e).__name__}')
                print(f'❌ OPENAI_AVAILABLE: {OPENAI_AVAILABLE}')
                raise RuntimeError(f"OpenAI API error: {str(e)}")

        except anthropic.RateLimitError as e:
            if not ANTHROPIC_AVAILABLE:
                print(f'❌ ERROR: Anthropic not available and rate limit hit: {str(e)}')
                print(f'❌ ANTHROPIC_AVAILABLE: {ANTHROPIC_AVAILABLE}')
                raise RuntimeError(f"Anthropic not available and rate limit hit: {str(e)}")
            count += 1
            print(f'Anthropic API error: {str(e)}')
            wait_time = 60 + 10*random()
            print(f"Attempt {count}/{max_retries}. Waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)

        except anthropic.BadRequestError as e:
            print(f"❌ ERROR: Bad request to Anthropic: {str(e)}")
            print(f"❌ ANTHROPIC_AVAILABLE: {ANTHROPIC_AVAILABLE}")
            raise RuntimeError(f"Bad request to Anthropic: {str(e)}")
                
        except Exception as e:
            print(f"❌ ERROR: Unexpected error: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            traceback.print_exc()
            raise RuntimeError(f"Unexpected error: {str(e)}")
            
    print(f"❌ ERROR: Exceeded max retries ({max_retries}) for model: {model}")
    raise RuntimeError(f"Exceeded max retries ({max_retries}) for model: {model}")


def _generate_mock_sage_response(history: List[Dict[str, Any]]) -> str:
    """
    Generate a mock SAGE response when real API calls are not available.
    This simulates the behavior of a real SAGE agent.
    """
    # Check if we should provide a final description
    if len(history) > 20:  # After many rounds, provide final description
        return "[DESCRIPTION] Based on the analysis, this SAE feature appears to be related to programming concepts, specifically Python syntax and function definitions. The feature shows consistent activation patterns with code-related text inputs."
    
    # 返回符合DESIGN_TEST格式的模拟响应，避免无限循环
    return """TESTING HYPOTHESIS: Legal motion/filing constructions produce strong activations, especially when "file/filing" occurs within 0–3 tokens of "motion/petition."

TEST DESIGN:
Prompt: 'Positive: The defendant will file a petition for post-conviction relief. Control: The defendant will petition for post-conviction relief.'

[TOOL] model.run prompt='Positive: The defendant will file a petition for post-conviction relief. Control: The defendant will petition for post-conviction relief.'"""


def validate_agent_response(response: str) -> bool:
    """
    Validate that an agent response is meaningful and not empty.
    
    Parameters
    ----------
    response : str
        The agent response to validate
        
    Returns
    -------
    bool
        True if response is valid, False otherwise
    """
    if not response or not response.strip():
        return False
    
    # Check for minimum meaningful length
    if len(response.strip()) < 10:
        return False
    
    # Check for common error patterns
    error_patterns = [
        "error",
        "failed",
        "timeout",
        "quota exceeded",
        "rate limit",
        "unavailable"
    ]
    
    response_lower = response.lower()
    for pattern in error_patterns:
        if pattern in response_lower:
            return False
    
    return True


__all__ = ["AgentConfig", "SAGEAgent", "ask_agent"]


