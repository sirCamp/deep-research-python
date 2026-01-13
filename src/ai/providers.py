import os
import json
import tiktoken
from pathlib import Path
from openai import OpenAI
from typing import Optional, Union
from .text_splitter import RecursiveCharacterTextSplitter

# Optional Bedrock support
try:
    import boto3
    from botocore.config import Config
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

# Ensure environment variables are loaded
if not os.getenv("OPENAI_KEY") and not os.getenv("FIRECRAWL_KEY"):
    try:
        from dotenv import load_dotenv
        # Load environment variables from .env.local in the project root
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / ".env.local"
        load_dotenv(env_path)
    except ImportError:
        pass


class AIProvider:
    def __init__(self):
        # Initialize OpenAI clients for different providers
        self.openai_client = None
        self.nvidia_client = None
        self.fireworks_client = None
        self.custom_client = None
        self.openrouter_client = None
        self.bedrock_client = None
        self.bedrock_model = None

        # Initialize providers based on available API keys
        if os.getenv("OPENAI_KEY"):
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_KEY"),
                base_url=os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
            )

        if os.getenv("NVIDIA_API_KEY"):
            self.nvidia_client = OpenAI(
                api_key=os.getenv("NVIDIA_API_KEY"),
                base_url="https://integrate.api.nvidia.com/v1"
            )

        if os.getenv("FIREWORKS_KEY"):
            self.fireworks_client = OpenAI(
                api_key=os.getenv("FIREWORKS_KEY"),
                base_url="https://api.fireworks.ai/inference/v1"
            )

        if os.getenv("OPEN_ROUTER_KEY"):
            self.openrouter_client = OpenAI(
                api_key=os.getenv("OPEN_ROUTER_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

        if os.getenv("CUSTOM_MODEL") and self.openai_client:
            self.custom_client = self.openai_client

        # Initialize AWS Bedrock client
        if BEDROCK_AVAILABLE and os.getenv("AWS_BEDROCK_ENABLED", "").lower() == "true":
            try:
                region = os.getenv("AWS_REGION", "us-east-1")
                # Increase timeouts for long-running operations like report generation
                bedrock_timeout = int(os.getenv("AWS_BEDROCK_TIMEOUT", "300"))  # 5 minutes default
                config = Config(
                    region_name=region,
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    read_timeout=bedrock_timeout,
                    connect_timeout=60
                )
                self.bedrock_client = boto3.client(
                    "bedrock-runtime",
                    config=config
                )
                # Default to Claude 3.5 Sonnet, can be overridden via env var
                self.bedrock_model = os.getenv(
                    "AWS_BEDROCK_MODEL",
                    "anthropic.claude-3-5-sonnet-20241022-v2:0"
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Bedrock client: {e}")
                self.bedrock_client = None

    def get_model(self) -> tuple[Optional[OpenAI], str]:
        """Get the best available model and client.

        Returns:
            tuple: (client, model_name) where client is None for Bedrock provider
        """
        # Priority order based on the TypeScript version
        if self.custom_client and os.getenv("CUSTOM_MODEL"):
            custom_model = os.getenv("CUSTOM_MODEL")
            if not custom_model:
                raise ValueError("CUSTOM_MODEL environment variable is empty")
            return self.custom_client, custom_model

        # AWS Bedrock (high priority when enabled)
        if self.bedrock_client and self.bedrock_model:
            return None, self.bedrock_model  # None signals Bedrock provider

        # OpenRouter DeepSeek R1
        if self.openrouter_client:
            return self.openrouter_client, "deepseek/deepseek-r1-0528:free"

        # NVIDIA models (start with smaller, more stable models)
        if self.nvidia_client:
            return self.nvidia_client, "meta/llama-3.1-70b-instruct"

        # Fireworks DeepSeek R1
        if self.fireworks_client:
            return self.fireworks_client, "accounts/fireworks/models/deepseek-r1"

        # OpenAI fallback
        if self.openai_client:
            return self.openai_client, "gpt-4o-mini"

        raise ValueError("No model found. Please set at least one API key.")

    def _call_bedrock(self, system_prompt: str, user_prompt: str, schema: dict = None, timeout: int = 60):
        """Call AWS Bedrock API with Claude models.

        Args:
            system_prompt: System message for the model
            user_prompt: User message/query
            schema: Optional JSON schema for structured output (uses tool_use)
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON response or raw text content
        """
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized")

        messages = [{"role": "user", "content": user_prompt}]

        # Use higher max_tokens for report generation (reports need more space)
        max_tokens = int(os.getenv("AWS_BEDROCK_MAX_TOKENS", "16384"))

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages
        }

        # If schema provided, use tool_use for structured output
        if schema:
            request_body["tools"] = [{
                "name": "respond_with_structure",
                "description": "Respond with the requested structured data",
                "input_schema": schema
            }]
            request_body["tool_choice"] = {"type": "tool", "name": "respond_with_structure"}

        print(f"DEBUG: Calling Bedrock model {self.bedrock_model} with max_tokens={max_tokens}")

        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.bedrock_model,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
        except Exception as e:
            print(f"ERROR: Bedrock invoke_model failed: {e}")
            raise

        response_body = json.loads(response["body"].read())
        print(f"DEBUG: Bedrock response stop_reason: {response_body.get('stop_reason')}")
        print(f"DEBUG: Bedrock full response: {json.dumps(response_body, indent=2)[:2000]}")

        # Parse response based on whether we used tools
        if schema and response_body.get("content"):
            for content_block in response_body["content"]:
                if content_block.get("type") == "tool_use":
                    result = content_block.get("input", {})
                    print(f"DEBUG: Extracted tool_use input with {len(str(result))} chars")
                    return result

        # Fallback to text content
        print(f"DEBUG: No tool_use found, falling back to text content")
        if response_body.get("content"):
            for content_block in response_body["content"]:
                if content_block.get("type") == "text":
                    text = content_block.get("text", "")
                    print(f"DEBUG: Got text content with {len(text)} chars")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        print(f"DEBUG: Text is not JSON, returning as content")
                        return {"content": text}

        print(f"DEBUG: No content found, returning raw response_body")
        return response_body

    def generate_object(self, system_prompt: str, user_prompt: str, schema: dict, timeout: int = 60):
        """Generate structured output using the best available model"""
        client, model_name = self.get_model()

        # Handle AWS Bedrock (client is None when using Bedrock)
        if client is None and self.bedrock_client:
            return self._call_bedrock(system_prompt, user_prompt, schema, timeout)

        # For OpenAI models, use structured outputs
        if "gpt-" in model_name:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                timeout=timeout
            )
        else:
            # For other models, use tool calling to get structured output
            tools = [{
                "type": "function",
                "function": {
                    "name": "respond_with_structure",
                    "description": "Respond with the requested structured data",
                    "parameters": schema
                }
            }]

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "respond_with_structure"}},
                timeout=timeout
            )

        return response

def parse_structured_response(response):
    """Parse structured response from either tool calls, function calls, or Bedrock responses"""

    # Handle Bedrock responses (already parsed as dict)
    if isinstance(response, dict):
        return response

    if hasattr(response, 'choices') and response.choices:
        choice = response.choices[0]

        # Check for tool calls (new format)
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            return json.loads(choice.message.tool_calls[0].function.arguments)

        # Check for function call (deprecated format)
        elif hasattr(choice, 'function_call'):
            return json.loads(choice.function_call.arguments)

        # Fallback to message content
        else:
            return json.loads(choice.message.content)

    raise ValueError("Unable to parse response")


# Initialize global provider
_ai_provider = AIProvider()

def get_model() -> tuple[Optional[OpenAI], str]:
    """Get the current model client and name.

    Returns:
        tuple: (client, model_name) where client may be None for Bedrock provider
    """
    return _ai_provider.get_model()

def generate_object(system_prompt: str, user_prompt: str, schema: dict, timeout: int = 60):
    """Generate structured output"""
    return _ai_provider.generate_object(system_prompt, user_prompt, schema, timeout)

def parse_response(response):
    """Parse structured response from API"""
    return parse_structured_response(response)


MIN_CHUNK_SIZE = 140

def trim_prompt(prompt: str, context_size: int = None) -> str:
    """Trim prompt to maximum context size"""
    if context_size is None:
        context_size = int(os.getenv("CONTEXT_SIZE", "128000"))
    
    if not prompt:
        return ""
    
    try:
        encoder = tiktoken.get_encoding("o200k_base")
        length = len(encoder.encode(prompt))
        
        if length <= context_size:
            return prompt
        
        overflow_tokens = length - context_size
        # On average it's 3 characters per token, so multiply by 3 to get a rough estimate
        chunk_size = len(prompt) - overflow_tokens * 3
        
        if chunk_size < MIN_CHUNK_SIZE:
            return prompt[:MIN_CHUNK_SIZE]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0
        )
        
        chunks = splitter.split_text(prompt)
        trimmed_prompt = chunks[0] if chunks else ""
        
        # Last catch, recursively trim if needed
        if len(trimmed_prompt) == len(prompt):
            return trim_prompt(prompt[:chunk_size], context_size)
        
        # Recursively trim until the prompt is within the context size
        return trim_prompt(trimmed_prompt, context_size)
    
    except Exception as e:
        print(f"Error trimming prompt: {e}")
        # Fallback to simple truncation
        return prompt[:context_size * 3]  # Rough estimate


def is_bedrock_available() -> bool:
    """Check if AWS Bedrock is available and configured"""
    return _ai_provider.bedrock_client is not None


def get_active_provider() -> str:
    """Get the name of the currently active provider"""
    client, model = _ai_provider.get_model()
    if client is None and _ai_provider.bedrock_client:
        return f"bedrock ({model})"
    elif _ai_provider.custom_client and os.getenv("CUSTOM_MODEL"):
        return f"custom ({model})"
    elif _ai_provider.openrouter_client and client == _ai_provider.openrouter_client:
        return f"openrouter ({model})"
    elif _ai_provider.nvidia_client and client == _ai_provider.nvidia_client:
        return f"nvidia ({model})"
    elif _ai_provider.fireworks_client and client == _ai_provider.fireworks_client:
        return f"fireworks ({model})"
    elif _ai_provider.openai_client:
        return f"openai ({model})"
    return "unknown"
