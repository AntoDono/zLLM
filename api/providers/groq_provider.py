from typing import Any, Dict, Generator, List, Optional, Union
from groq import Groq
import requests
from .provider import BaseLLMProvider


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen/qwen3-32b", **kwargs):
        """Initialize the Groq provider.
        
        Args:
            api_key: Groq API key
            model: Default model to use
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.model = model
        self.client = Groq(api_key=api_key) if api_key else Groq()
    
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from Groq without streaming.
        
        Args:
            prompt: Input prompt string or list of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Groq-specific parameters
            
        Returns:
            Dict containing the response and metadata
        """
        # Handle prompt format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        # Prepare parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": temperature or 0.6,
            "max_completion_tokens": max_tokens or 4096,
            "top_p": kwargs.get("top_p", 0.95),
            "reasoning_effort": kwargs.get("reasoning_effort", "default"),
            "stream": False,
            "stop": kwargs.get("stop", None),
        }
        
        # Make the API call
        completion = self.client.chat.completions.create(**params)
        
        # Return response in a standardized format
        return {
            "content": completion.choices[0].message.content,
            "model": params["model"],
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
                "total_tokens": completion.usage.total_tokens if completion.usage else None,
            },
            "finish_reason": completion.choices[0].finish_reason,
        }
    
    def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate a streaming response from Groq.
        
        Args:
            prompt: Input prompt string or list of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Groq-specific parameters
            
        Yields:
            Dict containing response chunks and metadata
        """
        # Handle prompt format
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        
        # Prepare parameters
        params = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": temperature or 0.6,
            "max_completion_tokens": max_tokens or 4096,
            "top_p": kwargs.get("top_p", 0.95),
            "reasoning_effort": kwargs.get("reasoning_effort", "none"),
            "stream": True,
            "stop": kwargs.get("stop", None),
        }
        
        # Make the streaming API call
        completion = self.client.chat.completions.create(**params)
        
        # Yield response chunks
        for chunk in completion:
            delta_content = chunk.choices[0].delta.content or ""
            yield {
                "content": delta_content,
                "model": params["model"],
                "finish_reason": chunk.choices[0].finish_reason,
            }
    
    def get_model_list(self) -> List[str]:
        """Get list of available Groq models.
        
        Returns:
            List of model names
        """
        if not self.api_key:
            # Return a default list if no API key is available
            return []
        
        try:
            url = "https://api.groq.com/openai/v1/models"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            models_data = response.json()
            # Extract model IDs from the response
            return [model["id"] for model in models_data.get("data", [])]
        
        except Exception:
            # Fall back to default list if API call fails
            return []
