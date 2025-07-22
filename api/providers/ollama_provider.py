from typing import Any, Dict, Generator, List, Optional, Union
from ollama import chat
from .provider import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3.2", **kwargs):
        """Initialize the Ollama provider.
        
        Args:
            api_key: Not used for Ollama (local models)
            model: Default model to use
            **kwargs: Additional configuration
        """
        super().__init__(api_key, **kwargs)
        self.model = model
    
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from Ollama without streaming.
        
        Args:
            prompt: Input prompt string or list of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Ollama-specific parameters
            
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
            "stream": False,
        }
        
        # Add optional parameters
        if temperature is not None:
            params["options"] = params.get("options", {})
            params["options"]["temperature"] = temperature
        
        if max_tokens is not None:
            params["options"] = params.get("options", {})
            params["options"]["num_predict"] = max_tokens
        
        # Add any additional options from kwargs
        if "options" in kwargs:
            params["options"] = {**params.get("options", {}), **kwargs["options"]}
        
        # Make the API call
        response = chat(**params)
        
        # Return response in a standardized format
        return {
            "content": response["message"]["content"],
            "model": params["model"],
            "usage": {
                "prompt_tokens": response.get("prompt_eval_count"),
                "completion_tokens": response.get("eval_count"),
                "total_tokens": (response.get("prompt_eval_count", 0) + 
                                response.get("eval_count", 0)) if response.get("prompt_eval_count") else None,
            },
            "finish_reason": "stop",  # Ollama doesn't provide finish reason
        }
    
    def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate a streaming response from Ollama.
        
        Args:
            prompt: Input prompt string or list of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional Ollama-specific parameters
            
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
            "stream": True,
        }
        
        # Add optional parameters
        if temperature is not None:
            params["options"] = params.get("options", {})
            params["options"]["temperature"] = temperature
        
        if max_tokens is not None:
            params["options"] = params.get("options", {})
            params["options"]["num_predict"] = max_tokens
        
        # Add any additional options from kwargs
        if "options" in kwargs:
            params["options"] = {**params.get("options", {}), **kwargs["options"]}
        
        # Make the streaming API call
        stream = chat(**params)
        
        # Yield response chunks
        for chunk in stream:
            yield {
                "content": chunk["message"]["content"],
                "model": params["model"],
                "finish_reason": "stop" if chunk.get("done") else None,
            }
    
    def get_model_list(self) -> List[str]:
        """Get list of available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            import ollama
            models = ollama.list()
            return [model["name"] for model in models.get("models", [])]
        except Exception:
            # Return empty list if listing fails
            return []
    
    def validate_config(self) -> bool:
        """Validate the provider configuration.
        
        Ollama doesn't require an API key as it runs locally.
        
        Returns:
            True (always valid for local Ollama)
        """
        return True 