from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the LLM provider.
        
        Args:
            api_key: API key for authentication
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def generate(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a response from the LLM without streaming.
        
        Args:
            prompt: Input prompt string or list of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict containing the response and metadata
        """
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate a streaming response from the LLM.
        
        Args:
            prompt: Input prompt string or list of messages
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Dict containing response chunks and metadata
        """
        pass
    
    def validate_config(self) -> bool:
        """Validate the provider configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        return bool(self.api_key)
    
    def get_model_list(self) -> List[str]:
        """Get list of available models for this provider.
        
        Returns:
            List of model names
        """
        return []
