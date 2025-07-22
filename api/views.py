import json
import os
import importlib
import inspect
from typing import Dict, Type
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from .providers.provider import BaseLLMProvider
from .models import APIKey

# Provider instances cache
_provider_instances = {}

# Dynamically discovered providers
_discovered_providers = {}

def discover_providers():
    """Dynamically discover all provider classes in the providers directory."""
    global _discovered_providers
    
    if _discovered_providers:
        return _discovered_providers
    
    providers_dir = os.path.join(os.path.dirname(__file__), 'providers')
    
    for filename in os.listdir(providers_dir):
        if filename.endswith('_provider.py') and not filename.startswith('__'):
            module_name = filename[:-3]  # Remove .py extension
            module_path = f'api.providers.{module_name}'
            
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Find all classes that inherit from BaseLLMProvider
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseLLMProvider) and 
                        obj != BaseLLMProvider):
                        _discovered_providers[obj.__name__] = obj
            except Exception as e:
                print(f"Error loading provider {module_name}: {e}")
    
    return _discovered_providers


def get_model_providers() -> Dict[str, Type[BaseLLMProvider]]:
    """Build a dynamic mapping of models to their providers."""
    providers = discover_providers()
    model_mapping = {}
    
    for provider_class in providers.values():
        try:
            # Create a temporary instance to get the model list
            # For providers that need API keys, this might return empty list
            temp_instance = provider_class()
            models = temp_instance.get_model_list()
            
            for model in models:
                model_mapping[model] = provider_class
        except Exception:
            # If we can't instantiate without API key, skip for now
            # These will be handled dynamically when needed
            pass
    
    return model_mapping


def get_provider_instance(provider_class: Type[BaseLLMProvider], api_key: str = None) -> BaseLLMProvider:
    """Get or create a provider instance."""
    cache_key = f"{provider_class.__name__}:{api_key or 'default'}"
    if cache_key not in _provider_instances:
        # For providers that need API keys from environment
        if not api_key and hasattr(provider_class, '__name__'):
            # Try to get API key from environment based on provider name
            env_key_name = f"{provider_class.__name__.replace('Provider', '').upper()}_API_KEY"
            api_key = os.environ.get(env_key_name)
        _provider_instances[cache_key] = provider_class(api_key=api_key)
    return _provider_instances[cache_key]


def validate_api_key(request) -> tuple[bool, str, APIKey]:
    """Validate API key from request headers."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return False, "Missing or invalid Authorization header", None
    
    api_key = auth_header.replace("Bearer ", "", 1)  # Remove "Bearer " prefix
    
    try:
        api_key_obj = APIKey.objects.get(key=api_key, is_active=True)
        # Check if API key has exceeded limits
        if not api_key_obj.can_make_request():
            return False, "API key has exceeded usage limits", None
        return True, "", api_key_obj
    except APIKey.DoesNotExist:
        return False, "Invalid API key", None


@csrf_exempt
@require_POST
def chat_completions(request):
    """Handle OpenAI-compatible chat completions endpoint."""
    # Validate API key
    is_valid, error_msg, api_key_obj = validate_api_key(request)
    if not is_valid:
        return JsonResponse({"error": error_msg}, status=401)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    # Extract parameters
    model = data.get("model")
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    temperature = data.get("temperature")
    max_tokens = data.get("max_tokens")
    
    # Validate required parameters
    if not model:
        return JsonResponse({"error": "model is required"}, status=400)
    if not messages:
        return JsonResponse({"error": "messages is required"}, status=400)
    
    # Get provider for model
    model_providers = get_model_providers()
    provider_class = model_providers.get(model)
    
    if not provider_class:
        # Try each provider to see if they have this model
        providers = discover_providers()
        for provider_cls in providers.values():
            try:
                provider_instance = get_provider_instance(provider_cls)
                if model in provider_instance.get_model_list():
                    provider_class = provider_cls
                    break
            except Exception:
                continue
        
        if not provider_class:
            return JsonResponse({"error": f"Model {model} not found"}, status=400)
    
    # Get provider instance
    provider = get_provider_instance(provider_class)
    
    # Prepare kwargs for provider
    provider_kwargs = {
        "model": model,
        "top_p": data.get("top_p"),
        "stop": data.get("stop"),
        "presence_penalty": data.get("presence_penalty"),
        "frequency_penalty": data.get("frequency_penalty"),
        "logit_bias": data.get("logit_bias"),
        "user": data.get("user"),
    }
    # Remove None values
    provider_kwargs = {k: v for k, v in provider_kwargs.items() if v is not None}
    
    if stream:
        return handle_streaming_response(
            provider, messages, max_tokens, temperature, provider_kwargs, api_key_obj
        )
    else:
        return handle_non_streaming_response(
            provider, messages, max_tokens, temperature, provider_kwargs, api_key_obj, model
        )


def handle_non_streaming_response(provider, messages, max_tokens, temperature, kwargs, api_key_obj, model):
    """Handle non-streaming chat completion."""
    try:
        # Generate response
        result = provider.generate(
            prompt=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Update API key usage
        tokens_used = result["usage"].get("total_tokens", 0)
        if tokens_used and api_key_obj:
            api_key_obj.increment_usage(tokens_used)
        
        # Format response in OpenAI format
        response = {
            "id": f"chatcmpl-{api_key_obj.id if api_key_obj else 'local'}",
            "object": "chat.completion",
            "created": int(api_key_obj.created_at.timestamp() if api_key_obj else 0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["content"]
                },
                "finish_reason": result.get("finish_reason", "stop")
            }],
            "usage": {
                "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                "completion_tokens": result["usage"].get("completion_tokens", 0),
                "total_tokens": result["usage"].get("total_tokens", 0)
            }
        }
        
        return JsonResponse(response)
    
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


def handle_streaming_response(provider, messages, max_tokens, temperature, kwargs, api_key_obj):
    """Handle streaming chat completion with SSE."""
    
    def generate_sse_events():
        """Generate Server-Sent Events."""
        try:
            total_tokens = 0
            
            # Send initial event
            yield f"data: {json.dumps({'id': f'chatcmpl-{api_key_obj.id if api_key_obj else 'local'}', 'object': 'chat.completion.chunk', 'created': int(api_key_obj.created_at.timestamp() if api_key_obj else 0), 'model': kwargs.get('model', ''), 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # Stream response chunks
            for chunk in provider.generate_stream(
                prompt=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            ):
                # Estimate tokens (rough approximation)
                content = chunk.get("content", "")
                if content:
                    total_tokens += len(content.split()) * 1.3  # Rough token estimate
                
                # Format chunk in OpenAI format
                sse_chunk = {
                    "id": f"chatcmpl-{api_key_obj.id if api_key_obj else 'local'}",
                    "object": "chat.completion.chunk",
                    "created": int(api_key_obj.created_at.timestamp() if api_key_obj else 0),
                    "model": kwargs.get("model", ""),
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": content
                        },
                        "finish_reason": chunk.get("finish_reason")
                    }]
                }
                
                yield f"data: {json.dumps(sse_chunk)}\n\n"
            
            # Send final event
            yield "data: [DONE]\n\n"
            
            # Update API key usage after streaming completes
            if api_key_obj and total_tokens > 0:
                api_key_obj.increment_usage(int(total_tokens))
                
        except Exception as e:
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    response = StreamingHttpResponse(
        generate_sse_events(),
        content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
def models_list(request):
    """List available models in OpenAI format."""
    # Validate API key
    is_valid, error_msg, _ = validate_api_key(request)
    if not is_valid:
        return JsonResponse({"error": error_msg}, status=401)
    
    models = []
    providers = discover_providers()
    
    # Get models from each provider
    for provider_name, provider_class in providers.items():
        try:
            provider_instance = get_provider_instance(provider_class)
            provider_models = provider_instance.get_model_list()
            
            # Determine owner name from provider class name
            owner = provider_name.replace('Provider', '').lower()
            
            for model_id in provider_models:
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": owner
                })
        except Exception:
            # Skip providers that fail to instantiate or list models
            continue
    
    return JsonResponse({
        "object": "list",
        "data": models
    })