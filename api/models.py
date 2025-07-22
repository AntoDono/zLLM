from django.db import models
import secrets
from datetime import timedelta
from django.utils import timezone

api_key_length = 32

def create_api_key():
    return f"zLLM-{secrets.token_hex(api_key_length - 5)}"

def get_tokens_usage_reset_time():
    return timezone.now() + timedelta(days=30)

def get_rpm_reset_time():
    return timezone.now() + timedelta(minutes=1)

# Create your models here.
class APIKey(models.Model):
    key = models.CharField(max_length=api_key_length, unique=True, default=create_api_key)
    name = models.CharField(max_length=100, help_text="Human-readable name for this API key")
    notes = models.TextField(blank=True, help_text="Notes about this API key")
    
    tokens_used = models.IntegerField(default=0)
    tokens_limit = models.IntegerField(default=0)
    tokens_usage_reset_duration_days = models.IntegerField(default=30)
    tokens_usage_reset_at = models.DateTimeField(default=get_tokens_usage_reset_time)
    
    requests_sent= models.IntegerField(default=0)
    rpm = models.IntegerField(default=0) # requests per minute
    max_rpm = models.IntegerField(default=1000)
    rpm_reset_at = models.DateTimeField(default=get_rpm_reset_time)
    
    is_active = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.name} - {self.key}"
    
    def is_valid(self):
        
        if timezone.now() >= self.rpm_reset_at:
            self.requests_sent = 0
            self.rpm_reset_at = get_rpm_reset_time()
            
        if timezone.now() >= self.tokens_usage_reset_at:
            self.tokens_used = 0
            self.tokens_usage_reset_at = get_tokens_usage_reset_time()
            
        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < timezone.now(): # If no expire date, it's valid
            return False
        if self.tokens_limit > 0 and self.tokens_used >= self.tokens_limit:
            return False
        if self.rpm > 0 and self.requests_sent >= self.rpm: # If no rpm limit, it's valid
            return False
        return True
    
    def can_make_request(self):
        """Check if the API key can make a request (alias for is_valid)."""
        return self.is_valid()
    
    def increment_usage(self, tokens):
        self.tokens_used += tokens
        self.requests_sent += 1
        self.save()
        
class LLMModel(models.Model):
    allowed_providers = [
        "ollama",
        "vllm",
        "groq"
    ]
    
    model_name = models.CharField(max_length=256, unique=True)
    provider = models.CharField(max_length=50, choices=[(p, p) for p in allowed_providers])
    cost_per_input_token = models.DecimalField(max_digits=10, decimal_places=8)
    cost_per_output_token = models.DecimalField(max_digits=10, decimal_places=8)
    
    device = models.CharField(max_length=100)
    
    max_context_length = models.IntegerField()
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.model_name} - {self.provider}"
    
class APIRequest(models.Model):
    api_key = models.ForeignKey(APIKey, on_delete=models.CASCADE)
    model = models.ForeignKey(LLMModel, on_delete=models.CASCADE)
    
    input_text = models.TextField()
    output_text = models.TextField()
    time_taken = models.IntegerField()
    
    ip_address = models.GenericIPAddressField()
    timestamp = models.DateTimeField(auto_now_add=True)
    cost = models.DecimalField(max_digits=10, decimal_places=6)
    
    def __str__(self):
        return f"{self.api_key.name} - {self.model.model_name} - {self.timestamp}"
    