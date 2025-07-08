from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import APIKey, LLMModel, APIRequest

# Custom Admin for APIKey
@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'key_preview', 'is_active', 'tokens_used', 'tokens_limit', 
        'requests_sent', 'max_rpm', 'created_at', 'expires_at', 'status_indicator'
    ]
    list_filter = [
        'is_active', 'created_at', 'expires_at'
    ]
    search_fields = ['name', 'key', 'notes']
    readonly_fields = [
        'key', 'tokens_used', 'requests_sent', 'rpm', 'rpm_reset_at', 
        'created_at', 'request_count', 'usage_stats'
    ]
    ordering = ['-created_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'notes', 'key')
        }),
        ('Status & Activation', {
            'fields': ('is_active', 'expires_at')
        }),
        ('Usage Limits', {
            'fields': ('tokens_limit', 'tokens_usage_reset_duration_days', 'max_rpm')
        }),
        ('Usage Statistics (Read-Only)', {
            'fields': ('tokens_used', 'requests_sent', 'rpm', 'rpm_reset_at', 'created_at'),
            'classes': ('collapse',)
        }),
        ('Analytics', {
            'fields': ('request_count', 'usage_stats'),
            'classes': ('collapse',)
        })
    )
    
    def key_preview(self, obj):
        """Show abbreviated key for security"""
        return f"{obj.key[:15]}...{obj.key[-5:]}" if obj.key else ""
    key_preview.short_description = "API Key"
    
    def status_indicator(self, obj):
        """Visual status indicator"""
        if obj.is_valid():
            return format_html(
                '<span style="color: green; font-weight: bold;">✓ Valid</span>'
            )
        else:
            return format_html(
                '<span style="color: red; font-weight: bold;">✗ Invalid</span>'
            )
    status_indicator.short_description = "Status"
    
    def request_count(self, obj):
        """Show total API requests made with this key"""
        count = APIRequest.objects.filter(api_key=obj).count()
        if count > 0:
            url = reverse('admin:api_apirequest_changelist') + f'?api_key__id__exact={obj.id}'
            return format_html('<a href="{}">{} requests</a>', url, count)
        return "0 requests"
    request_count.short_description = "Total Requests"
    
    def usage_stats(self, obj):
        """Show usage statistics"""
        if obj.tokens_limit > 0:
            percentage = (obj.tokens_used / obj.tokens_limit) * 100
            color = "red" if percentage > 80 else "orange" if percentage > 60 else "green"
            return format_html(
                '<div style="color: {};">{:.1f}% tokens used</div>', 
                color, percentage
            )
        return "No limit set"
    usage_stats.short_description = "Usage %"
    


# Custom Admin for LLMModel  
@admin.register(LLMModel)
class LLMModelAdmin(admin.ModelAdmin):
    list_display = [
        'model_name', 'provider', 'is_active', 'cost_per_input_token', 
        'cost_per_output_token', 'max_context_length', 'device', 'request_count'
    ]
    list_filter = ['provider', 'is_active', 'device']
    search_fields = ['model_name', 'device']
    readonly_fields = ['request_count', 'total_cost', 'avg_request_time']
    ordering = ['provider', 'model_name']
    
    fieldsets = (
        ('Model Information', {
            'fields': ('model_name', 'provider', 'device')
        }),
        ('Configuration', {
            'fields': ('max_context_length', 'is_active')
        }),
        ('Pricing', {
            'fields': ('cost_per_input_token', 'cost_per_output_token')
        }),
        ('Statistics (Read-Only)', {
            'fields': ('request_count', 'total_cost', 'avg_request_time'),
            'classes': ('collapse',)
        })
    )
    
    def request_count(self, obj):
        """Show number of requests for this model"""
        count = APIRequest.objects.filter(model=obj).count()
        if count > 0:
            url = reverse('admin:api_apirequest_changelist') + f'?model__id__exact={obj.id}'
            return format_html('<a href="{}">{} requests</a>', url, count)
        return "0 requests"
    request_count.short_description = "Requests"
    
    def total_cost(self, obj):
        """Calculate total cost for this model"""
        total = sum(APIRequest.objects.filter(model=obj).values_list('cost', flat=True))
        return f"${total:.4f}"
    total_cost.short_description = "Total Cost"
    
    def avg_request_time(self, obj):
        """Calculate average request time"""
        requests = APIRequest.objects.filter(model=obj)
        if requests.exists():
            avg_time = sum(r.time_taken for r in requests) / requests.count()
            return f"{avg_time:.2f}ms"
        return "No data"
    avg_request_time.short_description = "Avg Time"

# Custom Admin for APIRequest
@admin.register(APIRequest)
class APIRequestAdmin(admin.ModelAdmin):
    list_display = [
        'timestamp', 'api_key_name', 'model_name', 'cost', 'time_taken', 
        'ip_address', 'input_preview', 'output_preview'
    ]
    list_filter = [
        'timestamp', 'model__provider', 'model', 'api_key'
    ]
    search_fields = [
        'api_key__name', 'model__model_name', 'ip_address', 
        'input_text', 'output_text'
    ]
    readonly_fields = [
        'api_key', 'model', 'input_text', 'output_text', 'time_taken',
        'ip_address', 'timestamp', 'cost', 'input_length', 'output_length'
    ]
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'
    
    fieldsets = (
        ('Request Information', {
            'fields': ('timestamp', 'api_key', 'model', 'ip_address')
        }),
        ('Performance', {
            'fields': ('time_taken', 'cost')
        }),
        ('Content', {
            'fields': ('input_text', 'output_text', 'input_length', 'output_length')
        })
    )
    
    def has_add_permission(self, request):
        """Disable adding new API requests manually"""
        return False
    
    def has_change_permission(self, request, obj=None):
        """Make API requests read-only"""
        return True  # Allow viewing but fields are readonly
    
    def api_key_name(self, obj):
        """Show API key name instead of full key"""
        return obj.api_key.name
    api_key_name.short_description = "API Key"
    api_key_name.admin_order_field = 'api_key__name'
    
    def model_name(self, obj):
        """Show model name"""
        return obj.model.model_name
    model_name.short_description = "Model"
    model_name.admin_order_field = 'model__model_name'
    
    def input_preview(self, obj):
        """Show truncated input text"""
        text = obj.input_text[:50] + "..." if len(obj.input_text) > 50 else obj.input_text
        return text
    input_preview.short_description = "Input Preview"
    
    def output_preview(self, obj):
        """Show truncated output text"""
        text = obj.output_text[:50] + "..." if len(obj.output_text) > 50 else obj.output_text
        return text
    output_preview.short_description = "Output Preview"
    
    def input_length(self, obj):
        """Show input text length"""
        return f"{len(obj.input_text)} chars"
    input_length.short_description = "Input Length"
    
    def output_length(self, obj):
        """Show output text length"""
        return f"{len(obj.output_text)} chars"
    output_length.short_description = "Output Length"

