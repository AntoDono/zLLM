# Generated by Django 5.2.4 on 2025-07-08 13:59

import api.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0004_rename_token_limit_reset_days_apikey_tokens_usage_reset_duration_days'),
    ]

    operations = [
        migrations.AddField(
            model_name='apikey',
            name='tokens_usage_reset_at',
            field=models.DateTimeField(default=api.models.get_tokens_usage_reset_time),
        ),
    ]
