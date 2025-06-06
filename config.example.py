"""
Snowflake Cost Management Tool - Configuration

This file contains configuration settings for connecting to Snowflake.
Copy this file to config.py and update the settings with your own values.

For security, credentials should be stored as environment variables.
"""

import os

# Snowflake connection settings
SNOWFLAKE_CONFIG = {
    'account': os.environ.get('SNOWFLAKE_ACCOUNT', 'your_account_identifier'),  # e.g., 'xy12345.us-east-1'
    'user': os.environ.get('SNOWFLAKE_USER', 'your_username'),
    'password': os.environ.get('SNOWFLAKE_PASSWORD', 'your_password'),  # Use environment variable for security
    'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE', 'your_warehouse'),
    'database': os.environ.get('SNOWFLAKE_DATABASE', 'your_database'),
    'schema': os.environ.get('SNOWFLAKE_SCHEMA', 'your_schema'),
    'role': os.environ.get('SNOWFLAKE_ROLE', 'your_role')  # Role should have access to ACCOUNT_USAGE views
}

# Data collection settings
DATA_COLLECTION = {
    'history_days': int(os.environ.get('SNOWFLAKE_HISTORY_DAYS', 30)),  # Number of days of history to collect
    'query_limit': int(os.environ.get('SNOWFLAKE_QUERY_LIMIT', 10000)),  # Maximum number of queries to collect
    'refresh_interval': int(os.environ.get('SNOWFLAKE_REFRESH_INTERVAL', 24))  # Hours between data refreshes
}

# Model settings
MODEL_SETTINGS = {
    'warehouse_model_path': os.environ.get('WAREHOUSE_MODEL_PATH', 'models/warehouse_sizing_model.joblib'),
    'query_model_path': os.environ.get('QUERY_MODEL_PATH', 'models/query_optimization_model.joblib'),
    'retrain_interval': int(os.environ.get('MODEL_RETRAIN_INTERVAL', 7))  # Days between model retraining
}

# Application settings
APP_SETTINGS = {
    'host': os.environ.get('APP_HOST', '0.0.0.0'),
    'port': int(os.environ.get('APP_PORT', 5000)),
    'debug': os.environ.get('APP_DEBUG', 'False').lower() == 'true',
    'log_level': os.environ.get('APP_LOG_LEVEL', 'INFO'),
    'secret_key': os.environ.get('APP_SECRET_KEY', 'generate_a_secure_random_key'),  # For session encryption
    'auth_required': os.environ.get('APP_AUTH_REQUIRED', 'True').lower() == 'true'
}

# Authentication settings
AUTH_SETTINGS = {
    'users': {
        # Default admin user (change in production)
        'admin': {
            'password': os.environ.get('ADMIN_PASSWORD', 'change_this_password'),
            'role': 'admin'
        }
    },
    # Additional users can be added in config.py
    # Example:
    # 'john': {
    #     'password': 'secure_password',
    #     'role': 'user'
    # }
}

# Security settings
SECURITY_SETTINGS = {
    'data_encryption_key': os.environ.get('DATA_ENCRYPTION_KEY', None),  # For encrypting stored data
    'data_retention_days': int(os.environ.get('DATA_RETENTION_DAYS', 90)),  # Days to keep collected data
    'session_timeout': int(os.environ.get('SESSION_TIMEOUT', 30)),  # Minutes until session expires
    'allowed_ips': os.environ.get('ALLOWED_IPS', '').split(',') if os.environ.get('ALLOWED_IPS') else []
}

