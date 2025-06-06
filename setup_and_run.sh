#!/bin/bash

# Snowflake Cost Management Tool - Setup and Run Script

echo "Setting up Snowflake Cost Management Tool..."

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories if they don't exist
mkdir -p data models templates

# Test models
echo "Testing models..."
python test_models.py

# Run the application
echo "Starting web application on http://localhost:5000"
python app.py
