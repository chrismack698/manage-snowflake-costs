#!/bin/bash

# Snowflake Cost Management Tool - Startup Script

echo "Starting Snowflake Cost Management Tool..."

# Check if Python dependencies are installed
echo "Checking dependencies..."
pip install -r requirements.txt

# Create directories if they don't exist
mkdir -p templates

# Run the application
echo "Starting web application on http://localhost:5000"
python app.py
