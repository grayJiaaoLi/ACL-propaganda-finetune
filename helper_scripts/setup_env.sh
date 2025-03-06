#!/bin/bash
# setup_env.sh - Script to set up the environment for Llama 3 (8B) training with Unsloth
# This script installs all required dependencies and prepares the environment

echo "Setting up environment for Llama 3 (8B) training with Unsloth..."

# Create a virtual environment (optional)
echo "Creating virtual environment..."
python -m venv llama3_env
source llama3_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Unsloth
echo "Installing Unsloth..."
pip install unsloth

# Create necessary directories if they don't exist
echo "Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p models/final_model
mkdir -p logs

echo "Environment setup complete!"
echo "Activate the environment with: source llama3_env/bin/activate" 