#!/bin/bash

# Minimal setup script for low disk space environments

# Clean up first
echo "Cleaning up disk space..."
sudo apt-get clean
sudo apt-get autoremove -y
pip cache purge

echo "Setting up Audio Classification API (minimal version)..."

# Install only the essential packages
echo "Installing ffmpeg (required for audio processing)..."
sudo apt-get update -q
sudo apt-get install -y ffmpeg -q --no-install-recommends

echo "Installing minimal Python dependencies..."
pip install fastapi uvicorn python-multipart --no-cache-dir
pip install tqdm numpy --no-cache-dir

# Install Whisper-AT with minimal dependencies
echo "Installing Whisper-AT..."
pip install whisper-at --no-cache-dir

echo "Setup complete!"
echo "To start the API server, run: uvicorn minimal_app:app --host 0.0.0.0 --port 8080"