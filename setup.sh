#!/bin/bash

echo "Cleaning up disk space..."
sudo apt-get clean
sudo apt-get autoremove -y
pip cache purge

echo "Installing ffmpeg..."
sudo apt-get update -q
sudo apt-get install -y ffmpeg -q --no-install-recommends

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel --no-cache-dir

echo "Installing Rust (for tiktoken)..."
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt --no-cache-dir

echo "Installing whisper-at without dependencies..."
pip install whisper-at --no-deps --no-cache-dir

echo "✅ Setup complete!"