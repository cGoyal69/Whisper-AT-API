# Base image: slim variant for minimal size
FROM python:3.10-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="$HOME/.cargo/bin:$PATH"

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        build-essential \
        gcc \
        libsndfile1 \
        git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Rust (required for building tiktoken)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Copy requirements (excluding whisper-at)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# Manually install whisper-at without triton
RUN pip install --no-cache-dir --no-deps whisper-at

# Copy your app source
COPY . /app
WORKDIR /app

# Default command
CMD ["uvicorn", "minimal_app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]