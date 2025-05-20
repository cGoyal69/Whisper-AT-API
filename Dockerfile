FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg is required for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for tiktoken)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application
COPY app.py .

# Expose port
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]