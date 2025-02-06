# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for temporary files and outputs
RUN mkdir -p test_outputs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Create Google Cloud credentials file from environment variable
RUN echo $GOOGLE_APPLICATION_CREDENTIALS_JSON > /app/google_credentials.json

# Run the bot
CMD ["python", "main.py"] 