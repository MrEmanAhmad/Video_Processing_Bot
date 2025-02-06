# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Install system dependencies and FFmpeg with all necessary codecs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    libavformat-extra \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libass-dev \
    libfreetype6-dev \
    libfontconfig1 \
    libfribidi-dev \
    libharfbuzz-dev \
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
RUN mkdir -p /tmp/test_outputs && \
    mkdir -p /tmp/processed_videos && \
    chmod -R 777 /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TEMP=/tmp
ENV TMPDIR=/tmp

# Set proxy bypass for metadata server
ENV no_proxy=169.254.169.254,metadata,metadata.google.internal
ENV NO_PROXY=169.254.169.254,metadata,metadata.google.internal

# Create directory for Google Cloud credentials
RUN mkdir -p /app/credentials && \
    chmod 777 /app/credentials

# Set FFmpeg environment variables for optimal performance
ENV FFREPORT=level=32
ENV FFMPEG_THREADS=auto

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8080}')"

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Set proxy bypass for metadata server\n\
export no_proxy=169.254.169.254,metadata,metadata.google.internal\n\
export NO_PROXY=169.254.169.254,metadata,metadata.google.internal\n\
\n\
# Handle credentials\n\
if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then\n\
    echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /app/credentials/google_credentials.json\n\
    export GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/google_credentials.json\n\
elif [ ! -z "$GOOGLE_CREDENTIALS" ]; then\n\
    echo "$GOOGLE_CREDENTIALS" > /app/credentials/google_credentials.json\n\
    export GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/google_credentials.json\n\
fi\n\
\n\
# Verify credentials file\n\
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then\n\
    echo "Credentials file created successfully at $GOOGLE_APPLICATION_CREDENTIALS"\n\
    chmod 600 "$GOOGLE_APPLICATION_CREDENTIALS"\n\
else\n\
    echo "Error: Credentials file not created. Please check environment variables."\n\
    exit 1\n\
fi\n\
\n\
python main.py' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Run the bot using entrypoint script
CMD ["/app/entrypoint.sh"] 