# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install system dependencies and Chrome
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    gnupg \
    git \
    && wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Set display port to avoid crash
ENV DISPLAY=:99

# Set working directory
WORKDIR /app

# Copy the entire application
COPY . .

# Install Python dependencies with HTTP/2 support
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "httpx[http2]" "python-telegram-bot[http2]"

# Ensure proper package structure
RUN python -m pip install -e .

# Create necessary directories and set permissions
RUN mkdir -p credentials && \
    mkdir -p analysis_temp && \
    mkdir -p /root/.config/google-chrome && \
    chmod -R 777 credentials analysis_temp /root/.config/google-chrome && \
    chmod -R 755 pipeline && \
    # Verify package installation
    python -c "from pipeline.Step_4_generate_commentary import CommentaryStyle; print('Package verified successfully')"

# Command to run the bot
CMD ["python", "-m", "bot"] 