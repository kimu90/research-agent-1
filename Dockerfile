FROM python:3.12-slim

WORKDIR /app

# Install system dependencies in a single layer to reduce image size
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    chromium \
    chromium-driver \
    sqlite3 \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    USER_AGENT="ResearchAgent/1.0" \
    PYTHONPATH=/app \
    DB_PATH=/data/content.db

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies including Streamlit
RUN pip install --no-cache-dir -r requirements.txt \
    streamlit \
    selenium \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl

# Create directory for database
RUN mkdir -p /data

# Copy project files
COPY . .

# Expose ports for both Streamlit and the application
EXPOSE 8000 8501