FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -r requirements.txt \
    streamlit \
    fastapi \
    uvicorn \
    selenium \
    aws-sam-cli \
    awscli

# Create necessary directories
RUN mkdir -p /data /root/.streamlit

# Copy project files
COPY . .

# Expose ports
EXPOSE 8000 8501 5678

# Default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]