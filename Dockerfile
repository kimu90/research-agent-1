FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git wget unzip chromium chromium-driver \
    sqlite3 build-essential \
    libblas-dev liblapack-dev \
    gfortran libatlas-base-dev \
    netcat-traditional \
    cython3 libgomp1 \
    postgresql-client libpq-dev \
    python3-dev

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt