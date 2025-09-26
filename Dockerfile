# Dockerfile
FROM python:3.11-slim

# Speed up installs
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . ./

# Expose Flask port
EXPOSE 8000

# By default just run the API (assumes you've mounted or baked in a saved_model/)
ENV MODEL_DIR=/app/saved_model
CMD ["python", "app.py"]
