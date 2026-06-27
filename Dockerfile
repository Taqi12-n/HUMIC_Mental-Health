FROM python:3.12-slim

# Install system dependencies for audio processing and libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and model files
COPY backend/ ./backend/
COPY Model/ ./Model/

EXPOSE 8000

# Run uvicorn server pointing to the backend main module
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
