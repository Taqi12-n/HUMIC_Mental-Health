FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/app/backend/.hf_cache
ENV TRANSFORMERS_CACHE=/app/backend/.hf_cache/transformers
ENV HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY Model/ ./Model/

EXPOSE 7860

CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","7860"]