FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV HF_HOME=/app/backend/.hf_cache
ENV TRANSFORMERS_CACHE=/app/backend/.hf_cache/transformers

WORKDIR /app

COPY backend/requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY backend/ ./backend/
COPY Model/ ./Model/

EXPOSE 7860

CMD ["uvicorn","backend.main:app","--host","0.0.0.0","--port","7860"]