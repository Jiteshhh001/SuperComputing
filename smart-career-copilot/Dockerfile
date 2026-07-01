# Single-container build: React frontend + FastAPI backend (used for cloud deployment).
# Local development with docker-compose still uses backend/Dockerfile and frontend/Dockerfile.

FROM node:20-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app app/
COPY --from=frontend /fe/dist ./static

RUN mkdir -p data workspace uploads \
    && useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000 \
    FRONTEND_DIST=/app/static \
    DEBUG=false
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
