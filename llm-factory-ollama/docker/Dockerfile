# All-in-one app container (CPU by default)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Copy backend and install
COPY backend/ /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy frontend into backend/public (no build step; vanilla JS)
COPY frontend/ /app/backend/public/

# Copy project scripts (optional)
COPY scripts/ /app/scripts/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
