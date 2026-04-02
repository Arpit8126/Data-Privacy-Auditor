# ─────────────────────────────────────────────────────────────
# Data Privacy & Integrity Auditor — Docker Image
# Target: 2 vCPU / 8 GB RAM
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="Arpit Pandey"
LABEL description="OpenEnv RL Environment — Data Privacy & Integrity Auditor"

# Prevent Python from writing .pyc and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY models.py main.py inference.py openenv.yaml ./
COPY dataset/ ./dataset/
COPY server/ ./server/

# Expose the FastAPI server port
EXPOSE 8000

# Health-check (every 30 s, 5 s timeout, 3 retries)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the server — 2 workers for 2-vCPU machines
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
