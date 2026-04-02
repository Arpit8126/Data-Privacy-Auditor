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

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. Copy requirement files first for better caching
COPY requirements.txt pyproject.toml ./

# 2. Install base requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire project source
# (Ensure your folder structure is preserved)
COPY . .

# 4. Install the current project in editable mode or as a package
# This is CRITICAL to satisfy the "multi-mode deployment" validator check
RUN pip install --no-cache-dir .

# Expose the FastAPI server port
EXPOSE 8000

# Health-check (using the health endpoint we verified earlier)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the server using the entry point defined in pyproject.toml
# Or stick to uvicorn but ensure it points to the correct app location
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]