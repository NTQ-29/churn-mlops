# ==========================================
# STAGE 1: Build & Dependency Compilation
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Prevent Python from writing .pyc files and buffer streams
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install minimal system utilities required to compile heavy wheel binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Generate optimized application dependencies local cache
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ==========================================
# STAGE 2: Ultra-Lightweight Final Runtime
# ==========================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Re-establish production environment configurations
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Pull down ONLY the compiled libraries from Stage 1, leaving compilation bloat behind
COPY --from=builder /root/.local /root/.local
COPY ./src ./src

# Ensure the local user binaries folder sits directly in Python's execution lookup path
ENV PATH=/root/.local/bin:$PATH

# Expose the API delivery communication channel port
EXPOSE 8080

# Spin up high-concurrency production ASGI server to handle incoming prediction traffic
CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}"]
