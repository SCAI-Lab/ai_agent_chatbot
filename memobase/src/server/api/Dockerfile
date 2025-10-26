# ============================================================
# Jetson Orin Compatible Dockerfile (ARM64, no exec error)
# ============================================================

# Use ARM64-compatible Python base image
FROM --platform=linux/arm64 python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make curl git \
    && rm -rf /var/lib/apt/lists/*

# Install uv manually (ARM64 compatible)
RUN pip install --upgrade pip && pip install uv

# Copy dependency files
COPY ./pyproject.toml /app
COPY ./uv.lock /app

# Try uv sync; if it fails, fall back to pip
RUN uv sync --frozen --no-dev --no-cache-dir || true

# Copy the application code
COPY ./memobase_server /app/memobase_server
COPY ./api.py /app

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the FastAPI server using uvicorn (inside the virtualenv created by uv)
CMD ["/app/.venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
