# Use a minimal Python image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables for better Python behavior
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Default inference environment (override via docker run -e or HF Space secrets)
ENV API_BASE_URL=https://api-inference.huggingface.co/v1
ENV MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose the port used by the API server (HF Spaces default)
EXPOSE 7860

# Health check against the API server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/health || exit 1

# Default: run the API server
# To run inference manually: docker run <image> python inference.py
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
