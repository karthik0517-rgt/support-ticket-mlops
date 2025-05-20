FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Env setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache

# Install system deps
RUN apt-get update && apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    rm -rf ~/.cache/pip

# Copy app code
COPY . .

# Optional preload for cold start performance
# RUN python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='facebook/bart-large-mnli')"

# Expose FastAPI port
EXPOSE 8000

# Run app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
