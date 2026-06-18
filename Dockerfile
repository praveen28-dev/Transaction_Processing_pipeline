FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output for logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command — overridden per service in docker-compose
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
