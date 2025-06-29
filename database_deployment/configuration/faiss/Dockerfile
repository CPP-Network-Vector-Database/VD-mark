# Base image
FROM python:3.12-slim

# Settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy code
COPY . .

# Expose port & run with Gunicorn
EXPOSE 7000
CMD ["python3", "app.py"]
