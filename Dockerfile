# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Additional serving dependencies
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    gunicorn==21.2.0

# Copy serving code
COPY serving/model_server.py .

# Set environment variables
ENV PORT=8080
ENV PROJECT_ID=ihg-mlops
ENV BUCKET_NAME=ihg-mlops
ENV MODEL_PATH=models/ensemble_model.pkl
ENV FEATURES_PATH=models/feature_names.pkl

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 900 model_server:app