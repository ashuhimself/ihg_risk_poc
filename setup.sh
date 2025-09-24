#!/bin/bash

# IHG Risk POC Setup Script
# This script sets up the development environment for the MLOps platform

set -e  # Exit on any error

echo "🚀 Setting up IHG Risk POC MLOps Platform..."

# Check if Python 3.9+ is installed
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Setup Google Cloud SDK (if not already installed)
if ! command -v gcloud &> /dev/null; then
    echo "⚠️  Google Cloud SDK not found. Please install it from: https://cloud.google.com/sdk/docs/install"
else
    echo "✅ Google Cloud SDK is installed"
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p artifacts
mkdir -p models
mkdir -p data

# Setup pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

# Setup React frontend
echo "⚛️  Setting up React frontend..."
cd portal/frontend
if command -v npm &> /dev/null; then
    npm install
    echo "✅ Frontend dependencies installed"
else
    echo "⚠️  npm not found. Please install Node.js to set up the frontend."
fi
cd ../..

# Create sample configuration files
echo "⚙️  Creating sample configuration files..."

# Sample service account key structure
cat > service-account-key.json.template << EOF
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "your-private-key-id",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nYour-Private-Key\\n-----END PRIVATE KEY-----\\n",
  "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "your-client-id",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com"
}
EOF

# Environment variables template
cat > .env.template << EOF
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json

# BigQuery Configuration
BQ_DATASET_ID=ihg_risk_dataset
BQ_TABLE_ID=risk_data

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-api-key

# Frontend Configuration
REACT_APP_API_BASE_URL=http://localhost:8000
EOF

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Copy your GCP service account key to 'service-account-key.json'"
echo "2. Copy '.env.template' to '.env' and update with your configuration"
echo "3. Update 'pipelines/configs/pipeline_config.yaml' with your project details"
echo "4. Run 'source venv/bin/activate' to activate the virtual environment"
echo "5. Run 'uvicorn portal.backend.main:app --reload' to start the backend"
echo "6. Run 'cd portal/frontend && npm start' to start the frontend"
echo ""
echo "📖 See README.md for detailed documentation"