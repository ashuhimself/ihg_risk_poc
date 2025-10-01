#!/bin/bash

# Fraud Detection MLOps Setup Script for GCP
# This script sets up and runs the fraud detection pipeline on Vertex AI

set -e  # Exit on error

# Configuration
PROJECT_ID="ihg-mlops"
LOCATION="us-central1"
BUCKET_NAME="ihg-mlops"
DATASET_ID="ihg_training_data"
TABLE_ID="booking"
SERVICE_ACCOUNT=""  # Add your service account if needed

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Fraud Detection MLOps Pipeline Setup${NC}"
echo -e "${GREEN}=====================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

if ! command_exists gcloud; then
    echo -e "${RED}Error: gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.8+.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"

# Set up GCP project
echo -e "\n${YELLOW}Setting up GCP project...${NC}"
gcloud config set project ${PROJECT_ID}

# Enable required APIs
echo -e "\n${YELLOW}Enabling required APIs...${NC}"
gcloud services enable \
    aiplatform.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    bigquery.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com

echo -e "${GREEN}✓ APIs enabled${NC}"

# Create GCS bucket if it doesn't exist
echo -e "\n${YELLOW}Setting up GCS bucket...${NC}"
if ! gsutil ls -b gs://${BUCKET_NAME} &>/dev/null; then
    gsutil mb -l ${LOCATION} gs://${BUCKET_NAME}
    echo -e "${GREEN}✓ Created bucket: gs://${BUCKET_NAME}${NC}"
else
    echo -e "${GREEN}✓ Bucket already exists: gs://${BUCKET_NAME}${NC}"
fi

# Create pipeline directories in GCS
gsutil -m mkdir -p \
    gs://${BUCKET_NAME}/pipeline-root \
    gs://${BUCKET_NAME}/models \
    gs://${BUCKET_NAME}/evaluations \
    gs://${BUCKET_NAME}/predictions

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Function to run training pipeline
run_training_pipeline() {
    echo -e "\n${YELLOW}Running training pipeline...${NC}"
    
    cd pipeline
    python training_pipeline.py \
        --project-id ${PROJECT_ID} \
        --location ${LOCATION} \
        --pipeline-root gs://${BUCKET_NAME}/pipeline-root
    
    cd ..
    echo -e "${GREEN}✓ Training pipeline submitted${NC}"
}

# Function to build and push Docker image
build_docker_image() {
    echo -e "\n${YELLOW}Building Docker image for model serving...${NC}"
    
    IMAGE_URI="gcr.io/${PROJECT_ID}/fraud-detection-model:latest"
    
    # Build image
    docker build -t ${IMAGE_URI} .
    
    # Push to Container Registry
    docker push ${IMAGE_URI}
    
    echo -e "${GREEN}✓ Docker image built and pushed: ${IMAGE_URI}${NC}"
}

# Function to deploy model
deploy_model() {
    echo -e "\n${YELLOW}Deploying model to Vertex AI Endpoint...${NC}"
    
    python deploy_model.py \
        --project-id ${PROJECT_ID} \
        --location ${LOCATION} \
        --model-uri gs://${BUCKET_NAME}/models \
        --container-image gcr.io/${PROJECT_ID}/fraud-detection-model:latest \
        --test
    
    echo -e "${GREEN}✓ Model deployed successfully${NC}"
}

# Function to run batch prediction
run_batch_prediction() {
    echo -e "\n${YELLOW}Running batch prediction...${NC}"
    
    # This would be integrated into a pipeline or run separately
    echo "Batch prediction can be run through:"
    echo "1. Vertex AI Batch Prediction UI"
    echo "2. Using the batch_prediction component in a pipeline"
    echo "3. Scheduled as a recurring pipeline"
}

# Main menu
echo -e "\n${YELLOW}What would you like to do?${NC}"
echo "1. Run training pipeline only"
echo "2. Build Docker image for serving"
echo "3. Deploy model to endpoint"
echo "4. Run complete MLOps pipeline (training + deployment)"
echo "5. Run batch prediction"
echo "6. Compile pipeline only"
echo "7. Exit"

read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        run_training_pipeline
        ;;
    2)
        build_docker_image
        ;;
    3)
        deploy_model
        ;;
    4)
        run_training_pipeline
        echo -e "\n${YELLOW}Waiting for training to complete...${NC}"
        echo "Check pipeline status at: https://console.cloud.google.com/vertex-ai/pipelines"
        read -p "Press Enter when training is complete to continue with deployment..."
        build_docker_image
        deploy_model
        ;;
    5)
        run_batch_prediction
        ;;
    6)
        cd pipeline
        python training_pipeline.py --compile-only
        cd ..
        echo -e "${GREEN}✓ Pipeline compiled to fraud_detection_pipeline.json${NC}"
        ;;
    7)
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}=====================================${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Monitor pipeline: https://console.cloud.google.com/vertex-ai/pipelines"
echo "2. View models: https://console.cloud.google.com/vertex-ai/models"
echo "3. Check endpoints: https://console.cloud.google.com/vertex-ai/endpoints"
echo "4. Monitor costs: https://console.cloud.google.com/billing"

echo -e "\n${YELLOW}Useful commands:${NC}"
echo "- Check pipeline status: gcloud ai custom-jobs list --region=${LOCATION}"
echo "- View logs: gcloud logging read 'resource.type=ml_job'"
echo "- Test endpoint: curl -X POST [ENDPOINT_URL] -d @test_data.json"