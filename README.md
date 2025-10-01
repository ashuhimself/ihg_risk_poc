# Fraud Detection MLOps Pipeline on GCP

A production-ready MLOps pipeline for fraud detection using Vertex AI, BigQuery, and GCS.

## 🎯 Overview

This pipeline transforms your existing fraud detection notebook into a scalable, production-ready MLOps solution on Google Cloud Platform. It includes:

- **Data ingestion** from BigQuery
- **Automated model training** with XGBoost, LightGBM, and Logistic Regression ensemble
- **Feature selection** based on importance scores
- **Model evaluation** with business metrics
- **Model deployment** to Vertex AI endpoints
- **Batch prediction** capabilities
- **Real-time serving** with REST API

## 📋 Prerequisites

1. **GCP Project**: `ihg-mlops`
2. **BigQuery Table**: `ihg-mlops.ihg_training_data.booking`
3. **GCS Bucket**: `ihg-mlops`
4. **Required APIs**:
   - Vertex AI API
   - BigQuery API
   - Cloud Storage API
   - Container Registry API

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo>
cd fraud-detection-mlops

# Make setup script executable
chmod +x setup_and_run.sh

# Run setup
./setup_and_run.sh
```

### 2. Configure Service Account (if needed)

```bash
# Create service account
gcloud iam service-accounts create fraud-detection-sa \
    --display-name="Fraud Detection Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding ihg-mlops \
    --member="serviceAccount:fraud-detection-sa@ihg-mlops.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding ihg-mlops \
    --member="serviceAccount:fraud-detection-sa@ihg-mlops.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding ihg-mlops \
    --member="serviceAccount:fraud-detection-sa@ihg-mlops.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

## 📁 Project Structure

```
fraud-detection-mlops/
├── components/               # Pipeline components
│   ├── data_preprocessing.py    # Data loading and splitting
│   ├── model_training.py        # Model training logic
│   ├── model_evaluation.py      # Evaluation metrics
│   └── batch_prediction.py      # Batch inference
├── pipeline/                 # Pipeline orchestration
│   └── training_pipeline.py     # Main pipeline definition
├── serving/                  # Model serving
│   └── model_server.py          # Flask API server
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container for serving
├── deploy_model.py          # Deployment script
├── setup_and_run.sh         # Setup automation
└── README.md               # Documentation
```

## 🔄 Pipeline Workflow

### Training Pipeline

1. **Data Preprocessing**
   - Reads data from BigQuery table
   - Handles missing values
   - Splits into train/test sets
   - Logs data statistics

2. **Model Training**
   - Trains XGBoost, LightGBM, and Logistic Regression
   - Calculates feature importance
   - Selects top features based on threshold
   - Creates voting ensemble
   - Saves model to GCS

3. **Model Evaluation**
   - Calculates performance metrics (ROC-AUC, PR-AUC)
   - Generates confusion matrix
   - Computes business impact metrics
   - Saves evaluation report

### Deployment Pipeline

1. **Model Registry**
   - Uploads model to Vertex AI Model Registry
   - Versions models automatically

2. **Endpoint Creation**
   - Creates scalable serving endpoint
   - Auto-scaling based on traffic

3. **Model Deployment**
   - Deploys model with specified resources
   - Configures traffic splitting

## 💻 Running the Pipeline

### Training Pipeline

```bash
# Run training pipeline
python pipeline/training_pipeline.py \
    --project-id ihg-mlops \
    --location us-central1 \
    --pipeline-root gs://ihg-mlops/pipeline-root
```

### Batch Prediction

```python
from components.batch_prediction import batch_predict

# Run batch predictions
batch_predict(
    project_id="ihg-mlops",
    bucket_name="ihg-mlops",
    input_table="ihg_training_data.booking_new",
    output_table="ihg_training_data.predictions",
    model_path="models/ensemble_model.pkl",
    features_path="models/feature_names.pkl"
)
```

### Model Deployment

```bash
# Build Docker image
docker build -t gcr.io/ihg-mlops/fraud-detection-model:latest .
docker push gcr.io/ihg-mlops/fraud-detection-model:latest

# Deploy model
python deploy_model.py \
    --project-id ihg-mlops \
    --location us-central1 \
    --model-uri gs://ihg-mlops/models \
    --test
```

## 🔍 Testing the Endpoint

### REST API Call

```bash
# Get endpoint URL
ENDPOINT_URL="https://us-central1-aiplatform.googleapis.com/v1/projects/ihg-mlops/locations/us-central1/endpoints/YOUR_ENDPOINT_ID:predict"

# Test prediction
curl -X POST $ENDPOINT_URL \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "feature_1": 0.5,
        "feature_2": 1.0,
        ...
      }
    ]
  }'
```

### Python Client

```python
from google.cloud import aiplatform

# Initialize
aiplatform.init(project="ihg-mlops", location="us-central1")

# Get endpoint
endpoint = aiplatform.Endpoint("YOUR_ENDPOINT_ID")

# Make prediction
test_instance = {
    "feature_1": 0.5,
    "feature_2": 1.0,
    # ... add all required features
}

prediction = endpoint.predict(instances=[test_instance])
print(prediction.predictions)
```

## 📊 Monitoring

### Pipeline Monitoring
- View pipelines: https://console.cloud.google.com/vertex-ai/pipelines
- Check logs: `gcloud logging read 'resource.type=ml_job'`

### Model Monitoring
- Model metrics: https://console.cloud.google.com/vertex-ai/models
- Endpoint health: https://console.cloud.google.com/vertex-ai/endpoints
- Prediction logs: Cloud Logging

### Cost Monitoring
- Billing dashboard: https://console.cloud.google.com/billing
- Set up budget alerts for cost control

## 🔧 Configuration

### Pipeline Parameters

Edit `pipeline/training_pipeline.py` to modify:

```python
parameter_values={
    "project_id": "ihg-mlops",
    "dataset_id": "ihg_training_data",
    "table_id": "booking",
    "bucket_name": "ihg-mlops",
    "test_size": 0.3,              # Test split ratio
    "random_state": 42,             # Random seed
    "importance_threshold": 0.05,   # Feature selection threshold
    "model_name": "fraud_detection_ensemble"
}
```

### Serving Configuration

Edit `deploy_model.py` to modify:

```python
# Machine type
machine_type="n1-standard-4"

# Scaling
min_replica_count=1
max_replica_count=3

# Add GPU if needed
accelerator_type="NVIDIA_TESLA_T4"
accelerator_count=1
```

## 🐛 Troubleshooting

### Common Issues

1. **BigQuery Permission Error**
   ```bash
   # Grant BigQuery access
   gcloud projects add-iam-policy-binding ihg-mlops \
       --member="user:YOUR_EMAIL" \
       --role="roles/bigquery.dataViewer"
   ```

2. **Pipeline Fails to Start**
   ```bash
   # Check if APIs are enabled
   gcloud services list --enabled
   
   # Enable missing APIs
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Docker Build Fails**
   ```bash
   # Authenticate Docker
   gcloud auth configure-docker
   ```

4. **Endpoint Returns 500 Error**
   ```bash
   # Check logs
   gcloud logging read "resource.labels.endpoint_id=YOUR_ENDPOINT_ID" \
       --limit 50 --format json
   ```

## 📈 Performance Optimization

### Training Optimization
- Use larger machine types for faster training
- Enable GPU for deep learning models
- Adjust batch_size for memory efficiency

### Serving Optimization
- Enable autoscaling for variable traffic
- Use GPU endpoints for complex models
- Implement caching for frequent predictions

### Cost Optimization
- Use preemptible VMs for training
- Schedule batch predictions during off-peak
- Monitor and optimize resource usage

## 🔒 Security Best Practices

1. **Service Account Permissions**
   - Use least privilege principle
   - Create dedicated service accounts
   - Rotate keys regularly

2. **Data Security**
   - Encrypt data at rest and in transit
   - Use VPC Service Controls
   - Implement data retention policies

3. **Model Security**
   - Version control models
   - Implement model validation
   - Monitor for adversarial inputs

