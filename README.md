# IHG Risk Assessment MLOps Platform

A full-featured MLOps framework built on Google Cloud Platform for risk assessment modeling. This repository implements an end-to-end machine learning lifecycle with automated training, deployment, monitoring, and human-in-the-loop feedback capabilities.

## 🏗️ Architecture Overview

The platform is designed with a modular architecture supporting continuous ML workflows:

```
📁 ihg_risk_poc/
├── 🔄 pipelines/          # Vertex AI ML pipelines
│   ├── vertex_ai/         # Training & deployment pipelines
│   ├── components/        # Reusable pipeline components
│   └── configs/           # Pipeline configurations
├── 🎯 training/           # Model training modules
│   ├── models/            # Model training scripts
│   ├── data_processing/   # Data preprocessing utilities
│   └── evaluation/        # Model evaluation tools
├── 🚀 deployment/         # Model deployment
│   ├── endpoints/         # Vertex AI endpoint management
│   ├── monitoring/        # Model monitoring setup
│   └── scripts/           # Deployment automation
├── 🌐 portal/             # Web application
│   ├── backend/           # FastAPI REST API
│   └── frontend/          # React web interface
├── 🔧 utils/              # Utility modules
│   ├── bigquery/          # BigQuery data operations
│   └── pipeline_triggers/ # Pipeline orchestration
└── 📚 docs/               # Documentation
    ├── architecture/      # System architecture docs
    └── api/               # API documentation
```

## ✨ Key Features

### 🤖 Automated ML Pipeline
- **Data Extraction**: Automated BigQuery data fetching with quality validation
- **Feature Engineering**: Scalable data preprocessing and feature creation
- **Model Training**: Support for multiple ML frameworks (scikit-learn, TensorFlow, PyTorch)
- **AutoML Integration**: Vertex AI AutoML for rapid model development
- **Hyperparameter Tuning**: Automated optimization with Vertex AI

### 🎯 Model Management
- **Version Control**: Complete model lifecycle management
- **A/B Testing**: Canary deployments with traffic splitting
- **Performance Monitoring**: Real-time model performance tracking
- **Rollback Capabilities**: Safe deployment with instant rollback options

### 📊 Data Quality & Monitoring
- **Data Validation**: Automated schema and quality checks
- **Drift Detection**: Statistical drift monitoring for features and predictions
- **Outlier Detection**: Anomaly detection in incoming data
- **Data Lineage**: Complete traceability from source to predictions

### 🌐 User Interface
- **Web Portal**: React-based interface for data scientists and business users
- **Real-time Dashboards**: Performance metrics and system health monitoring
- **Prediction Interface**: Interactive prediction testing and batch processing
- **Model Management**: GUI for model deployment and lifecycle management

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+ (for frontend)
- Google Cloud SDK
- Docker (optional)

### 1. Clone and Setup
```bash
git clone https://github.com/ashuhimself/ihg_risk_poc.git
cd ihg_risk_poc
chmod +x setup.sh
./setup.sh
```

### 2. Configure GCP
```bash
# Set up your GCP service account key
cp service-account-key.json.template service-account-key.json
# Update with your actual service account credentials

# Configure environment
cp .env.template .env
# Update .env with your project details
```

### 3. Start the Services
```bash
# Backend API
source venv/bin/activate
uvicorn portal.backend.main:app --reload

# Frontend (in a new terminal)
cd portal/frontend
npm start

# Or use Docker Compose
docker-compose up -d
```

### 4. Access the Platform
- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📋 Usage Examples

### Training a New Model
```bash
# Using the CLI
python training/models/risk_model.py \
  --data_path gs://your-bucket/data.csv \
  --output_path ./models/risk_model_v1

# Using the API
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "risk_dataset",
    "table_id": "training_data",
    "model_name": "risk_classifier_v1"
  }'
```

### Making Predictions
```bash
# Single prediction via API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "feature1": 0.5,
      "feature2": 100,
      "category": "high_risk"
    }
  }'

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "input_uri": "gs://your-bucket/batch_input.jsonl",
    "output_uri": "gs://your-bucket/predictions/"
  }'
```

### Pipeline Management
```bash
# Trigger training pipeline
python utils/pipeline_triggers/pipeline_manager.py \
  --project_id your-project \
  --operation trigger \
  --config_path pipeline_config.json

# Check pipeline status
python utils/pipeline_triggers/pipeline_manager.py \
  --project_id your-project \
  --operation status \
  --job_id pipeline-job-id
```

## 🛠️ Development

### Code Structure
- **Modular Design**: Each component is self-contained with clear interfaces
- **Type Hints**: Comprehensive type annotations for better code quality
- **Documentation**: Inline documentation and comprehensive API docs
- **Testing**: Unit tests for critical components
- **Logging**: Structured logging throughout the application

### Adding New Models
1. Create model class in `training/models/`
2. Add preprocessing logic in `training/data_processing/`
3. Update pipeline configuration in `pipelines/configs/`
4. Test locally before deployment

### Extending the API
1. Add new endpoints in `portal/backend/main.py`
2. Update API documentation
3. Add corresponding frontend components if needed
4. Update tests and validation

## 🔧 Configuration

### Pipeline Configuration
Update `pipelines/configs/pipeline_config.yaml`:
```yaml
PROJECT_ID: "your-gcp-project"
LOCATION: "us-central1"
BUCKET_NAME: "your-ml-bucket"
DATASET_ID: "your_dataset"
MODEL_NAME: "your-model-name"
```

### Environment Variables
Key environment variables in `.env`:
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_LOCATION`: GCP region (default: us-central1)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key
- `BQ_DATASET_ID`: BigQuery dataset for training data
- `API_KEY`: API authentication key

## 📊 Monitoring & Observability

### Model Performance
- **Accuracy Tracking**: Real-time accuracy monitoring
- **Latency Metrics**: Response time percentiles
- **Error Rates**: Request failure monitoring
- **Resource Usage**: CPU/memory utilization

### Data Quality
- **Schema Validation**: Automatic schema drift detection
- **Data Freshness**: Monitoring data update frequency
- **Quality Scores**: Automated data quality assessment
- **Anomaly Detection**: Statistical outlier identification

### System Health
- **Service Status**: Component health monitoring
- **Infrastructure**: GCP resource monitoring
- **Alerts**: Automated alert configuration
- **Dashboards**: Real-time system dashboards

## 🔒 Security & Compliance

- **IAM Integration**: GCP Identity and Access Management
- **Data Encryption**: At-rest and in-transit encryption
- **Audit Logging**: Comprehensive activity logging
- **VPC Security**: Network-level security controls
- **Secrets Management**: Secure credential handling

## 📚 Documentation

- [**System Architecture**](docs/architecture/system_architecture.md): Detailed architecture overview
- [**API Reference**](docs/api/api_reference.md): Complete API documentation
- [**Deployment Guide**](docs/deployment/README.md): Production deployment instructions
- [**Troubleshooting**](docs/troubleshooting/README.md): Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make changes and add tests
4. Run tests: `pytest`
5. Commit changes: `git commit -am 'Add new feature'`
6. Push to branch: `git push origin feature/new-feature`
7. Submit a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in this repository
- Check the [troubleshooting guide](docs/troubleshooting/README.md)
- Review the [API documentation](docs/api/api_reference.md)

## 🗺️ Roadmap

### Current Version (v1.0.0)
- ✅ Core MLOps pipeline framework
- ✅ Vertex AI integration
- ✅ FastAPI backend with React frontend
- ✅ BigQuery data integration
- ✅ Basic monitoring and logging

### Upcoming Features
- 🔄 Advanced AutoML integration
- 🔄 Edge deployment support
- 🔄 Advanced monitoring dashboards
- 🔄 MLflow experiment tracking
- 🔄 Kubernetes deployment options
- 🔄 Multi-region deployment
- 🔄 Advanced security features

---

Built with ❤️ for scalable MLOps on Google Cloud Platform
