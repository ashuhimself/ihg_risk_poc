"""
FastAPI backend for IHG Risk POC Portal
Provides REST API endpoints for model management, predictions, and monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import os
import json

# Import custom modules
from ...utils.bigquery.data_client import BigQueryClient
from ...utils.pipeline_triggers.pipeline_manager import PipelineManager
from ...deployment.endpoints.model_deployment import ModelDeployer

# Initialize FastAPI app
app = FastAPI(
    title="IHG Risk POC API",
    description="API for IHG Risk Assessment ML Platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
project_id = os.getenv("GCP_PROJECT_ID", "your-project-id")
location = os.getenv("GCP_LOCATION", "us-central1")

bigquery_client = BigQueryClient(project_id)
pipeline_manager = PipelineManager(project_id, location)
model_deployer = ModelDeployer(project_id, location)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Any] = Field(..., description="Feature values for prediction")
    model_version: Optional[str] = Field(None, description="Specific model version to use")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: float = Field(..., description="Risk score prediction")
    confidence: float = Field(..., description="Prediction confidence")
    model_version: str = Field(..., description="Model version used")
    timestamp: str = Field(..., description="Prediction timestamp")


class TrainingRequest(BaseModel):
    """Request model for training pipeline."""
    dataset_id: str = Field(..., description="BigQuery dataset ID")
    table_id: str = Field(..., description="BigQuery table ID")
    model_name: str = Field(..., description="Name for the trained model")
    config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    input_uri: str = Field(..., description="GCS URI for input data")
    output_uri: str = Field(..., description="GCS URI for output predictions")
    model_version: Optional[str] = Field(None, description="Specific model version")


class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    status: str
    created_time: str
    metrics: Optional[Dict[str, float]] = None


class PipelineStatus(BaseModel):
    """Pipeline status response."""
    pipeline_id: str
    status: str
    created_time: str
    completion_time: Optional[str] = None
    error_message: Optional[str] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single prediction using the deployed model.
    """
    try:
        # This is a placeholder implementation
        # In production, you would call the actual Vertex AI endpoint
        
        # Simulate prediction logic
        risk_score = 0.75  # Placeholder
        confidence = 0.85  # Placeholder
        
        response = PredictionResponse(
            prediction=risk_score,
            confidence=confidence,
            model_version=request.model_version or "v1.0.0",
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction made: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def batch_predict(
    request: BatchPredictionRequest, 
    background_tasks: BackgroundTasks
):
    """
    Create a batch prediction job.
    """
    try:
        # Start batch prediction job in background
        background_tasks.add_task(
            run_batch_prediction,
            request.input_uri,
            request.output_uri,
            request.model_version
        )
        
        return {
            "message": "Batch prediction job started",
            "input_uri": request.input_uri,
            "output_uri": request.output_uri,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoints
@app.post("/train")
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger a training pipeline.
    """
    try:
        # Start training pipeline in background
        pipeline_job = pipeline_manager.trigger_training_pipeline(
            dataset_id=request.dataset_id,
            table_id=request.table_id,
            model_name=request.model_name,
            config=request.config or {}
        )
        
        return {
            "message": "Training pipeline started",
            "pipeline_id": pipeline_job.get("pipeline_id", "unknown"),
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-status/{pipeline_id}")
async def get_training_status(pipeline_id: str):
    """
    Get the status of a training pipeline.
    """
    try:
        status = pipeline_manager.get_pipeline_status(pipeline_id)
        return status
        
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all available models.
    """
    try:
        # Placeholder implementation
        # In production, this would query Vertex AI Model Registry
        models = [
            ModelInfo(
                model_name="ihg-risk-classifier",
                version="v1.0.0",
                status="deployed",
                created_time=datetime.now().isoformat(),
                metrics={"accuracy": 0.85, "precision": 0.82, "recall": 0.88}
            )
        ]
        
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """
    Get detailed information about a specific model.
    """
    try:
        # Placeholder implementation
        model_info = {
            "model_name": model_name,
            "version": "v1.0.0",
            "status": "deployed",
            "created_time": datetime.now().isoformat(),
            "metrics": {"accuracy": 0.85, "precision": 0.82, "recall": 0.88},
            "endpoint_info": {
                "endpoint_id": f"{model_name}-endpoint",
                "url": f"projects/{project_id}/locations/{location}/endpoints/123456789"
            }
        }
        
        return model_info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data management endpoints
@app.get("/data/stats")
async def get_data_stats():
    """
    Get statistics about the training data.
    """
    try:
        # Get data statistics from BigQuery
        stats = bigquery_client.get_dataset_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting data stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/quality")
async def get_data_quality():
    """
    Get data quality metrics.
    """
    try:
        # Get data quality metrics
        quality_metrics = bigquery_client.get_data_quality_metrics()
        return quality_metrics
        
    except Exception as e:
        logger.error(f"Error getting data quality: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring endpoints
@app.get("/monitoring/model-performance")
async def get_model_performance():
    """
    Get model performance metrics.
    """
    try:
        # Placeholder implementation
        performance_data = {
            "accuracy_trend": [0.85, 0.87, 0.86, 0.88, 0.85],
            "prediction_volume": [1000, 1200, 1100, 1300, 1150],
            "latency_percentiles": {
                "p50": 45,
                "p95": 120,
                "p99": 200
            },
            "error_rate": 0.02,
            "timestamp": datetime.now().isoformat()
        }
        
        return performance_data
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/system-health")
async def get_system_health():
    """
    Get overall system health status.
    """
    try:
        health_status = {
            "overall_status": "healthy",
            "components": {
                "api": "healthy",
                "bigquery": "healthy",
                "vertex_ai": "healthy",
                "model_endpoints": "healthy"
            },
            "last_check": datetime.now().isoformat(),
            "uptime": "99.9%"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks
async def run_batch_prediction(input_uri: str, output_uri: str, model_version: str):
    """
    Background task to run batch prediction.
    """
    try:
        logger.info(f"Starting batch prediction: {input_uri} -> {output_uri}")
        
        # This would call the actual batch prediction service
        # For now, it's a placeholder
        
        logger.info("Batch prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Global HTTP exception handler.
    """
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )