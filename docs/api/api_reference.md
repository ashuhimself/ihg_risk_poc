# API Documentation

## IHG Risk Assessment Platform API

Base URL: `http://localhost:8000` (development) | `https://your-domain.com` (production)

### Authentication
All API endpoints require proper authentication. Include your API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Health Check

### GET /health
Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

## Prediction Endpoints

### POST /predict
Make a single prediction using the deployed model.

**Request Body:**
```json
{
  "features": {
    "feature1": 0.5,
    "feature2": 100,
    "feature3": "category_a"
  },
  "model_version": "v1.0.0"
}
```

**Response:**
```json
{
  "prediction": 0.75,
  "confidence": 0.85,
  "model_version": "v1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### POST /batch-predict
Create a batch prediction job.

**Request Body:**
```json
{
  "input_uri": "gs://your-bucket/input/data.jsonl",
  "output_uri": "gs://your-bucket/output/",
  "model_version": "v1.0.0"
}
```

**Response:**
```json
{
  "message": "Batch prediction job started",
  "input_uri": "gs://your-bucket/input/data.jsonl",
  "output_uri": "gs://your-bucket/output/",
  "status": "started",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Training Endpoints

### POST /train
Trigger a training pipeline.

**Request Body:**
```json
{
  "dataset_id": "risk_dataset",
  "table_id": "training_data",
  "model_name": "risk_classifier_v2",
  "config": {
    "max_trials": 50,
    "train_budget_milli_node_hours": 1000
  }
}
```

**Response:**
```json
{
  "message": "Training pipeline started",
  "pipeline_id": "12345678-1234-1234-1234-123456789012",
  "status": "started",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /training-status/{pipeline_id}
Get the status of a training pipeline.

**Response:**
```json
{
  "pipeline_id": "12345678-1234-1234-1234-123456789012",
  "status": "RUNNING",
  "created_time": "2024-01-15T10:30:00Z",
  "completion_time": null,
  "error_message": null
}
```

## Model Management Endpoints

### GET /models
List all available models.

**Response:**
```json
[
  {
    "model_name": "ihg-risk-classifier",
    "version": "v1.0.0",
    "status": "deployed",
    "created_time": "2024-01-15T10:00:00Z",
    "metrics": {
      "accuracy": 0.85,
      "precision": 0.82,
      "recall": 0.88
    }
  }
]
```

### GET /models/{model_name}
Get detailed information about a specific model.

**Response:**
```json
{
  "model_name": "ihg-risk-classifier",
  "version": "v1.0.0",
  "status": "deployed",
  "created_time": "2024-01-15T10:00:00Z",
  "metrics": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.88
  },
  "endpoint_info": {
    "endpoint_id": "ihg-risk-classifier-endpoint",
    "url": "projects/your-project/locations/us-central1/endpoints/123456789"
  }
}
```

## Data Management Endpoints

### GET /data/stats
Get statistics about the training data.

**Response:**
```json
{
  "total_rows": 1000000,
  "total_columns": 25,
  "last_updated": "2024-01-15T08:00:00Z",
  "size_mb": 512.5,
  "quality_score": 95.2
}
```

### GET /data/quality
Get data quality metrics.

**Response:**
```json
{
  "quality_score": 95.2,
  "total_rows": 1000000,
  "null_percentages": {
    "feature1_null_pct": 0.1,
    "feature2_null_pct": 0.05
  },
  "assessment": "Good",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Monitoring Endpoints

### GET /monitoring/model-performance
Get model performance metrics.

**Response:**
```json
{
  "accuracy_trend": [0.85, 0.87, 0.86, 0.88, 0.85],
  "prediction_volume": [1000, 1200, 1100, 1300, 1150],
  "latency_percentiles": {
    "p50": 45,
    "p95": 120,
    "p99": 200
  },
  "error_rate": 0.02,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /monitoring/system-health
Get overall system health status.

**Response:**
```json
{
  "overall_status": "healthy",
  "components": {
    "api": "healthy",
    "bigquery": "healthy",
    "vertex_ai": "healthy",
    "model_endpoints": "healthy"
  },
  "last_check": "2024-01-15T10:30:00Z",
  "uptime": "99.9%"
}
```

## Error Responses

All error responses follow this format:

```json
{
  "error": "Detailed error message",
  "status_code": 400,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common HTTP Status Codes
- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error

## Rate Limiting
API requests are rate limited to:
- 100 requests per minute per API key
- 1000 requests per hour per API key

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## Pagination
List endpoints support pagination using query parameters:
- `page`: Page number (default: 1)
- `per_page`: Items per page (default: 20, max: 100)

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 150,
    "pages": 8
  }
}
```

## WebSocket Endpoints
For real-time updates, connect to WebSocket endpoints:

### /ws/pipeline-status/{pipeline_id}
Receive real-time updates on pipeline execution status.

### /ws/model-monitoring
Receive real-time model performance updates.

## SDK Usage Examples

### Python
```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "features": {"feature1": 0.5, "feature2": 100},
        "model_version": "v1.0.0"
    },
    headers={"Authorization": "Bearer YOUR_API_KEY"}
)
prediction = response.json()
print(f"Risk score: {prediction['prediction']}")
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    features: { feature1: 0.5, feature2: 100 },
    model_version: 'v1.0.0'
  })
});
const prediction = await response.json();
console.log(`Risk score: ${prediction.prediction}`);
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"features": {"feature1": 0.5, "feature2": 100}, "model_version": "v1.0.0"}'
```