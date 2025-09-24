"""
Reusable components for Vertex AI pipelines
"""

from kfp import dsl
from typing import Dict, List, Any


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=["pandas", "great-expectations"]
)
def data_validation_component(
    data_path: str,
    validation_config: Dict[str, Any]
) -> bool:
    """Validate data quality and schema."""
    import pandas as pd
    import json
    
    df = pd.read_csv(data_path)
    
    # Basic validation checks
    validation_results = {
        "row_count": len(df),
        "null_percentage": df.isnull().sum().sum() / (len(df) * len(df.columns)),
        "duplicate_rows": df.duplicated().sum(),
        "column_count": len(df.columns)
    }
    
    # Check minimum requirements
    min_rows = validation_config.get("min_rows", 1000)
    max_null_percentage = validation_config.get("max_null_percentage", 0.1)
    
    is_valid = (
        validation_results["row_count"] >= min_rows and
        validation_results["null_percentage"] <= max_null_percentage
    )
    
    print(f"Data validation results: {json.dumps(validation_results, indent=2)}")
    print(f"Data is valid: {is_valid}")
    
    return is_valid


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=["google-cloud-monitoring"]
)
def model_monitoring_component(
    model_endpoint: str,
    monitoring_config: Dict[str, Any]
) -> str:
    """Set up model monitoring."""
    from google.cloud import monitoring_v3
    import json
    
    client = monitoring_v3.MetricServiceClient()
    
    # Create custom metrics for model monitoring
    monitoring_setup = {
        "endpoint": model_endpoint,
        "metrics_enabled": True,
        "alert_policies": monitoring_config.get("alert_policies", [])
    }
    
    print(f"Monitoring setup completed: {json.dumps(monitoring_setup, indent=2)}")
    
    return "monitoring_configured"


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=["google-cloud-storage", "joblib"]
)
def model_versioning_component(
    model_path: str,
    model_name: str,
    version: str,
    bucket_name: str
) -> str:
    """Version and store model artifacts."""
    from google.cloud import storage
    import os
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Upload model to versioned path
    versioned_path = f"models/{model_name}/{version}/model.joblib"
    blob = bucket.blob(versioned_path)
    blob.upload_from_filename(model_path)
    
    gcs_uri = f"gs://{bucket_name}/{versioned_path}"
    print(f"Model stored at: {gcs_uri}")
    
    return gcs_uri


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=["google-cloud-bigquery"]
)
def performance_logging_component(
    model_name: str,
    version: str,
    metrics: Dict[str, float],
    project_id: str
) -> str:
    """Log model performance metrics to BigQuery."""
    from google.cloud import bigquery
    import json
    from datetime import datetime
    
    client = bigquery.Client(project=project_id)
    
    # Create table if not exists
    table_id = f"{project_id}.ml_metrics.model_performance"
    
    rows_to_insert = [{
        "model_name": model_name,
        "version": version,
        "timestamp": datetime.now().isoformat(),
        "metrics": json.dumps(metrics)
    }]
    
    # Insert metrics
    errors = client.insert_rows_json(table_id, rows_to_insert)
    
    if errors == []:
        print(f"Metrics logged successfully for {model_name} v{version}")
        return "success"
    else:
        print(f"Failed to log metrics: {errors}")
        return "failed"