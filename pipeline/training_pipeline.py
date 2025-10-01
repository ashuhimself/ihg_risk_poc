"""Fraud Detection Training Pipeline for Vertex AI"""

from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform
from datetime import datetime
import os
import sys

# Add parent directory to path to import components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.data_preprocessing import preprocess_data
from components.model_training import train_ensemble_model
from components.model_evaluation import evaluate_model

# Pipeline definition
@dsl.pipeline(
    name="fraud-detection-training-pipeline",
    description="Train fraud detection ensemble model using BigQuery data"
)
def fraud_detection_pipeline(
    project_id: str = "ihg-mlops",
    dataset_id: str = "ihg_training_data",
    table_id: str = "booking",
    bucket_name: str = "ihg-mlops",
    test_size: float = 0.3,
    random_state: int = 42,
    importance_threshold: float = 0.05,
    model_name: str = "fraud_detection_ensemble"
):
    """
    End-to-end training pipeline for fraud detection model
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        bucket_name: GCS bucket name
        test_size: Test split ratio
        random_state: Random seed
        importance_threshold: Feature importance threshold
        model_name: Name for the trained model
    """
    
    # Step 1: Data Preprocessing
    preprocess_task = preprocess_data(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        test_size=test_size,
        random_state=random_state
    )
    
    # Step 2: Model Training
    train_task = train_ensemble_model(
        project_id=project_id,
        bucket_name=bucket_name,
        importance_threshold=importance_threshold,
        train_data=preprocess_task.outputs["train_data"],
        test_data=preprocess_task.outputs["test_data"],
        feature_columns=preprocess_task.outputs["feature_columns"]
    )
    
    # Step 3: Model Evaluation
    evaluate_task = evaluate_model(
        project_id=project_id,
        bucket_name=bucket_name,
        model_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        test_data=preprocess_task.outputs["test_data"],
        selected_features=train_task.outputs["selected_features"],
        model=train_task.outputs["model"]
    )
    
    # Set pipeline level configurations - REDUCED FOR QUOTA
    preprocess_task.set_cpu_limit('2')
    preprocess_task.set_memory_limit('8G')
    
    train_task.set_cpu_limit('2')
    train_task.set_memory_limit('8G')
    
    evaluate_task.set_cpu_limit('2')
    evaluate_task.set_memory_limit('8G')


def compile_pipeline():
    """Compile the pipeline to JSON"""
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_pipeline,
        package_path="fraud_detection_pipeline.json"
    )
    print("Pipeline compiled successfully to fraud_detection_pipeline.json")


def run_pipeline(
    project_id: str = "ihg-mlops",
    location: str = "us-central1",
    pipeline_root: str = "gs://ihg-mlops/pipeline-root",
    service_account: str = None
):
    """
    Run the pipeline on Vertex AI
    
    Args:
        project_id: GCP project ID
        location: GCP region
        pipeline_root: GCS path for pipeline artifacts
        service_account: Service account email for pipeline execution
    """
    
    # Initialize Vertex AI
    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=pipeline_root
    )
    
    # Compile pipeline
    compile_pipeline()
    
    # Create pipeline job
    job = aiplatform.PipelineJob(
        display_name=f"fraud-detection-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        template_path="fraud_detection_pipeline.json",
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "dataset_id": "ihg_training_data",
            "table_id": "booking",
            "bucket_name": "ihg-mlops",
            "test_size": 0.3,
            "random_state": 42,
            "importance_threshold": 0.05,
            "model_name": "fraud_detection_ensemble"
        },
        enable_caching=True
    )
    
    # Submit pipeline job
    job.submit(service_account=service_account)
    
    print(f"Pipeline job submitted: {job.resource_name}")
    print(f"View in console: https://console.cloud.google.com/vertex-ai/pipelines/runs/{job.name}?project={project_id}")
    
    return job


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fraud Detection Training Pipeline")
    parser.add_argument("--compile-only", action="store_true", help="Only compile the pipeline")
    parser.add_argument("--project-id", default="ihg-mlops", help="GCP Project ID")
    parser.add_argument("--location", default="us-central1", help="GCP Region")
    parser.add_argument("--pipeline-root", default="gs://ihg-mlops/pipeline-root", help="Pipeline root path")
    parser.add_argument("--service-account", help="Service account for pipeline execution")
    
    args = parser.parse_args()
    
    if args.compile_only:
        compile_pipeline()
    else:
        run_pipeline(
            project_id=args.project_id,
            location=args.location,
            pipeline_root=args.pipeline_root,
            service_account=args.service_account
        )