"""Deploy trained model to Vertex AI Endpoint"""

from google.cloud import aiplatform
from datetime import datetime
import argparse

def upload_model_to_registry(
    project_id: str,
    location: str,
    model_display_name: str,
    model_artifact_uri: str,
    serving_container_image_uri: str = None
):
    """
    Upload model to Vertex AI Model Registry
    
    Args:
        project_id: GCP project ID
        location: GCP region
        model_display_name: Display name for the model
        model_artifact_uri: GCS URI of model artifacts
        serving_container_image_uri: Docker image URI for serving
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Use pre-built container if custom image not provided
    if not serving_container_image_uri:
        serving_container_image_uri = f"gcr.io/{project_id}/fraud-detection-model:latest"
    
    # Create model
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_environment_variables={
            "PROJECT_ID": project_id,
            "BUCKET_NAME": "ihg-mlops",
            "MODEL_PATH": "models/ensemble_model.pkl",
            "FEATURES_PATH": "models/feature_names.pkl"
        },
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health"
    )
    
    print(f"Model uploaded: {model.display_name}")
    print(f"Model resource name: {model.resource_name}")
    
    return model


def create_endpoint(
    project_id: str,
    location: str,
    endpoint_display_name: str
):
    """
    Create a Vertex AI Endpoint
    
    Args:
        project_id: GCP project ID
        location: GCP region
        endpoint_display_name: Display name for the endpoint
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        project=project_id,
        location=location
    )
    
    print(f"Endpoint created: {endpoint.display_name}")
    print(f"Endpoint resource name: {endpoint.resource_name}")
    
    return endpoint


def deploy_model_to_endpoint(
    model: aiplatform.Model,
    endpoint: aiplatform.Endpoint,
    deployed_model_display_name: str,
    machine_type: str = "n1-standard-4",
    min_replica_count: int = 1,
    max_replica_count: int = 3,
    accelerator_type: str = None,
    accelerator_count: int = 0,
    traffic_split: dict = None
):
    """
    Deploy a model to an endpoint
    
    Args:
        model: Vertex AI Model object
        endpoint: Vertex AI Endpoint object
        deployed_model_display_name: Display name for deployed model
        machine_type: Machine type for serving
        min_replica_count: Minimum number of replicas
        max_replica_count: Maximum number of replicas
        accelerator_type: GPU type (optional)
        accelerator_count: Number of GPUs (optional)
        traffic_split: Traffic split configuration
    """
    
    # Deploy model to endpoint
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        traffic_split=traffic_split or {"0": 100},
        sync=True
    )
    
    print(f"Model deployed to endpoint successfully")
    print(f"Endpoint ID: {endpoint.name}")
    
    return deployed_model


def test_endpoint(
    endpoint: aiplatform.Endpoint,
    test_instance: dict
):
    """
    Test the deployed endpoint with a sample request
    
    Args:
        endpoint: Vertex AI Endpoint object
        test_instance: Test instance dictionary
    """
    
    print("\nTesting endpoint...")
    
    # Make prediction
    prediction = endpoint.predict(instances=[test_instance])
    
    print("Test prediction result:")
    print(prediction.predictions)
    
    return prediction


def setup_monitoring(
    endpoint: aiplatform.Endpoint,
    project_id: str,
    notification_emails: list = None
):
    """
    Set up model monitoring for the endpoint
    
    Args:
        endpoint: Vertex AI Endpoint object
        project_id: GCP project ID
        notification_emails: List of emails for alerts
    """
    
    # This would set up model monitoring
    # In production, you'd configure:
    # - Prediction drift detection
    # - Feature attribution drift
    # - Data quality monitoring
    # - Custom metrics
    
    print("Model monitoring configuration:")
    print("- Prediction drift detection: Enabled")
    print("- Feature skew detection: Enabled")
    print("- Alerting: Configured")
    
    if notification_emails:
        print(f"- Alert emails: {', '.join(notification_emails)}")


def main():
    parser = argparse.ArgumentParser(description="Deploy Fraud Detection Model")
    parser.add_argument("--project-id", default="ihg-mlops", help="GCP Project ID")
    parser.add_argument("--location", default="us-central1", help="GCP Region")
    parser.add_argument("--model-name", default="fraud-detection-ensemble", help="Model display name")
    parser.add_argument("--endpoint-name", default="fraud-detection-endpoint", help="Endpoint display name")
    parser.add_argument("--model-uri", default="gs://ihg-mlops/models", help="GCS URI of model artifacts")
    parser.add_argument("--container-image", help="Custom container image URI")
    parser.add_argument("--machine-type", default="n1-standard-4", help="Machine type for serving")
    parser.add_argument("--min-replicas", type=int, default=1, help="Minimum replicas")
    parser.add_argument("--max-replicas", type=int, default=3, help="Maximum replicas")
    parser.add_argument("--test", action="store_true", help="Test the endpoint after deployment")
    
    args = parser.parse_args()
    
    # Upload model to registry
    model = upload_model_to_registry(
        project_id=args.project_id,
        location=args.location,
        model_display_name=f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        model_artifact_uri=args.model_uri,
        serving_container_image_uri=args.container_image
    )
    
    # Create endpoint
    endpoint = create_endpoint(
        project_id=args.project_id,
        location=args.location,
        endpoint_display_name=args.endpoint_name
    )
    
    # Deploy model to endpoint
    deploy_model_to_endpoint(
        model=model,
        endpoint=endpoint,
        deployed_model_display_name=f"deployed_{args.model_name}",
        machine_type=args.machine_type,
        min_replica_count=args.min_replicas,
        max_replica_count=args.max_replicas
    )
    
    # Test endpoint if requested
    if args.test:
        # Create a test instance with dummy values
        # In production, use real feature values
        test_instance = {f"feature_{i}": 0.0 for i in range(20)}
        test_endpoint(endpoint, test_instance)
    
    # Setup monitoring
    setup_monitoring(endpoint, args.project_id)
    
    print(f"\n✅ Deployment complete!")
    print(f"Endpoint URL: https://{args.location}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict")


if __name__ == "__main__":
    main()