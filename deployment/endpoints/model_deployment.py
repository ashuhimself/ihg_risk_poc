"""
Model deployment utilities for GCP Vertex AI
Handles model deployment to endpoints and batch prediction setup.
"""

from google.cloud import aiplatform
from google.cloud import storage
import logging
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime


class ModelDeployer:
    """
    Handles model deployment operations on GCP Vertex AI.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the model deployer.
        
        Args:
            project_id: GCP project ID
            location: GCP region for deployment
        """
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def upload_model_to_registry(
        self,
        model_path: str,
        display_name: str,
        serving_container_image_uri: str = "gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest",
        description: Optional[str] = None
    ) -> aiplatform.Model:
        """
        Upload a model to Vertex AI Model Registry.
        
        Args:
            model_path: Path to model artifacts (local or GCS)
            display_name: Display name for the model
            serving_container_image_uri: Container image for serving
            description: Optional model description
            
        Returns:
            Vertex AI Model object
        """
        self.logger.info(f"Uploading model {display_name} to Model Registry")
        
        try:
            model = aiplatform.Model.upload(
                display_name=display_name,
                artifact_uri=model_path,
                serving_container_image_uri=serving_container_image_uri,
                description=description or f"IHG Risk Model - {datetime.now().isoformat()}",
                sync=True
            )
            
            self.logger.info(f"Model uploaded successfully: {model.resource_name}")
            return model
        
        except Exception as e:
            self.logger.error(f"Error uploading model: {e}")
            raise
    
    def create_endpoint(
        self,
        display_name: str,
        description: Optional[str] = None
    ) -> aiplatform.Endpoint:
        """
        Create a Vertex AI endpoint.
        
        Args:
            display_name: Display name for the endpoint
            description: Optional endpoint description
            
        Returns:
            Vertex AI Endpoint object
        """
        self.logger.info(f"Creating endpoint: {display_name}")
        
        try:
            endpoint = aiplatform.Endpoint.create(
                display_name=display_name,
                description=description or f"IHG Risk Endpoint - {datetime.now().isoformat()}",
                sync=True
            )
            
            self.logger.info(f"Endpoint created successfully: {endpoint.resource_name}")
            return endpoint
        
        except Exception as e:
            self.logger.error(f"Error creating endpoint: {e}")
            raise
    
    def deploy_model_to_endpoint(
        self,
        model: aiplatform.Model,
        endpoint: aiplatform.Endpoint,
        deployed_model_display_name: str,
        machine_type: str = "n1-standard-2",
        min_replica_count: int = 1,
        max_replica_count: int = 3,
        traffic_percentage: int = 100
    ) -> aiplatform.Endpoint:
        """
        Deploy a model to an endpoint.
        
        Args:
            model: Vertex AI Model to deploy
            endpoint: Vertex AI Endpoint to deploy to
            deployed_model_display_name: Name for the deployed model
            machine_type: Machine type for serving
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            traffic_percentage: Percentage of traffic to route to this model
            
        Returns:
            Updated Vertex AI Endpoint object
        """
        self.logger.info(f"Deploying model {model.display_name} to endpoint {endpoint.display_name}")
        
        try:
            endpoint = endpoint.deploy(
                model=model,
                deployed_model_display_name=deployed_model_display_name,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                traffic_percentage=traffic_percentage,
                sync=True
            )
            
            self.logger.info(f"Model deployed successfully to endpoint")
            return endpoint
        
        except Exception as e:
            self.logger.error(f"Error deploying model: {e}")
            raise
    
    def create_batch_prediction_job(
        self,
        model: aiplatform.Model,
        input_uri: str,
        output_uri: str,
        job_display_name: str,
        machine_type: str = "n1-standard-4",
        instances_format: str = "jsonl"
    ) -> aiplatform.BatchPredictionJob:
        """
        Create a batch prediction job.
        
        Args:
            model: Vertex AI Model for prediction
            input_uri: GCS URI for input data
            output_uri: GCS URI for output predictions
            job_display_name: Display name for the batch job
            machine_type: Machine type for batch prediction
            instances_format: Format of input instances
            
        Returns:
            Vertex AI BatchPredictionJob object
        """
        self.logger.info(f"Creating batch prediction job: {job_display_name}")
        
        try:
            job = model.batch_predict(
                job_display_name=job_display_name,
                gcs_source=input_uri,
                gcs_destination_prefix=output_uri,
                machine_type=machine_type,
                instances_format=instances_format,
                sync=True
            )
            
            self.logger.info(f"Batch prediction job created: {job.resource_name}")
            return job
        
        except Exception as e:
            self.logger.error(f"Error creating batch prediction job: {e}")
            raise
    
    def full_deployment_pipeline(
        self,
        model_path: str,
        model_name: str,
        endpoint_name: str,
        deployment_config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Complete deployment pipeline from model artifacts to serving endpoint.
        
        Args:
            model_path: Path to model artifacts
            model_name: Name for the model
            endpoint_name: Name for the endpoint
            deployment_config: Configuration for deployment
            
        Returns:
            Dictionary with deployment results
        """
        self.logger.info("Starting full deployment pipeline")
        
        # Upload model to registry
        model = self.upload_model_to_registry(
            model_path=model_path,
            display_name=model_name,
            serving_container_image_uri=deployment_config.get(
                'serving_container_image_uri',
                "gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest"
            )
        )
        
        # Create endpoint
        endpoint = self.create_endpoint(
            display_name=endpoint_name
        )
        
        # Deploy model to endpoint
        endpoint = self.deploy_model_to_endpoint(
            model=model,
            endpoint=endpoint,
            deployed_model_display_name=f"{model_name}-deployed",
            machine_type=deployment_config.get('machine_type', 'n1-standard-2'),
            min_replica_count=deployment_config.get('min_replica_count', 1),
            max_replica_count=deployment_config.get('max_replica_count', 3)
        )
        
        results = {
            'model_resource_name': model.resource_name,
            'endpoint_resource_name': endpoint.resource_name,
            'model_id': model.name,
            'endpoint_id': endpoint.name,
            'deployment_time': datetime.now().isoformat()
        }
        
        self.logger.info("Deployment pipeline completed successfully")
        return results
    
    def update_endpoint_traffic(
        self,
        endpoint_name: str,
        traffic_split: Dict[str, int]
    ) -> aiplatform.Endpoint:
        """
        Update traffic split on an endpoint.
        
        Args:
            endpoint_name: Name or resource name of the endpoint
            traffic_split: Dictionary mapping deployed model IDs to traffic percentages
            
        Returns:
            Updated Vertex AI Endpoint object
        """
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        self.logger.info(f"Updating traffic split for endpoint {endpoint_name}")
        
        endpoint = endpoint.update(
            traffic_split=traffic_split,
            sync=True
        )
        
        self.logger.info("Traffic split updated successfully")
        return endpoint
    
    def undeploy_model(self, endpoint_name: str, deployed_model_id: str) -> aiplatform.Endpoint:
        """
        Undeploy a model from an endpoint.
        
        Args:
            endpoint_name: Name or resource name of the endpoint
            deployed_model_id: ID of the deployed model to remove
            
        Returns:
            Updated Vertex AI Endpoint object
        """
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        self.logger.info(f"Undeploying model {deployed_model_id} from endpoint {endpoint_name}")
        
        endpoint = endpoint.undeploy(
            deployed_model_id=deployed_model_id,
            sync=True
        )
        
        self.logger.info("Model undeployed successfully")
        return endpoint
    
    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Delete an endpoint.
        
        Args:
            endpoint_name: Name or resource name of the endpoint
        """
        endpoint = aiplatform.Endpoint(endpoint_name)
        
        self.logger.info(f"Deleting endpoint {endpoint_name}")
        
        endpoint.delete(sync=True)
        
        self.logger.info("Endpoint deleted successfully")


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy model to Vertex AI')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--location', type=str, default='us-central1', help='GCP location')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model artifacts')
    parser.add_argument('--model_name', type=str, required=True, help='Model display name')
    parser.add_argument('--endpoint_name', type=str, required=True, help='Endpoint display name')
    parser.add_argument('--config_path', type=str, help='Path to deployment config file')
    
    args = parser.parse_args()
    
    # Default deployment configuration
    deployment_config = {
        'machine_type': 'n1-standard-2',
        'min_replica_count': 1,
        'max_replica_count': 3,
        'serving_container_image_uri': 'gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest'
    }
    
    # Load config if provided
    if args.config_path:
        with open(args.config_path, 'r') as f:
            deployment_config.update(json.load(f))
    
    # Deploy model
    deployer = ModelDeployer(args.project_id, args.location)
    results = deployer.full_deployment_pipeline(
        model_path=args.model_path,
        model_name=args.model_name,
        endpoint_name=args.endpoint_name,
        deployment_config=deployment_config
    )
    
    print("Deployment completed successfully!")
    print(f"Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()