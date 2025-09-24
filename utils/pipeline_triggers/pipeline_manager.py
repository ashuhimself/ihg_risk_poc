"""
Pipeline management utilities for triggering and monitoring ML training pipelines
Handles Vertex AI pipeline orchestration and job management.
"""

from google.cloud import aiplatform
from google.cloud import storage
from kfp.v2 import compiler
import logging
from typing import Dict, List, Any, Optional
import json
import os
import time
from datetime import datetime


class PipelineManager:
    """
    Manages Vertex AI pipeline execution and monitoring.
    """
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize pipeline manager.
        
        Args:
            project_id: GCP project ID
            location: GCP region for pipeline execution
        """
        self.project_id = project_id
        self.location = location
        self.logger = logging.getLogger(__name__)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def compile_and_upload_pipeline(
        self,
        pipeline_func,
        pipeline_name: str,
        pipeline_description: str = "",
        bucket_name: Optional[str] = None
    ) -> str:
        """
        Compile a KFP pipeline and upload to GCS.
        
        Args:
            pipeline_func: KFP pipeline function
            pipeline_name: Name for the compiled pipeline
            pipeline_description: Description of the pipeline
            bucket_name: GCS bucket for storing pipeline JSON
            
        Returns:
            GCS URI of the compiled pipeline
        """
        try:
            # Compile pipeline
            pipeline_file = f"{pipeline_name}.json"
            compiler.Compiler().compile(
                pipeline_func=pipeline_func,
                package_path=pipeline_file
            )
            
            self.logger.info(f"Pipeline compiled: {pipeline_file}")
            
            # Upload to GCS if bucket provided
            if bucket_name:
                storage_client = storage.Client(project=self.project_id)
                bucket = storage_client.bucket(bucket_name)
                
                blob_name = f"pipelines/{pipeline_name}/{pipeline_file}"
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(pipeline_file)
                
                gcs_uri = f"gs://{bucket_name}/{blob_name}"
                self.logger.info(f"Pipeline uploaded to GCS: {gcs_uri}")
                
                # Clean up local file
                os.remove(pipeline_file)
                
                return gcs_uri
            
            return pipeline_file
            
        except Exception as e:
            self.logger.error(f"Error compiling/uploading pipeline: {e}")
            raise
    
    def create_pipeline_job(
        self,
        display_name: str,
        template_path: str,
        parameter_values: Dict[str, Any],
        pipeline_root: Optional[str] = None,
        enable_caching: bool = True
    ) -> aiplatform.PipelineJob:
        """
        Create a Vertex AI pipeline job.
        
        Args:
            display_name: Display name for the pipeline job
            template_path: Path to the compiled pipeline template
            parameter_values: Parameter values for the pipeline
            pipeline_root: GCS root path for pipeline artifacts
            enable_caching: Whether to enable pipeline caching
            
        Returns:
            Vertex AI PipelineJob object
        """
        try:
            if not pipeline_root:
                pipeline_root = f"gs://{self.project_id}-vertex-pipelines"
            
            job = aiplatform.PipelineJob(
                display_name=display_name,
                template_path=template_path,
                parameter_values=parameter_values,
                pipeline_root=pipeline_root,
                enable_caching=enable_caching
            )
            
            self.logger.info(f"Pipeline job created: {display_name}")
            return job
            
        except Exception as e:
            self.logger.error(f"Error creating pipeline job: {e}")
            raise
    
    def submit_pipeline_job(
        self,
        job: aiplatform.PipelineJob,
        service_account: Optional[str] = None,
        network: Optional[str] = None,
        sync: bool = False
    ) -> aiplatform.PipelineJob:
        """
        Submit a pipeline job for execution.
        
        Args:
            job: PipelineJob to submit
            service_account: Service account for pipeline execution
            network: Network configuration
            sync: Whether to wait for completion
            
        Returns:
            Submitted PipelineJob
        """
        try:
            job.submit(
                service_account=service_account,
                network=network
            )
            
            self.logger.info(f"Pipeline job submitted: {job.resource_name}")
            
            if sync:
                job.wait()
                self.logger.info(f"Pipeline job completed: {job.state}")
            
            return job
            
        except Exception as e:
            self.logger.error(f"Error submitting pipeline job: {e}")
            raise
    
    def get_pipeline_job(self, job_id: str) -> aiplatform.PipelineJob:
        """
        Get a pipeline job by ID.
        
        Args:
            job_id: Pipeline job ID
            
        Returns:
            PipelineJob object
        """
        try:
            job = aiplatform.PipelineJob.get(job_id)
            return job
        
        except Exception as e:
            self.logger.error(f"Error getting pipeline job: {e}")
            raise
    
    def get_pipeline_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed status of a pipeline job.
        
        Args:
            job_id: Pipeline job ID
            
        Returns:
            Dictionary with pipeline status information
        """
        try:
            job = self.get_pipeline_job(job_id)
            
            status = {
                'job_id': job.resource_name.split('/')[-1],
                'display_name': job.display_name,
                'state': job.state.name if job.state else 'UNKNOWN',
                'create_time': job.create_time.isoformat() if job.create_time else None,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None,
                'update_time': job.update_time.isoformat() if job.update_time else None,
                'pipeline_spec': job.pipeline_spec.to_dict() if job.pipeline_spec else None,
                'error': None
            }
            
            # Add error information if job failed
            if job.error:
                status['error'] = {
                    'code': job.error.code,
                    'message': job.error.message
                }
            
            # Calculate duration if completed
            if job.start_time and job.end_time:
                duration = job.end_time - job.start_time
                status['duration_seconds'] = duration.total_seconds()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting pipeline status: {e}")
            raise
    
    def list_pipeline_jobs(
        self,
        filter_expr: Optional[str] = None,
        order_by: Optional[str] = None,
        page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List pipeline jobs with optional filtering.
        
        Args:
            filter_expr: Filter expression
            order_by: Ordering specification
            page_size: Number of results per page
            
        Returns:
            List of pipeline job information
        """
        try:
            jobs = aiplatform.PipelineJob.list(
                filter=filter_expr,
                order_by=order_by,
                page_size=page_size
            )
            
            job_list = []
            for job in jobs:
                job_info = {
                    'job_id': job.resource_name.split('/')[-1],
                    'display_name': job.display_name,
                    'state': job.state.name if job.state else 'UNKNOWN',
                    'create_time': job.create_time.isoformat() if job.create_time else None,
                    'update_time': job.update_time.isoformat() if job.update_time else None
                }
                job_list.append(job_info)
            
            return job_list
            
        except Exception as e:
            self.logger.error(f"Error listing pipeline jobs: {e}")
            raise
    
    def cancel_pipeline_job(self, job_id: str) -> Dict[str, str]:
        """
        Cancel a running pipeline job.
        
        Args:
            job_id: Pipeline job ID to cancel
            
        Returns:
            Cancellation result
        """
        try:
            job = self.get_pipeline_job(job_id)
            job.cancel()
            
            self.logger.info(f"Pipeline job cancelled: {job_id}")
            
            return {
                'job_id': job_id,
                'status': 'cancelled',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error cancelling pipeline job: {e}")
            raise
    
    def trigger_training_pipeline(
        self,
        dataset_id: str,
        table_id: str,
        model_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trigger a complete training pipeline.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            model_name: Name for the trained model
            config: Training configuration
            
        Returns:
            Pipeline job information
        """
        try:
            # Import the training pipeline
            from ...pipelines.vertex_ai.training_pipeline import training_pipeline
            
            # Set pipeline parameters
            parameter_values = {
                'project_id': self.project_id,
                'dataset_id': dataset_id,
                'table_id': table_id,
                'model_name': model_name,
                'location': self.location
            }
            
            # Add config parameters
            parameter_values.update(config)
            
            # Compile pipeline (in production, this should be pre-compiled)
            pipeline_file = f"training_pipeline_{int(time.time())}.json"
            compiler.Compiler().compile(
                pipeline_func=training_pipeline,
                package_path=pipeline_file
            )
            
            # Create and submit pipeline job
            display_name = f"training-{model_name}-{int(time.time())}"
            
            job = self.create_pipeline_job(
                display_name=display_name,
                template_path=pipeline_file,
                parameter_values=parameter_values
            )
            
            job = self.submit_pipeline_job(job, sync=False)
            
            # Clean up pipeline file
            os.remove(pipeline_file)
            
            result = {
                'pipeline_id': job.resource_name.split('/')[-1],
                'display_name': display_name,
                'status': 'submitted',
                'parameters': parameter_values,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Training pipeline triggered: {result['pipeline_id']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error triggering training pipeline: {e}")
            raise
    
    def schedule_periodic_training(
        self,
        schedule_name: str,
        cron_expression: str,
        pipeline_config: Dict[str, Any]
    ) -> str:
        """
        Schedule periodic training pipeline execution.
        
        Args:
            schedule_name: Name for the schedule
            cron_expression: Cron expression for scheduling
            pipeline_config: Configuration for the pipeline
            
        Returns:
            Schedule resource name
        """
        try:
            # This is a placeholder implementation
            # In production, you would use Cloud Scheduler or Vertex AI Pipelines scheduling
            
            schedule_config = {
                'name': schedule_name,
                'schedule': cron_expression,
                'pipeline_config': pipeline_config,
                'created': datetime.now().isoformat()
            }
            
            self.logger.info(f"Periodic training scheduled: {schedule_name}")
            
            # Return a placeholder schedule ID
            return f"projects/{self.project_id}/locations/{self.location}/schedules/{schedule_name}"
            
        except Exception as e:
            self.logger.error(f"Error scheduling periodic training: {e}")
            raise
    
    def retrigger_failed_pipeline(self, failed_job_id: str) -> Dict[str, Any]:
        """
        Retrigger a failed pipeline with the same parameters.
        
        Args:
            failed_job_id: ID of the failed pipeline job
            
        Returns:
            New pipeline job information
        """
        try:
            # Get original job
            original_job = self.get_pipeline_job(failed_job_id)
            
            # Extract parameters from original job
            original_params = {}
            if original_job.pipeline_spec and original_job.pipeline_spec.runtime_config:
                original_params = original_job.pipeline_spec.runtime_config.parameter_values
            
            # Create new job with same parameters
            new_display_name = f"retry-{original_job.display_name}-{int(time.time())}"
            
            new_job = self.create_pipeline_job(
                display_name=new_display_name,
                template_path=original_job.template_path,
                parameter_values=original_params
            )
            
            new_job = self.submit_pipeline_job(new_job, sync=False)
            
            result = {
                'original_job_id': failed_job_id,
                'new_pipeline_id': new_job.resource_name.split('/')[-1],
                'display_name': new_display_name,
                'status': 'retriggered',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Pipeline retriggered: {result['new_pipeline_id']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error retriggering pipeline: {e}")
            raise


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage ML pipelines')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--location', type=str, default='us-central1', help='GCP location')
    parser.add_argument('--operation', type=str, 
                       choices=['list', 'status', 'cancel', 'trigger', 'retrigger'],
                       required=True, help='Operation to perform')
    parser.add_argument('--job_id', type=str, help='Pipeline job ID')
    parser.add_argument('--config_path', type=str, help='Path to pipeline config file')
    
    args = parser.parse_args()
    
    manager = PipelineManager(args.project_id, args.location)
    
    if args.operation == 'list':
        result = manager.list_pipeline_jobs()
    elif args.operation == 'status':
        if not args.job_id:
            raise ValueError("job_id required for status operation")
        result = manager.get_pipeline_status(args.job_id)
    elif args.operation == 'cancel':
        if not args.job_id:
            raise ValueError("job_id required for cancel operation")
        result = manager.cancel_pipeline_job(args.job_id)
    elif args.operation == 'trigger':
        if not args.config_path:
            raise ValueError("config_path required for trigger operation")
        
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        
        result = manager.trigger_training_pipeline(**config)
    elif args.operation == 'retrigger':
        if not args.job_id:
            raise ValueError("job_id required for retrigger operation")
        result = manager.retrigger_failed_pipeline(args.job_id)
    
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()