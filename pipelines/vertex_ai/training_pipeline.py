"""
Vertex AI Training Pipeline for IHG Risk POC
This module defines the ML training pipeline using Google Cloud Vertex AI.
"""

from kfp import dsl
from kfp.v2 import compiler
from google.cloud import aiplatform
from typing import Dict, Any
import os


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=["google-cloud-bigquery", "pandas", "scikit-learn"]
)
def data_extraction_component(
    project_id: str,
    dataset_id: str,
    table_id: str
) -> str:
    """Extract data from BigQuery for training."""
    from google.cloud import bigquery
    import pandas as pd
    
    client = bigquery.Client(project=project_id)
    query = f"""
    SELECT *
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    """
    
    df = client.query(query).to_dataframe()
    data_path = "/tmp/training_data.csv"
    df.to_csv(data_path, index=False)
    
    return data_path


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/sklearn-cpu:latest",
    packages_to_install=["joblib", "pandas"]
)
def model_training_component(
    data_path: str,
    model_name: str
) -> str:
    """Train the ML model."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib
    import os
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Placeholder preprocessing - customize based on your data
    X = df.drop('target', axis=1, errors='ignore')  # Adjust column name
    y = df.get('target', pd.Series([0] * len(df)))  # Placeholder target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")
    
    # Save model
    model_path = f"/tmp/{model_name}.joblib"
    joblib.dump(model, model_path)
    
    return model_path


@dsl.component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu:latest",
    packages_to_install=["google-cloud-aiplatform"]
)
def model_deployment_component(
    model_path: str,
    model_name: str,
    project_id: str,
    location: str
) -> str:
    """Deploy model to Vertex AI endpoint."""
    from google.cloud import aiplatform
    
    aiplatform.init(project=project_id, location=location)
    
    # Upload model to Vertex AI Model Registry
    model = aiplatform.Model.upload(
        display_name=model_name,
        artifact_uri=model_path,
        serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.0-24:latest"
    )
    
    # Create endpoint
    endpoint = aiplatform.Endpoint.create(
        display_name=f"{model_name}-endpoint"
    )
    
    # Deploy model to endpoint
    endpoint.deploy(
        model=model,
        deployed_model_display_name=f"{model_name}-deployed",
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=3
    )
    
    return endpoint.name


@dsl.pipeline(
    name="ihg-risk-training-pipeline",
    description="Training pipeline for IHG Risk POC"
)
def training_pipeline(
    project_id: str = "your-project-id",
    dataset_id: str = "your_dataset",
    table_id: str = "your_table",
    model_name: str = "ihg-risk-model",
    location: str = "us-central1"
):
    """Main training pipeline."""
    
    # Step 1: Extract data
    data_task = data_extraction_component(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id
    )
    
    # Step 2: Train model
    training_task = model_training_component(
        data_path=data_task.output,
        model_name=model_name
    )
    
    # Step 3: Deploy model
    deployment_task = model_deployment_component(
        model_path=training_task.output,
        model_name=model_name,
        project_id=project_id,
        location=location
    )


def compile_pipeline():
    """Compile the pipeline."""
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.json"
    )


if __name__ == "__main__":
    compile_pipeline()
    print("Pipeline compiled successfully!")