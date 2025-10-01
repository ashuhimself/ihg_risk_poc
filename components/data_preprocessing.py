"""Data Preprocessing Component for Vertex AI Pipeline"""

from typing import NamedTuple, Dict
from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics
from kfp.v2 import compiler

@component(
    packages_to_install=[
        "google-cloud-bigquery==3.13.0",
        "pandas==2.0.3",
        "pandas-gbq==0.19.2",
        "db-dtypes==1.1.1",
        "pyarrow==14.0.1",
        "scikit-learn==1.3.2"
    ],
    base_image="python:3.10"
)
def preprocess_data(
    project_id: str,
    dataset_id: str,
    table_id: str,
    test_size: float,
    random_state: int,
    train_data: Output[Dataset],
    test_data: Output[Dataset],
    feature_columns: Output[Dataset],
    metrics: Output[Metrics]
) -> NamedTuple('Outputs', [('n_samples', int), ('n_features', int), ('fraud_ratio', float)]):
    """
    Read data from BigQuery and prepare train/test datasets
    """
    import pandas as pd
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    import json
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Construct query
    query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.{table_id}`
    """
    
    print(f"Reading data from BigQuery: {project_id}.{dataset_id}.{table_id}")
    
    # Read data from BigQuery
    df = client.query(query).to_dataframe()
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Data quality checks
    initial_rows = len(df)
    df = df.dropna()  # Remove any null values
    rows_after_cleaning = len(df)
    print(f"Removed {initial_rows - rows_after_cleaning} rows with null values")
    
    # Prepare features and labels
    X = df.drop(["fraud_flag", "booking_id"], axis=1, errors='ignore')
    y = df["fraud_flag"]
    
    # Calculate fraud ratio
    fraud_ratio = y.sum() / len(y)
    print(f"Fraud ratio: {fraud_ratio:.4f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create train and test DataFrames
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save datasets
    train_df.to_csv(train_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    # Save feature columns
    feature_cols = X.columns.tolist()
    pd.DataFrame({'features': feature_cols}).to_csv(feature_columns.path, index=False)
    
    # Log metrics
    metrics.log_metric("total_samples", len(df))
    metrics.log_metric("train_samples", len(train_df))
    metrics.log_metric("test_samples", len(test_df))
    metrics.log_metric("n_features", len(feature_cols))
    metrics.log_metric("fraud_ratio", fraud_ratio)
    
    from collections import namedtuple
    outputs = namedtuple('Outputs', ['n_samples', 'n_features', 'fraud_ratio'])
    return outputs(len(df), len(feature_cols), fraud_ratio)