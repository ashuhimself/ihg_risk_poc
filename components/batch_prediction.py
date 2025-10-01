"""Batch Prediction Component for Vertex AI"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model

@component(
    packages_to_install=[
        "google-cloud-bigquery==3.13.0",
        "google-cloud-storage==2.10.0",
        "pandas==2.0.3",
        "pandas-gbq==0.19.2",
        "joblib==1.3.2",
        "numpy==1.24.3"
    ],
    base_image="python:3.10"
)
def batch_predict(
    project_id: str,
    bucket_name: str,
    input_table: str,
    output_table: str,
    model_path: str,
    features_path: str,
    predictions: Output[Dataset],
    batch_size: int = 10000,
    
):
    """
    Perform batch predictions on BigQuery data
    
    Args:
        project_id: GCP project ID
        bucket_name: GCS bucket containing model
        input_table: BigQuery table with input data (format: dataset.table)
        output_table: BigQuery table for predictions (format: dataset.table)
        model_path: Path to model in GCS
        features_path: Path to feature names in GCS
        batch_size: Size of batches for prediction
    """
    import pandas as pd
    import numpy as np
    import joblib
    from google.cloud import bigquery, storage
    from datetime import datetime
    
    print(f"Starting batch prediction job...")
    print(f"Input table: {input_table}")
    print(f"Output table: {output_table}")
    
    # Initialize clients
    bq_client = bigquery.Client(project=project_id)
    storage_client = storage.Client(project=project_id)
    
    # Download model from GCS
    bucket = storage_client.bucket(bucket_name)
    
    model_blob = bucket.blob(model_path)
    model_blob.download_to_filename('/tmp/model.pkl')
    
    features_blob = bucket.blob(features_path)
    features_blob.download_to_filename('/tmp/features.pkl')
    
    # Load model and features
    model = joblib.load('/tmp/model.pkl')
    feature_names = joblib.load('/tmp/features.pkl')
    
    print(f"Model loaded. Features: {len(feature_names)}")
    
    # Read data from BigQuery in batches
    query = f"""
    SELECT *
    FROM `{project_id}.{input_table}`
    """
    
    # Get total row count
    count_query = f"""
    SELECT COUNT(*) as total
    FROM `{project_id}.{input_table}`
    """
    
    total_rows = bq_client.query(count_query).to_dataframe()['total'][0]
    print(f"Total rows to process: {total_rows}")
    
    # Process in batches
    all_predictions = []
    
    for offset in range(0, total_rows, batch_size):
        batch_query = f"""
        SELECT *
        FROM `{project_id}.{input_table}`
        LIMIT {batch_size}
        OFFSET {offset}
        """
        
        print(f"Processing batch: {offset} to {min(offset + batch_size, total_rows)}")
        
        # Read batch
        batch_df = bq_client.query(batch_query).to_dataframe()
        
        # Store booking_id for joining results
        booking_ids = batch_df['booking_id'].values
        
        # Prepare features (ensure same order as training)
        X_batch = batch_df[feature_names]
        
        # Make predictions
        fraud_probabilities = model.predict_proba(X_batch)[:, 1]
        fraud_predictions = model.predict(X_batch)
        
        # Create results dataframe
        batch_results = pd.DataFrame({
            'booking_id': booking_ids,
            'fraud_probability': fraud_probabilities,
            'fraud_prediction': fraud_predictions,
            'prediction_timestamp': datetime.now()
        })
        
        all_predictions.append(batch_results)
    
    # Combine all predictions
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Add additional risk categories
    def categorize_risk(prob):
        if prob < 0.3:
            return 'LOW'
        elif prob < 0.7:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    final_predictions['risk_category'] = final_predictions['fraud_probability'].apply(categorize_risk)
    
    # Save to BigQuery
    destination_table = f"{project_id}.{output_table}"
    
    print(f"Writing {len(final_predictions)} predictions to {destination_table}")
    
    # Configure BigQuery job
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",  # Replace table
        schema=[
            bigquery.SchemaField("booking_id", "STRING"),
            bigquery.SchemaField("fraud_probability", "FLOAT64"),
            bigquery.SchemaField("fraud_prediction", "INT64"),
            bigquery.SchemaField("risk_category", "STRING"),
            bigquery.SchemaField("prediction_timestamp", "TIMESTAMP"),
        ],
    )
    
    # Load to BigQuery
    job = bq_client.load_table_from_dataframe(
        final_predictions,
        destination_table,
        job_config=job_config
    )
    job.result()  # Wait for job to complete
    
    print(f"Predictions written to {destination_table}")
    
    # Save predictions summary to output
    final_predictions.to_csv(predictions.path, index=False)
    
    # Print summary statistics
    print("\n=== Prediction Summary ===")
    print(f"Total predictions: {len(final_predictions)}")
    print(f"Fraud predictions: {final_predictions['fraud_prediction'].sum()}")
    print(f"Risk distribution:")
    print(final_predictions['risk_category'].value_counts())
    print(f"Average fraud probability: {final_predictions['fraud_probability'].mean():.4f}")