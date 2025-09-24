"""
Data processing utilities for the IHG Risk POC
Handles data extraction, transformation, and preparation for ML training.
"""

import pandas as pd
import numpy as np
from google.cloud import bigquery
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import os


class DataProcessor:
    """
    Main class for data processing operations.
    """
    
    def __init__(self, project_id: str):
        """
        Initialize the data processor.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)
    
    def extract_data_from_bigquery(
        self, 
        dataset_id: str, 
        table_id: str,
        date_filter_days: int = 30,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data from BigQuery.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            date_filter_days: Number of days to look back for data
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with extracted data
        """
        # Build query
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT *
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL {date_filter_days} DAY)
        ORDER BY _PARTITIONTIME DESC
        {limit_clause}
        """
        
        self.logger.info(f"Executing query on {dataset_id}.{table_id}")
        
        try:
            df = self.bq_client.query(query).to_dataframe()
            self.logger.info(f"Extracted {len(df)} rows from BigQuery")
            return df
        except Exception as e:
            self.logger.error(f"Error extracting data from BigQuery: {e}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform feature engineering on the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create date-based features if datetime columns exist
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_weekday'] = df[col].dt.weekday
            df[f'{col}_is_weekend'] = (df[col].dt.weekday >= 5).astype(int)
        
        # Create interaction features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            # Create some interaction features (limit to avoid explosion)
            for i, col1 in enumerate(numeric_cols[:5]):
                for j, col2 in enumerate(numeric_cols[i+1:6]):
                    df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        # Create aggregated features
        if len(numeric_cols) > 0:
            df['numeric_sum'] = df[numeric_cols].sum(axis=1)
            df['numeric_mean'] = df[numeric_cols].mean(axis=1)
            df['numeric_std'] = df[numeric_cols].std(axis=1).fillna(0)
        
        self.logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return quality metrics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with data quality metrics
        """
        metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns)
        }
        
        # Quality checks
        quality_issues = []
        
        if metrics['duplicate_rows'] > 0:
            quality_issues.append(f"Found {metrics['duplicate_rows']} duplicate rows")
        
        high_missing_cols = [col for col, pct in metrics['missing_percentage'].items() if pct > 50]
        if high_missing_cols:
            quality_issues.append(f"Columns with >50% missing values: {high_missing_cols}")
        
        metrics['quality_issues'] = quality_issues
        metrics['quality_score'] = max(0, 100 - len(quality_issues) * 10)
        
        return metrics
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> str:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            output_path: Path to save the data
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_path.endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        
        self.logger.info(f"Data saved to {output_path}")
        return output_path
    
    def create_training_dataset(
        self, 
        dataset_id: str, 
        table_id: str, 
        output_path: str,
        **kwargs
    ) -> str:
        """
        Complete pipeline to create training dataset.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            output_path: Path to save processed data
            **kwargs: Additional parameters for data extraction
            
        Returns:
            Path to saved training dataset
        """
        # Extract data
        df = self.extract_data_from_bigquery(dataset_id, table_id, **kwargs)
        
        # Clean data
        df = self.clean_data(df)
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Validate data quality
        quality_metrics = self.validate_data_quality(df)
        self.logger.info(f"Data quality score: {quality_metrics['quality_score']}/100")
        
        # Save processed data
        return self.save_processed_data(df, output_path)


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process data for training')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--dataset_id', type=str, required=True, help='BigQuery dataset ID')
    parser.add_argument('--table_id', type=str, required=True, help='BigQuery table ID')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    parser.add_argument('--days', type=int, default=30, help='Days of data to extract')
    parser.add_argument('--limit', type=int, help='Limit number of rows')
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.project_id)
    
    output_file = processor.create_training_dataset(
        dataset_id=args.dataset_id,
        table_id=args.table_id,
        output_path=args.output_path,
        date_filter_days=args.days,
        limit=args.limit
    )
    
    print(f"Training dataset created: {output_file}")


if __name__ == "__main__":
    main()