"""
BigQuery client utilities for data fetching and management
Handles data extraction, monitoring, and analytics for the ML pipeline.
"""

from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import json


class BigQueryClient:
    """
    Wrapper class for BigQuery operations specific to the ML pipeline.
    """
    
    def __init__(self, project_id: str):
        """
        Initialize BigQuery client.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a BigQuery SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            
        Returns:
            Results as pandas DataFrame
        """
        try:
            self.logger.info(f"Executing query: {query[:100]}...")
            job_config = bigquery.QueryJobConfig()
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert to DataFrame
            df = results.to_dataframe()
            
            self.logger.info(f"Query completed, returned {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
    
    def get_training_data(
        self,
        dataset_id: str,
        table_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        where_clause: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch training data from BigQuery table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            start_date: Start date filter (YYYY-MM-DD format)
            end_date: End date filter (YYYY-MM-DD format)
            limit: Maximum number of rows to return
            where_clause: Additional WHERE conditions
            
        Returns:
            Training data as DataFrame
        """
        # Build query
        base_query = f"""
        SELECT *
        FROM `{self.project_id}.{dataset_id}.{table_id}`
        """
        
        conditions = []
        
        # Add date filters
        if start_date:
            conditions.append(f"DATE(_PARTITIONTIME) >= '{start_date}'")
        
        if end_date:
            conditions.append(f"DATE(_PARTITIONTIME) <= '{end_date}'")
        
        # Add custom where clause
        if where_clause:
            conditions.append(where_clause)
        
        # Add conditions to query
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Add ordering and limit
        base_query += " ORDER BY _PARTITIONTIME DESC"
        
        if limit:
            base_query += f" LIMIT {limit}"
        
        return self.execute_query(base_query)
    
    def get_dataset_stats(self, dataset_id: str = None, table_id: str = None) -> Dict[str, Any]:
        """
        Get comprehensive statistics about a dataset or table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID (optional)
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            stats = {}
            
            if dataset_id and table_id:
                # Get table-specific statistics
                table_ref = self.client.dataset(dataset_id).table(table_id)
                table = self.client.get_table(table_ref)
                
                stats = {
                    'table_name': table_id,
                    'row_count': table.num_rows,
                    'size_bytes': table.num_bytes,
                    'size_mb': round(table.num_bytes / 1024 / 1024, 2),
                    'column_count': len(table.schema),
                    'created': table.created.isoformat() if table.created else None,
                    'modified': table.modified.isoformat() if table.modified else None,
                    'schema': [{'name': field.name, 'type': field.field_type} for field in table.schema]
                }
                
                # Get data distribution statistics
                if table.num_rows > 0:
                    data_stats_query = f"""
                    SELECT
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT *) as unique_rows,
                        COUNT(*) - COUNT(DISTINCT *) as duplicate_rows
                    FROM `{self.project_id}.{dataset_id}.{table_id}`
                    """
                    
                    data_stats_df = self.execute_query(data_stats_query)
                    if not data_stats_df.empty:
                        stats.update(data_stats_df.iloc[0].to_dict())
            
            elif dataset_id:
                # Get dataset-level statistics
                dataset = self.client.get_dataset(dataset_id)
                tables = list(self.client.list_tables(dataset))
                
                stats = {
                    'dataset_id': dataset_id,
                    'table_count': len(tables),
                    'created': dataset.created.isoformat() if dataset.created else None,
                    'modified': dataset.modified.isoformat() if dataset.modified else None,
                    'location': dataset.location,
                    'tables': [table.table_id for table in tables]
                }
            
            stats['last_updated'] = datetime.now().isoformat()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting dataset stats: {e}")
            raise
    
    def get_data_quality_metrics(
        self,
        dataset_id: str,
        table_id: str
    ) -> Dict[str, Any]:
        """
        Calculate data quality metrics for a table.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            # Get table schema
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Build query to check data quality
            quality_checks = []
            
            for field in table.schema:
                col_name = field.name
                
                # Null value check
                quality_checks.append(f"""
                    COUNTIF({col_name} IS NULL) as {col_name}_nulls,
                    ROUND(COUNTIF({col_name} IS NULL) * 100.0 / COUNT(*), 2) as {col_name}_null_pct
                """)
                
                # For string fields, check empty values
                if field.field_type == 'STRING':
                    quality_checks.append(f"""
                        COUNTIF({col_name} = '' OR {col_name} IS NULL) as {col_name}_empty,
                        ROUND(COUNTIF({col_name} = '' OR {col_name} IS NULL) * 100.0 / COUNT(*), 2) as {col_name}_empty_pct
                    """)
            
            quality_query = f"""
            SELECT
                COUNT(*) as total_rows,
                {','.join(quality_checks)}
            FROM `{self.project_id}.{dataset_id}.{table_id}`
            """
            
            quality_df = self.execute_query(quality_query)
            
            if quality_df.empty:
                return {'error': 'No data quality metrics available'}
            
            metrics = quality_df.iloc[0].to_dict()
            
            # Calculate overall quality score
            null_columns = [k for k in metrics.keys() if k.endswith('_null_pct')]
            avg_null_pct = sum(metrics[col] for col in null_columns) / len(null_columns) if null_columns else 0
            
            quality_score = max(0, 100 - avg_null_pct)
            
            result = {
                'quality_score': round(quality_score, 2),
                'total_rows': metrics['total_rows'],
                'null_percentages': {k: metrics[k] for k in null_columns},
                'assessment': 'Good' if quality_score >= 80 else 'Fair' if quality_score >= 60 else 'Poor',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality metrics: {e}")
            raise
    
    def monitor_data_freshness(
        self,
        dataset_id: str,
        table_id: str,
        timestamp_column: str = '_PARTITIONTIME'
    ) -> Dict[str, Any]:
        """
        Monitor data freshness and update patterns.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            timestamp_column: Column to use for timestamp analysis
            
        Returns:
            Dictionary with data freshness metrics
        """
        try:
            freshness_query = f"""
            WITH daily_counts AS (
                SELECT
                    DATE({timestamp_column}) as date,
                    COUNT(*) as row_count
                FROM `{self.project_id}.{dataset_id}.{table_id}`
                WHERE DATE({timestamp_column}) >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
                GROUP BY DATE({timestamp_column})
                ORDER BY date DESC
            )
            SELECT
                MAX(date) as latest_date,
                MIN(date) as earliest_date,
                AVG(row_count) as avg_daily_rows,
                STDDEV(row_count) as stddev_daily_rows,
                COUNT(*) as days_with_data,
                DATE_DIFF(CURRENT_DATE(), MAX(date), DAY) as days_since_last_update
            FROM daily_counts
            """
            
            freshness_df = self.execute_query(freshness_query)
            
            if freshness_df.empty:
                return {'error': 'No freshness data available'}
            
            result = freshness_df.iloc[0].to_dict()
            
            # Assess freshness
            days_since_update = result.get('days_since_last_update', 999)
            
            if days_since_update == 0:
                freshness_status = 'Current'
            elif days_since_update <= 1:
                freshness_status = 'Recent'
            elif days_since_update <= 7:
                freshness_status = 'Stale'
            else:
                freshness_status = 'Very Stale'
            
            result['freshness_status'] = freshness_status
            result['timestamp'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error monitoring data freshness: {e}")
            raise
    
    def create_training_view(
        self,
        source_dataset_id: str,
        source_table_id: str,
        view_dataset_id: str,
        view_name: str,
        feature_columns: List[str],
        target_column: str,
        where_clause: Optional[str] = None
    ) -> str:
        """
        Create a view for training data with specific features and target.
        
        Args:
            source_dataset_id: Source dataset ID
            source_table_id: Source table ID
            view_dataset_id: Dataset ID for the view
            view_name: Name for the training view
            feature_columns: List of feature column names
            target_column: Target column name
            where_clause: Optional WHERE clause for filtering
            
        Returns:
            Full view ID
        """
        try:
            # Build query for the view
            columns = feature_columns + [target_column]
            column_list = ', '.join(columns)
            
            view_query = f"""
            SELECT {column_list}
            FROM `{self.project_id}.{source_dataset_id}.{source_table_id}`
            """
            
            if where_clause:
                view_query += f" WHERE {where_clause}"
            
            # Create view
            view_id = f"{self.project_id}.{view_dataset_id}.{view_name}"
            view = bigquery.Table(view_id)
            view.view_query = view_query
            
            view = self.client.create_table(view, exists_ok=True)
            
            self.logger.info(f"Training view created: {view_id}")
            return view_id
            
        except Exception as e:
            self.logger.error(f"Error creating training view: {e}")
            raise
    
    def export_to_gcs(
        self,
        dataset_id: str,
        table_id: str,
        gcs_uri: str,
        format: str = 'CSV'
    ) -> str:
        """
        Export BigQuery table to Google Cloud Storage.
        
        Args:
            dataset_id: BigQuery dataset ID
            table_id: BigQuery table ID
            gcs_uri: GCS URI for export (e.g., 'gs://bucket/path/file.csv')
            format: Export format (CSV, JSON, PARQUET)
            
        Returns:
            Job ID of the export job
        """
        try:
            table_ref = self.client.dataset(dataset_id).table(table_id)
            
            job_config = bigquery.ExtractJobConfig()
            job_config.destination_format = getattr(bigquery.DestinationFormat, format)
            
            extract_job = self.client.extract_table(
                table_ref,
                gcs_uri,
                job_config=job_config
            )
            
            extract_job.result()  # Wait for job to complete
            
            self.logger.info(f"Export completed: {gcs_uri}")
            return extract_job.job_id
            
        except Exception as e:
            self.logger.error(f"Error exporting to GCS: {e}")
            raise


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='BigQuery data operations')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--dataset_id', type=str, required=True, help='BigQuery dataset ID')
    parser.add_argument('--table_id', type=str, help='BigQuery table ID')
    parser.add_argument('--operation', type=str, choices=['stats', 'quality', 'freshness'], 
                       required=True, help='Operation to perform')
    
    args = parser.parse_args()
    
    client = BigQueryClient(args.project_id)
    
    if args.operation == 'stats':
        result = client.get_dataset_stats(args.dataset_id, args.table_id)
    elif args.operation == 'quality':
        if not args.table_id:
            raise ValueError("table_id required for quality operation")
        result = client.get_data_quality_metrics(args.dataset_id, args.table_id)
    elif args.operation == 'freshness':
        if not args.table_id:
            raise ValueError("table_id required for freshness operation")
        result = client.monitor_data_freshness(args.dataset_id, args.table_id)
    
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()