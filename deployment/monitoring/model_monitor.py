"""
Model monitoring utilities for tracking model performance and drift
Provides real-time monitoring and alerting capabilities.
"""

from google.cloud import monitoring_v3
from google.cloud import bigquery
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


class ModelMonitor:
    """
    Handles model monitoring and performance tracking.
    """
    
    def __init__(self, project_id: str):
        """
        Initialize model monitor.
        
        Args:
            project_id: GCP project ID
        """
        self.project_id = project_id
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        self.bq_client = bigquery.Client(project=project_id)
        self.logger = logging.getLogger(__name__)
    
    def log_prediction_metrics(
        self,
        model_name: str,
        prediction_latency: float,
        prediction_value: float,
        confidence: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log prediction metrics to monitoring system.
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Log to BigQuery for historical tracking
        self._log_to_bigquery(
            model_name=model_name,
            prediction_latency=prediction_latency,
            prediction_value=prediction_value,
            confidence=confidence,
            timestamp=timestamp
        )
        
        # Create custom metrics for Cloud Monitoring
        self._create_custom_metric(
            metric_type=f"custom.googleapis.com/ml/{model_name}/latency",
            value=prediction_latency,
            timestamp=timestamp
        )
    
    def _log_to_bigquery(
        self,
        model_name: str,
        prediction_latency: float,
        prediction_value: float,
        confidence: float,
        timestamp: datetime
    ) -> None:
        """Log metrics to BigQuery."""
        try:
            table_id = f"{self.project_id}.ml_monitoring.prediction_logs"
            
            rows_to_insert = [{
                "model_name": model_name,
                "timestamp": timestamp.isoformat(),
                "prediction_latency": prediction_latency,
                "prediction_value": prediction_value,
                "confidence": confidence,
                "created_at": datetime.now().isoformat()
            }]
            
            errors = self.bq_client.insert_rows_json(table_id, rows_to_insert)
            
            if errors:
                self.logger.error(f"Error logging to BigQuery: {errors}")
            
        except Exception as e:
            self.logger.error(f"Failed to log to BigQuery: {e}")
    
    def _create_custom_metric(
        self,
        metric_type: str,
        value: float,
        timestamp: datetime
    ) -> None:
        """Create custom metric in Cloud Monitoring."""
        try:
            project_name = f"projects/{self.project_id}"
            
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            series.resource.type = "global"
            
            now = timestamp
            seconds = int(now.timestamp())
            nanos = int((now.timestamp() - seconds) * 10**9)
            
            interval = monitoring_v3.TimeInterval(
                {
                    "end_time": {"seconds": seconds, "nanos": nanos}
                }
            )
            point = monitoring_v3.Point(
                {
                    "interval": interval,
                    "value": {"double_value": value}
                }
            )
            series.points = [point]
            
            self.monitoring_client.create_time_series(
                name=project_name, time_series=[series]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create custom metric: {e}")
    
    def check_data_drift(
        self,
        dataset_id: str,
        table_id: str,
        reference_data: Optional[pd.DataFrame] = None,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Check for data drift in recent data.
        """
        try:
            # Get recent data
            query = f"""
            SELECT *
            FROM `{self.project_id}.{dataset_id}.{table_id}`
            WHERE DATE(_PARTITIONTIME) >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_days} DAY)
            """
            
            recent_data = self.bq_client.query(query).to_dataframe()
            
            if recent_data.empty:
                return {"status": "no_data", "message": "No recent data found"}
            
            # Placeholder drift detection logic
            # In production, you would implement statistical tests like KS test, PSI, etc.
            drift_results = {
                "status": "checked",
                "timestamp": datetime.now().isoformat(),
                "data_points": len(recent_data),
                "drift_detected": False,  # Placeholder
                "drift_score": 0.1,  # Placeholder
                "message": "No significant drift detected"
            }
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Error checking data drift: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_model_report(
        self,
        model_name: str,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model performance report.
        """
        try:
            # Query prediction logs
            query = f"""
            SELECT 
                DATE(PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S', timestamp)) as date,
                AVG(prediction_latency) as avg_latency,
                AVG(confidence) as avg_confidence,
                COUNT(*) as prediction_count,
                STDDEV(prediction_value) as prediction_std
            FROM `{self.project_id}.ml_monitoring.prediction_logs`
            WHERE model_name = '{model_name}'
                AND DATE(PARSE_TIMESTAMP('%Y-%m-%dT%H:%M:%E*S', timestamp)) 
                    >= DATE_SUB(CURRENT_DATE(), INTERVAL {days_back} DAY)
            GROUP BY date
            ORDER BY date DESC
            """
            
            performance_data = self.bq_client.query(query).to_dataframe()
            
            if performance_data.empty:
                return {"status": "no_data", "message": "No performance data found"}
            
            # Calculate summary statistics
            report = {
                "model_name": model_name,
                "report_period": f"{days_back} days",
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "total_predictions": int(performance_data['prediction_count'].sum()),
                    "avg_daily_predictions": int(performance_data['prediction_count'].mean()),
                    "avg_latency_ms": float(performance_data['avg_latency'].mean()),
                    "avg_confidence": float(performance_data['avg_confidence'].mean()),
                    "performance_trend": self._calculate_trend(performance_data)
                },
                "daily_metrics": performance_data.to_dict('records')
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating model report: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_trend(self, data: pd.DataFrame) -> str:
        """Calculate performance trend from data."""
        if len(data) < 2:
            return "insufficient_data"
        
        # Simple trend calculation based on confidence
        recent_confidence = data.head(7)['avg_confidence'].mean()
        older_confidence = data.tail(7)['avg_confidence'].mean()
        
        if recent_confidence > older_confidence * 1.05:
            return "improving"
        elif recent_confidence < older_confidence * 0.95:
            return "declining"
        else:
            return "stable"


def main():
    """Example usage of ModelMonitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor model performance')
    parser.add_argument('--project_id', type=str, required=True, help='GCP project ID')
    parser.add_argument('--model_name', type=str, required=True, help='Model name to monitor')
    parser.add_argument('--operation', type=str, choices=['report', 'drift'], 
                       default='report', help='Operation to perform')
    
    args = parser.parse_args()
    
    monitor = ModelMonitor(args.project_id)
    
    if args.operation == 'report':
        report = monitor.generate_model_report(args.model_name)
        print(json.dumps(report, indent=2, default=str))
    elif args.operation == 'drift':
        # You would need to specify dataset details for drift detection
        drift_result = monitor.check_data_drift('your_dataset', 'your_table')
        print(json.dumps(drift_result, indent=2, default=str))


if __name__ == "__main__":
    main()