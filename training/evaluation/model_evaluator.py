# Evaluation utilities and metrics calculation
"""
Model evaluation utilities for the IHG Risk POC
Provides comprehensive model evaluation metrics and reporting.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            labels: Class labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        
        # ROC AUC if probabilities provided
        if y_pred_proba is not None:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:  # Multi-class
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                except ValueError:
                    self.logger.warning("Could not calculate ROC AUC for multi-class")
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive regression evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Example usage of ModelEvaluator."""
    # Example with dummy data
    evaluator = ModelEvaluator()
    
    # Generate dummy classification data
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_pred_proba = np.random.random(1000)
    
    # Evaluate
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_pred_proba)
    
    print("Classification Metrics:")
    for key, value in metrics.items():
        if key != 'classification_report':
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()