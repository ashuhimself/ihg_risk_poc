"""Model Evaluation Component for Vertex AI Pipeline"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics

@component(
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "joblib==1.3.2",
        "google-cloud-storage==2.10.0"
    ],
    base_image="python:3.10"
)
def evaluate_model(
    project_id: str,
    bucket_name: str,
    model_name: str,
    test_data: Input[Dataset],
    selected_features: Input[Dataset],
    model: Input[Model],
    metrics: Output[Metrics],
    classification_metrics: Output[ClassificationMetrics]
):
    """
    Evaluate the trained model and generate detailed metrics
    """
    import pandas as pd
    import numpy as np
    import joblib
    import json
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        roc_auc_score,
        average_precision_score,
        roc_curve,
        precision_recall_curve
    )
    from google.cloud import storage
    
    print("Loading model and data...")
    # Load model
    ensemble = joblib.load(model.path + ".pkl")
    
    # Load test data
    test_df = pd.read_csv(test_data.path)
    features_df = pd.read_csv(selected_features.path)
    features_to_keep = features_df['features'].tolist()
    
    # Prepare test data
    X_test = test_df[features_to_keep]
    y_test = test_df['fraud_flag']
    
    # Make predictions
    print("Making predictions...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate business metrics
    # Assuming average transaction value and fraud cost
    avg_transaction_value = 1000  # You can make this configurable
    fraud_investigation_cost = 50  # Cost per investigation
    
    # Cost-benefit analysis
    fraud_prevented = tp * avg_transaction_value
    false_alarm_cost = fp * fraud_investigation_cost
    missed_fraud_cost = fn * avg_transaction_value
    net_benefit = fraud_prevented - false_alarm_cost - missed_fraud_cost
    
    print(f"\n=== Model Evaluation Results ===")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"\n=== Confusion Matrix ===")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"\n=== Business Impact ===")
    print(f"Fraud Prevented: ${fraud_prevented:,.2f}")
    print(f"False Alarm Cost: ${false_alarm_cost:,.2f}")
    print(f"Missed Fraud Cost: ${missed_fraud_cost:,.2f}")
    print(f"Net Benefit: ${net_benefit:,.2f}")
    
    # Log metrics
    metrics.log_metric("roc_auc", roc_auc)
    metrics.log_metric("pr_auc", pr_auc)
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1_score", f1_score)
    metrics.log_metric("true_negatives", int(tn))
    metrics.log_metric("false_positives", int(fp))
    metrics.log_metric("false_negatives", int(fn))
    metrics.log_metric("true_positives", int(tp))
    metrics.log_metric("fraud_prevented_value", fraud_prevented)
    metrics.log_metric("false_alarm_cost", false_alarm_cost)
    metrics.log_metric("missed_fraud_cost", missed_fraud_cost)
    metrics.log_metric("net_benefit", net_benefit)
    
    # Set classification metrics for Vertex AI
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    classification_metrics.log_roc_curve(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        threshold=[float(t) for t in np.linspace(0, 1, len(fpr))]
    )
    
    # Save evaluation report to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    evaluation_report = {
        "model_name": model_name,
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "business_impact": {
            "fraud_prevented_value": fraud_prevented,
            "false_alarm_cost": false_alarm_cost,
            "missed_fraud_cost": missed_fraud_cost,
            "net_benefit": net_benefit
        },
        "classification_report": report
    }
    
    # Upload evaluation report
    report_blob = bucket.blob(f'evaluations/{model_name}_evaluation.json')
    report_blob.upload_from_string(json.dumps(evaluation_report, indent=2))
    print(f"\nEvaluation report saved to gs://{bucket_name}/evaluations/{model_name}_evaluation.json")