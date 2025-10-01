"""Model Training Component for Vertex AI Pipeline"""

from kfp.v2.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    packages_to_install=[
        "pandas==2.0.3",
        "numpy==1.24.3",
        "scikit-learn==1.3.2",
        "xgboost==2.0.2",
        "lightgbm==4.1.0",
        "joblib==1.3.2",
        "google-cloud-storage==2.10.0"
    ],
    base_image="python:3.10"
)
def train_ensemble_model(
    project_id: str,
    bucket_name: str,
    importance_threshold: float,
    train_data: Input[Dataset],
    test_data: Input[Dataset],
    feature_columns: Input[Dataset],
    model: Output[Model],
    selected_features: Output[Dataset],
    metrics: Output[Metrics]
):
    """
    Train ensemble model with XGBoost, LightGBM, and Logistic Regression
    """
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    from google.cloud import storage
    import os
    
    print("Loading datasets...")
    # Load data
    train_df = pd.read_csv(train_data.path)
    test_df = pd.read_csv(test_data.path)
    feature_cols_df = pd.read_csv(feature_columns.path)
    feature_cols = feature_cols_df['features'].tolist()
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['fraud_flag']
    X_test = test_df[feature_cols]
    y_test = test_df['fraud_flag']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"Scale_pos_weight: {scale_pos_weight}")
    
    # Initialize models
    print("Training individual models...")
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6
    )
    
    lgb = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6
    )
    
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    
    # Train individual models for feature importance
    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    
    # Calculate feature importances
    print("Calculating feature importances...")
    
    def get_feature_importances(lr, xgb, lgb, feature_names):
        # Logistic Regression importances (absolute coefficients)
        lr_importances = np.abs(lr.coef_[0])
        lr_df = pd.DataFrame({"feature": feature_names, "lr": lr_importances})
        
        # XGBoost importances
        xgb_importances = xgb.feature_importances_
        xgb_df = pd.DataFrame({"feature": feature_names, "xgb": xgb_importances})
        
        # LightGBM importances
        lgb_importances = lgb.feature_importances_
        lgb_df = pd.DataFrame({"feature": feature_names, "lgb": lgb_importances})
        
        # Merge
        all_importances = lr_df.merge(xgb_df, on="feature").merge(lgb_df, on="feature")
        
        # Normalize each column
        for col in ["lr", "xgb", "lgb"]:
            all_importances[col] = all_importances[col] / (all_importances[col].sum() + 1e-9)
        
        # Average importance
        all_importances["avg_importance"] = all_importances[["lr", "xgb", "lgb"]].mean(axis=1)
        
        # Sort by importance
        all_importances = all_importances.sort_values(by="avg_importance", ascending=False).reset_index(drop=True)
        
        return all_importances
    
    # Get feature importances
    feature_importances = get_feature_importances(lr, xgb, lgb, X_train.columns)
    print(f"Top 10 important features:\n{feature_importances.head(10)}")
    
    # Select features based on threshold
    features_to_keep = feature_importances[
        feature_importances['avg_importance'] > importance_threshold
    ]['feature'].tolist()
    
    print(f"Keeping {len(features_to_keep)} features out of {len(feature_importances)}")
    
    # Filter features
    X_train_filtered = X_train[features_to_keep]
    X_test_filtered = X_test[features_to_keep]
    
    # Re-train models with selected features
    print("Re-training models with selected features...")
    xgb_new = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6
    )
    
    lgb_new = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_estimators=100,
        max_depth=6
    )
    
    lr_new = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[("lr", lr_new), ("xgb", xgb_new), ("lgb", lgb_new)],
        voting="soft"
    )
    
    # Train ensemble
    print("Training ensemble model...")
    ensemble.fit(X_train_filtered, y_train)
    
    # Evaluate
    y_pred = ensemble.predict(X_test_filtered)
    y_proba = ensemble.predict_proba(X_test_filtered)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    
    # Save model and features
    model_path = model.path + ".pkl"
    joblib.dump(ensemble, model_path)
    
    # Save selected features
    pd.DataFrame({'features': features_to_keep}).to_csv(selected_features.path, index=False)
    
    # Upload to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    # Upload model
    model_blob = bucket.blob('models/ensemble_model.pkl')
    model_blob.upload_from_filename(model_path)
    print(f"Model uploaded to gs://{bucket_name}/models/ensemble_model.pkl")
    
    # Upload feature names
    features_blob = bucket.blob('models/feature_names.pkl')
    joblib.dump(features_to_keep, '/tmp/feature_names.pkl')
    features_blob.upload_from_filename('/tmp/feature_names.pkl')
    print(f"Features uploaded to gs://{bucket_name}/models/feature_names.pkl")
    
    # Log metrics
    metrics.log_metric("roc_auc", roc_auc)
    metrics.log_metric("pr_auc", pr_auc)
    metrics.log_metric("n_selected_features", len(features_to_keep))
    metrics.log_metric("scale_pos_weight", scale_pos_weight)