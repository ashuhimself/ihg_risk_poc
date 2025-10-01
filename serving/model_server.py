"""Model Serving for Vertex AI Endpoints"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and features
MODEL = None
FEATURE_NAMES = None

def load_model():
    """Load model from GCS"""
    global MODEL, FEATURE_NAMES
    
    # Get environment variables
    project_id = os.environ.get('PROJECT_ID', 'ihg-mlops')
    bucket_name = os.environ.get('BUCKET_NAME', 'ihg-mlops')
    model_path = os.environ.get('MODEL_PATH', 'models/ensemble_model.pkl')
    features_path = os.environ.get('FEATURES_PATH', 'models/feature_names.pkl')
    
    logger.info(f"Loading model from gs://{bucket_name}/{model_path}")
    
    # Initialize storage client
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    # Download model
    model_blob = bucket.blob(model_path)
    model_blob.download_to_filename('/tmp/model.pkl')
    
    # Download features
    features_blob = bucket.blob(features_path)
    features_blob.download_to_filename('/tmp/features.pkl')
    
    # Load into memory
    MODEL = joblib.load('/tmp/model.pkl')
    FEATURE_NAMES = joblib.load('/tmp/features.pkl')
    
    logger.info(f"Model loaded successfully. Features: {len(FEATURE_NAMES)}")

@app.before_first_request
def initialize():
    """Initialize model on first request"""
    load_model()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected input format:
    {
        "instances": [
            {
                "feature1": value1,
                "feature2": value2,
                ...
            }
        ]
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if 'instances' not in data:
            return jsonify({'error': 'Missing instances field'}), 400
        
        instances = data['instances']
        
        # Convert to DataFrame
        df = pd.DataFrame(instances)
        
        # Validate features
        missing_features = set(FEATURE_NAMES) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}'
            }), 400
        
        # Select and order features
        X = df[FEATURE_NAMES]
        
        # Make predictions
        probabilities = MODEL.predict_proba(X)[:, 1]
        predictions = MODEL.predict(X)
        
        # Categorize risk
        risk_categories = []
        for prob in probabilities:
            if prob < 0.3:
                risk_categories.append('LOW')
            elif prob < 0.7:
                risk_categories.append('MEDIUM')
            else:
                risk_categories.append('HIGH')
        
        # Format response
        response = {
            'predictions': []
        }
        
        for i in range(len(predictions)):
            response['predictions'].append({
                'fraud_prediction': int(predictions[i]),
                'fraud_probability': float(probabilities[i]),
                'risk_category': risk_categories[i]
            })
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/explain', methods=['POST'])
def explain():
    """
    Explain predictions (feature importance for specific instance)
    """
    try:
        # This is a placeholder for SHAP or LIME explanations
        # You can implement model-agnostic explanations here
        data = request.get_json()
        
        if 'instances' not in data:
            return jsonify({'error': 'Missing instances field'}), 400
        
        # For now, return global feature importance
        # In production, you'd calculate instance-specific explanations
        
        return jsonify({
            'explanations': [
                {
                    'method': 'global_importance',
                    'top_features': FEATURE_NAMES[:10]  # Top 10 features
                }
            ]
        }), 200
        
    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model at startup if running locally
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)