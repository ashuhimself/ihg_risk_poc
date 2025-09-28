import React, { useState } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Paper,
  Box,
  Alert,
  Divider
} from '@mui/material';
import apiService from '../services/apiService';

const PredictionInterface = () => {
  const [features, setFeatures] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Sample feature fields - customize based on your model
  const featureFields = [
    { name: 'feature1', label: 'Feature 1', type: 'number' },
    { name: 'feature2', label: 'Feature 2', type: 'number' },
    { name: 'feature3', label: 'Feature 3', type: 'text' },
    { name: 'feature4', label: 'Feature 4', type: 'number' },
  ];

  const handleFeatureChange = (featureName, value) => {
    setFeatures({
      ...features,
      [featureName]: value
    });
  };

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await apiService.predict(features);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFeatures({});
    setPrediction(null);
    setError(null);
  };

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Risk Prediction Interface
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Input Features */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Input Features
              </Typography>
              
              {featureFields.map((field) => (
                <TextField
                  key={field.name}
                  fullWidth
                  label={field.label}
                  type={field.type}
                  value={features[field.name] || ''}
                  onChange={(e) => handleFeatureChange(field.name, e.target.value)}
                  margin="normal"
                  variant="outlined"
                />
              ))}

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handlePredict}
                  disabled={loading}
                  fullWidth
                >
                  {loading ? 'Predicting...' : 'Get Prediction'}
                </Button>
                <Button
                  variant="outlined"
                  onClick={handleReset}
                  fullWidth
                >
                  Reset
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Result */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Prediction Result
              </Typography>

              {prediction ? (
                <Paper elevation={1} sx={{ p: 2, bgcolor: 'grey.50' }}>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Risk Score
                    </Typography>
                    <Typography variant="h3" color="primary">
                      {(prediction.prediction * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Divider sx={{ my: 2 }} />

                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence
                    </Typography>
                    <Typography variant="h6">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Model Version
                    </Typography>
                    <Typography variant="body1">
                      {prediction.model_version}
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" color="text.secondary">
                      Timestamp
                    </Typography>
                    <Typography variant="body1">
                      {new Date(prediction.timestamp).toLocaleString()}
                    </Typography>
                  </Box>
                </Paper>
              ) : (
                <Paper elevation={1} sx={{ p: 3, textAlign: 'center', bgcolor: 'grey.50' }}>
                  <Typography color="text.secondary">
                    Enter feature values and click "Get Prediction" to see results
                  </Typography>
                </Paper>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Batch Prediction */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Batch Prediction
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                For processing large datasets, use batch prediction with GCS URIs
              </Typography>

              <Grid container spacing={2}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Input GCS URI"
                    placeholder="gs://your-bucket/input/data.jsonl"
                    margin="normal"
                    variant="outlined"
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    fullWidth
                    label="Output GCS URI"
                    placeholder="gs://your-bucket/output/"
                    margin="normal"
                    variant="outlined"
                  />
                </Grid>
              </Grid>

              <Button
                variant="outlined"
                color="primary"
                sx={{ mt: 2 }}
              >
                Start Batch Prediction
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};

export default PredictionInterface;