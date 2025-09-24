import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: BASE_URL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for auth
    this.client.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response.data,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error.response?.data || error.message);
      }
    );
  }

  // Health check
  async healthCheck() {
    return this.client.get('/health');
  }

  // Prediction endpoints
  async predict(features, modelVersion = null) {
    return this.client.post('/predict', {
      features,
      model_version: modelVersion,
    });
  }

  async batchPredict(inputUri, outputUri, modelVersion = null) {
    return this.client.post('/batch-predict', {
      input_uri: inputUri,
      output_uri: outputUri,
      model_version: modelVersion,
    });
  }

  // Training endpoints
  async triggerTraining(datasetId, tableId, modelName, config = {}) {
    return this.client.post('/train', {
      dataset_id: datasetId,
      table_id: tableId,
      model_name: modelName,
      config,
    });
  }

  async getTrainingStatus(pipelineId) {
    return this.client.get(`/training-status/${pipelineId}`);
  }

  // Model management
  async getModels() {
    return this.client.get('/models');
  }

  async getModel(modelName) {
    return this.client.get(`/models/${modelName}`);
  }

  // Data management
  async getDataStats() {
    return this.client.get('/data/stats');
  }

  async getDataQuality() {
    return this.client.get('/data/quality');
  }

  // Monitoring
  async getModelPerformance() {
    return this.client.get('/monitoring/model-performance');
  }

  async getSystemHealth() {
    return this.client.get('/monitoring/system-health');
  }
}

export default new ApiService();