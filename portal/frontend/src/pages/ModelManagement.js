import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Alert
} from '@mui/material';
import apiService from '../services/apiService';

const ModelManagement = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [trainDialogOpen, setTrainDialogOpen] = useState(false);
  const [trainConfig, setTrainConfig] = useState({
    dataset_id: '',
    table_id: '',
    model_name: '',
  });

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await apiService.getModels();
      setModels(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModel = async () => {
    try {
      await apiService.triggerTraining(
        trainConfig.dataset_id,
        trainConfig.table_id,
        trainConfig.model_name
      );
      setTrainDialogOpen(false);
      setTrainConfig({ dataset_id: '', table_id: '', model_name: '' });
      // Refresh models list
      fetchModels();
    } catch (err) {
      setError(err.message);
    }
  };

  const getStatusColor = (status) => {
    switch (status.toLowerCase()) {
      case 'deployed':
        return 'success';
      case 'training':
        return 'warning';
      case 'failed':
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Model Management
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <Typography variant="h6">
                  Active Models
                </Typography>
                <Button
                  variant="contained"
                  color="primary"
                  onClick={() => setTrainDialogOpen(true)}
                >
                  Train New Model
                </Button>
              </div>

              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Model Name</TableCell>
                      <TableCell>Version</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell>Created</TableCell>
                      <TableCell>Accuracy</TableCell>
                      <TableCell>Actions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {models.map((model, index) => (
                      <TableRow key={index}>
                        <TableCell>{model.model_name}</TableCell>
                        <TableCell>{model.version}</TableCell>
                        <TableCell>
                          <Chip
                            label={model.status}
                            color={getStatusColor(model.status)}
                            size="small"
                          />
                        </TableCell>
                        <TableCell>
                          {new Date(model.created_time).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          {model.metrics?.accuracy ? 
                            `${(model.metrics.accuracy * 100).toFixed(1)}%` : 
                            'N/A'
                          }
                        </TableCell>
                        <TableCell>
                          <Button size="small" color="primary">
                            View Details
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Training Dialog */}
      <Dialog open={trainDialogOpen} onClose={() => setTrainDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Train New Model</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Dataset ID"
            fullWidth
            variant="outlined"
            value={trainConfig.dataset_id}
            onChange={(e) => setTrainConfig({ ...trainConfig, dataset_id: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Table ID"
            fullWidth
            variant="outlined"
            value={trainConfig.table_id}
            onChange={(e) => setTrainConfig({ ...trainConfig, table_id: e.target.value })}
            sx={{ mb: 2 }}
          />
          <TextField
            margin="dense"
            label="Model Name"
            fullWidth
            variant="outlined"
            value={trainConfig.model_name}
            onChange={(e) => setTrainConfig({ ...trainConfig, model_name: e.target.value })}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTrainDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleTrainModel} variant="contained">
            Start Training
          </Button>
        </DialogActions>
      </Dialog>
    </div>
  );
};

export default ModelManagement;