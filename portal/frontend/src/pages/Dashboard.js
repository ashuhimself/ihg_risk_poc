import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer
} from 'recharts';
import apiService from '../services/apiService';

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [dashboardData, setDashboardData] = useState({
    systemHealth: null,
    modelPerformance: null,
    dataStats: null
  });

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [systemHealth, modelPerformance, dataStats] = await Promise.all([
        apiService.getSystemHealth(),
        apiService.getModelPerformance(),
        apiService.getDataStats()
      ]);

      setDashboardData({
        systemHealth,
        modelPerformance,
        dataStats
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">Error loading dashboard: {error}</Alert>;
  }

  // Mock data for charts
  const performanceData = [
    { month: 'Jan', accuracy: 0.85, predictions: 1200 },
    { month: 'Feb', accuracy: 0.87, predictions: 1400 },
    { month: 'Mar', accuracy: 0.86, predictions: 1300 },
    { month: 'Apr', accuracy: 0.88, predictions: 1600 },
    { month: 'May', accuracy: 0.85, predictions: 1500 }
  ];

  const systemStatusData = [
    { name: 'Healthy', value: 95, color: '#4caf50' },
    { name: 'Warning', value: 4, color: '#ff9800' },
    { name: 'Error', value: 1, color: '#f44336' }
  ];

  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* System Health Overview */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health
              </Typography>
              <Box display="flex" alignItems="center">
                <Box
                  width={12}
                  height={12}
                  borderRadius="50%"
                  bgcolor="#4caf50"
                  mr={1}
                />
                <Typography variant="body1">
                  {dashboardData.systemHealth?.overall_status || 'Healthy'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Active Models */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Models
              </Typography>
              <Typography variant="h3" color="primary">
                3
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Daily Predictions */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Daily Predictions
              </Typography>
              <Typography variant="h3" color="primary">
                1,247
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Model Accuracy */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Accuracy
              </Typography>
              <Typography variant="h3" color="primary">
                87.3%
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Performance Trend
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={performanceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#8884d8"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* System Status */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={systemStatusData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    label
                  >
                    {systemStatusData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Activity
              </Typography>
              <Box>
                <Typography variant="body2" color="text.secondary">
                  • Model training completed successfully (2 hours ago)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • New data batch processed (4 hours ago)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • Endpoint deployment updated (6 hours ago)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  • System health check passed (8 hours ago)
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};

export default Dashboard;