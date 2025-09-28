import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box
} from '@mui/material';

// Placeholder components for the remaining pages
const DataExplorer = () => {
  return (
    <div>
      <Typography variant="h4" gutterBottom>
        Data Explorer
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Dataset Overview
              </Typography>
              <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography color="text.secondary">
                  Data exploration interface coming soon...
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};

const SystemMonitoring = () => {
  return (
    <div>
      <Typography variant="h4" gutterBottom>
        System Monitoring
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Health & Performance
              </Typography>
              <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                <Typography color="text.secondary">
                  Advanced monitoring dashboard coming soon...
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </div>
  );
};

export { DataExplorer, SystemMonitoring };