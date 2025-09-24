import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';

// Import pages
import Dashboard from './pages/Dashboard';
import ModelManagement from './pages/ModelManagement';
import { DataExplorer, SystemMonitoring } from './pages/PlaceholderPages';
import PredictionInterface from './pages/PredictionInterface';

// Import components
import Navigation from './components/Navigation';

// Create theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <div className="App">
          <AppBar position="static">
            <Toolbar>
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                IHG Risk Assessment Platform
              </Typography>
            </Toolbar>
          </AppBar>
          
          <Navigation />
          
          <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/data" element={<DataExplorer />} />
              <Route path="/predict" element={<PredictionInterface />} />
              <Route path="/monitoring" element={<SystemMonitoring />} />
            </Routes>
          </Container>
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;