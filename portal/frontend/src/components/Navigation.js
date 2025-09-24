import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const Navigation = () => {
  const location = useLocation();

  const navItems = [
    { label: 'Dashboard', path: '/' },
    { label: 'Models', path: '/models' },
    { label: 'Data Explorer', path: '/data' },
    { label: 'Predictions', path: '/predict' },
    { label: 'Monitoring', path: '/monitoring' },
  ];

  return (
    <AppBar position="static" color="default" elevation={1}>
      <Toolbar>
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              component={Link}
              to={item.path}
              color={location.pathname === item.path ? 'primary' : 'inherit'}
              variant={location.pathname === item.path ? 'outlined' : 'text'}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navigation;