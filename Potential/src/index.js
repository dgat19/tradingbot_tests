import * as Sentry from '@sentry/electron/renderer';
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import theme from './theme';
import { ThemeProvider } from '@mui/material/styles';
import Prediction from './components/Prediction';
import { AppBar, Toolbar, Typography, Container } from '@mui/material';
import TradingChart from './components/TradingChart';
import Sidebar from './components/Sidebar';

function Main() {
  return (
    <div>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6">Trading Bot Dashboard</Typography>
        </Toolbar>
      </AppBar>
      <Container maxWidth="lg">
        <div style={{ display: 'flex', marginTop: '20px' }}>
          <Sidebar />
          <TradingChart />
        </div>
      </Container>
    </div>
  );
}

export default Main;

Sentry.init({
  dsn: 'YOUR_SENTRY_DSN', // Replace with your actual DSN
});

ReactDOM.render(
  <React.StrictMode>
    <ThemeProvider theme={theme}>
      <Main />
    </ThemeProvider>
  </React.StrictMode>,
  document.getElementById('root')
);
