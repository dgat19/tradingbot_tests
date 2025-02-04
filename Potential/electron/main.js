const { app, BrowserWindow } = require('electron');
const path = require('path');
const Sentry = require('@sentry/electron');

Sentry.init({
  dsn: 'YOUR_SENTRY_DSN', // Replace with your actual DSN
});


function createWindow() {
  // Create the browser window.
  const win = new BrowserWindow({
    width: 1280,
    height: 800,
    webPreferences: {
      nodeIntegration: false, // Disable Node integration in the renderer process
      contextIsolation: true, // Enable context isolation
      preload: path.join(__dirname, 'preload.js'), // Preload script
    },
  });

  // Load the React app.
  win.loadURL('http://localhost:3000');

  // Open DevTools if needed
  // win.webContents.openDevTools();
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});
