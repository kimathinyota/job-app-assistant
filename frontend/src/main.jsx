// frontend/src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
// --- (NEW) ---
import { BrowserRouter as Router } from 'react-router-dom';
import App from './App.jsx';

// Your existing imports
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'; 
import './index.css'; 

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* --- (NEW) --- */}
    <Router>
      <App />
    </Router>
  </React.StrictMode>,
);