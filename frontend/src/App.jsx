// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import DashboardHome from './components/DashboardHome';
import CVManagerPage from './components/CVManagerPage';
import JobLibrary from './components/JobLibrary';
import AppTrackerPage from './components/AppTrackerPage';
import ApplicationsView from './components/applications/ApplicationsView';
import ApplicationDashboard from './components/applications/ApplicationDashboard';
import GoalTrackerPage from './components/GoalTrackerPage';
import LoginPage from './components/LoginPage'; 
import { getCurrentUser } from './api/authClient'; 

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // --- 1. THE AUTH CHECK ---
  useEffect(() => {
    getCurrentUser()
        .then(response => {
            setUser(response.data);
        })
        .catch(() => {
            setUser(null);
        })
        // 2. ADD THIS FINALLY BLOCK
        .finally(() => {
            setLoading(false); // <--- This turns off the spinner
        });
  }, []);

  // --- 3. LOADING STATE ---
  if (loading) {
    return (
      <div className="h-screen w-full flex items-center justify-center bg-slate-50 text-slate-400">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-4 border-slate-300 border-t-slate-600 rounded-full animate-spin"></div>
          <p>Loading your workspace...</p>
        </div>
      </div>
    );
  }

  // --- 4. UNAUTHENTICATED STATE ---
  if (!user) {
    return <LoginPage />;
  }

  // --- 5. AUTHENTICATED STATE ---
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardHome />} />
        
        <Route path="cvs" element={<CVManagerPage />} />
        
        <Route path="jobs" element={<JobLibrary />} />
        
        <Route path="applications" element={<ApplicationsView />} />
        <Route path="applications/:appId" element={<ApplicationDashboard />} />
        
        <Route path="tracker" element={<AppTrackerPage />} />
        <Route path="goals" element={<GoalTrackerPage />} />
      </Route>
    </Routes>
  );
}

export default App;