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
import LoginPage from './components/LoginPage'; // <--- Import the new page

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // --- 1. THE AUTH CHECK ---
  useEffect(() => {
    const checkAuth = async () => {
      try {
        // We use 'credentials: include' to send the HttpOnly cookie
        const response = await fetch("http://localhost:8000/auth/me", {
          credentials: "include" 
        });
        
        if (response.ok) {
          const userData = await response.json();
          setUser(userData);
        } else {
          setUser(null); // Not logged in
        }
      } catch (error) {
        console.error("Auth check failed:", error);
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    checkAuth();
  }, []);

  // --- 2. LOADING STATE ---
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

  // --- 3. UNAUTHENTICATED STATE ---
  if (!user) {
    return <LoginPage />;
  }

  // --- 4. AUTHENTICATED STATE (Your Normal App) ---
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