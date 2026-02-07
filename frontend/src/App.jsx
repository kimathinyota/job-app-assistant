import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';

// --- Global Styles ---
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import './index.css';

// --- Auth API ---
import { getCurrentUser } from './api/authClient';

// --- Components ---
import Layout from './components/Layout';
import LoginPage from './components/LoginPage';
import DashboardHome from './components/DashboardHome';
import JobLibrary from './components/JobLibrary';
import GoalTrackerPage from './components/GoalTrackerPage';

// --- CV Components ---
import CVManagerPage from './components/CVManagerPage';

// --- Application & Tracker Components ---
import AppTrackerPage from './components/AppTrackerPage';
import ApplicationsView from './components/applications/ApplicationsView';
import ApplicationDashboard from './components/applications/ApplicationDashboard';

// --- Workspace Sub-Components ---
import RoleCasePage from './components/RoleCasePage'; // <--- NEW IMPORT
import TailoredCVManager from './components/applications/TailoredCVManager';
import SupportingDocStudio from './components/applications/SupportingDocStudio';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const location = useLocation();
  const cvState = location.state || {};

  useEffect(() => {
    getCurrentUser()
        .then(response => setUser(response.data))
        .catch(() => setUser(null))
        .finally(() => setLoading(false));
  }, []);

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

  if (!user) {
    return <LoginPage />;
  }

  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        {/* Dashboard */}
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardHome />} />
        
        {/* CV Manager */}
        <Route 
          path="cvs" 
          element={<CVManagerPage key="cv-base" initialSection={cvState.initialSection} />} 
        />
        <Route 
          path="cv/:cvId" 
          element={<CVManagerPage key="cv-id" initialSection={cvState.initialSection} />} 
        />
        
        {/* Job Library */}
        <Route path="jobs" element={<JobLibrary />} />
        
        {/* Application Views */}
        <Route path="applications" element={<AppTrackerPage />} />

        {/* --- APPLICATION WORKSPACE ROUTES --- */}
        {/* Root Dashboard */}
        <Route path="application/:applicationId" element={<ApplicationDashboard />} />
        
        {/* RoleCase Strategy Mapping (The new page) */}
        <Route path="application/:applicationId/mapping" element={<RoleCasePage />} />
        
        {/* Tailored CV */}
        <Route path="application/:applicationId/cv" element={<TailoredCVManager />} />
        
        {/* Supporting Documents */}
        <Route path="application/:applicationId/doc/:documentId" element={<SupportingDocStudio />} />
        
        {/* Goals */}
        <Route path="goals" element={<GoalTrackerPage />} />
      </Route>
    </Routes>
  );
}

export default App;