// frontend/src/App.jsx
import React from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'; 
import './index.css'; 

// Components
import Layout from './components/Layout';
import DashboardHome from './components/DashboardHome';
import CVManagerPage from './components/CVManagerPage';
import AppTrackerPage from './components/AppTrackerPage';
import GoalTrackerPage from './components/GoalTrackerPage';
import ApplicationWorkspace from './components/applications/ApplicationWorkspace';
import ApplicationDashboard from './components/applications/ApplicationDashboard';
// This component wraps pages that need the standard container
const PageWrapper = ({ children }) => (
  <div className="container-xxl py-4">
    {children}
  </div>
);

function App() {
  const location = useLocation();
  const cvState = location.state || {};

  return (
    <Layout>
      <Routes>
        <Route 
          path="/" 
          element={<PageWrapper><DashboardHome /></PageWrapper>} 
        />
        <Route 
          path="/applications" 
          element={<PageWrapper><AppTrackerPage /></PageWrapper>} 
        />
        <Route 
          path="/applications/:applicationId" 
          element={<ApplicationDashboard />} 
        />
        
        {/* --- UPDATED CV ROUTES --- */}
        {/* This route handles the base /cv URL */}
        <Route 
          path="/cv" 
          element={
            <PageWrapper>
              <CVManagerPage key={location.pathname} initialSection={cvState.initialSection} />
            </PageWrapper>
          } 
        />
        {/* This new route handles /cv/cv_id */}
        <Route 
          path="/cv/:cvId" 
          element={
            <PageWrapper>
              <CVManagerPage key={location.pathname} initialSection={cvState.initialSection} />
            </PageWrapper>
          } 
        />
        {/* --- END OF UPDATE --- */}

        <Route 
          path="/goals" 
          element={<PageWrapper><GoalTrackerPage /></PageWrapper>} 
        />
      </Routes>
    </Layout>
  );
}

export default App;