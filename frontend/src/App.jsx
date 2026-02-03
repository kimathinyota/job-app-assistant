import React from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'; 
import './index.css'; 

// --- Components ---
import Layout from './components/Layout';
import DashboardHome from './components/DashboardHome';
import CVManagerPage from './components/CVManagerPage';
import AppTrackerPage from './components/AppTrackerPage'; // The Kanban/List of Applications
import GoalTrackerPage from './components/GoalTrackerPage';
import JobLibrary from './components/JobLibrary'; 
import { RoleCasePage } from './components/RoleCasePage'; // Import the new page

// Applications & Jobs
// NOTE: JobLibrary is your new "Job Library" component
import ApplicationDashboard from './components/applications/ApplicationDashboard';
import MappingManager from './components/applications/MappingManager';
import TailoredCVManager from './components/applications/TailoredCVManager';
import SupportingDocStudio from './components/applications/SupportingDocStudio';

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
        {/* 1. HOME */}
        <Route 
          path="/" 
          element={<PageWrapper><DashboardHome /></PageWrapper>} 
        />

        {/* 2. JOB LIBRARY (New Separate Route) */}
        <Route 
          path="/jobs" 
          element={<PageWrapper><JobLibrary /></PageWrapper>} 
        />

        {/* 3. APPLICATION TRACKER (The Kanban Board) */}
        <Route 
          path="/applications" 
          element={<PageWrapper><AppTrackerPage /></PageWrapper>} 
        />

        {/* 4. SPECIFIC APPLICATION WORKSPACE (Deep Dive) */}
        {/* We use /application (singular) to match the navigation in JobLibrary */}
        <Route path="/application/:applicationId" element={<ApplicationDashboard />} />
        
        {/* Workspace Sub-routes */}
        <Route path="/application/:applicationId/mapping" element={<MappingManager />} />
        <Route path="/application/:applicationId/cv" element={<TailoredCVManager />} />
        <Route path="/application/:applicationId/doc/:documentId" element={<SupportingDocStudio />} />


        {/* 1. The Application Audit Route (Nested or Standalone) */}
       <Route path="/application/:appId/analysis" element={<RoleCasePage />} />

       {/* 2. The Draft/Scratch Route */}
       <Route path="/role-case" element={<RoleCasePage />} />
       
        {/* 5. CV MANAGER (Standard & Specific ID) */}
        <Route 
          path="/cv" 
          element={
            <PageWrapper>
              <CVManagerPage key="cv-base" initialSection={cvState.initialSection} />
            </PageWrapper>
          } 
        />
        <Route 
          path="/cv/:cvId" 
          element={
            <PageWrapper>
              <CVManagerPage key="cv-id" initialSection={cvState.initialSection} />
            </PageWrapper>
          } 
        />

        {/* 6. GOALS */}
        <Route 
          path="/goals" 
          element={<PageWrapper><GoalTrackerPage /></PageWrapper>} 
        />

      </Routes>
    </Layout>
  );
}

export default App;