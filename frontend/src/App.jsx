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

// --- CV Components (Refactored) ---
import CVLibraryPage from './components/CVLibraryPage'; // New List Page
import CVWorkspace from './components/cv/CVWorkspace';   // New Edit Layout
import CVDashboard from './components/cv/CVDashboard';   // New Dashboard Grid
import CVSectionWrapper from './components/cv/CVSectionWrapper'; // Adapter for sections

// --- Managers (Reused) ---
import ExperienceManager from './components/cv/ExperienceManager';
import EducationManager from './components/cv/EducationManager';
import ProjectManager from './components/cv/ProjectManager';
import SkillsetManager from './components/cv/SkillsetManager';
import AchievementHub from './components/cv/AchievementHub';
import HobbyManager from './components/cv/HobbyManager';

// --- Application & Tracker Components ---
import AppTrackerPage from './components/AppTrackerPage';
import ApplicationsView from './components/applications/ApplicationsView';
import ApplicationDashboard from './components/applications/ApplicationDashboard';
import RoleCasePage from './components/RoleCasePage'; 
import TailoredCVManager from './components/applications/TailoredCVManager';
import SupportingDocStudio from './components/applications/SupportingDocStudio';
import JobDetails from './components/JobDetails';

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  
  const location = useLocation();

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

  if (!user) return <LoginPage />;

  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<DashboardHome />} />
        
        {/* --- CV ROUTES REFACTORED --- */}
        
        {/* 1. CV Library (List View) */}
        <Route path="cvs" element={<CVLibraryPage />} />

        {/* 2. CV Workspace (Edit View) */}
        <Route path="cv/:cvId" element={<CVWorkspace />}>
            {/* Grid View */}
            <Route index element={<CVDashboard />} />
            
            {/* Individual Sections */}
            <Route 
              path="experience" 
              element={<CVSectionWrapper component={ExperienceManager} section="experiences" />} 
            />
            <Route 
              path="education" 
              element={<CVSectionWrapper component={EducationManager} section="education" />} 
            />
            <Route 
              path="projects" 
              element={<CVSectionWrapper component={ProjectManager} section="projects" />} 
            />
            <Route 
              path="skills" 
              element={<CVSectionWrapper component={SkillsetManager} section="skills" />} 
            />
            <Route 
              path="achievements" 
              element={<CVSectionWrapper component={AchievementHub} section="achievements" />} 
            />
            <Route 
              path="hobbies" 
              element={<CVSectionWrapper component={HobbyManager} section="hobbies" />} 
            />
        </Route>
        
        {/* Job Library */}
        <Route path="jobs" element={<JobLibrary />} />
        <Route path="/job/:jobId" element={<JobDetails />} />
        
        {/* Application Views */}
        <Route path="applications" element={<AppTrackerPage />} />

        {/* Application Workspace */}
        <Route path="application/:applicationId" element={<ApplicationDashboard />} />
        <Route path="application/:applicationId/mapping" element={<RoleCasePage />} />
        <Route path="application/:applicationId/cv" element={<TailoredCVManager />} />
        <Route path="application/:applicationId/doc/:documentId" element={<SupportingDocStudio />} />
        
        <Route path="goals" element={<GoalTrackerPage />} />
      </Route>
    </Routes>
  );
}

export default App;