import React, { useState, useEffect, Suspense, lazy } from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { Loader2 } from 'lucide-react';

// --- Global Styles ---
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import './index.css';

// --- Auth API ---
import { getCurrentUser } from './api/authClient';

// --- Static Imports (Shell) ---
import Layout from './components/Layout';
import LoginPage from './components/LoginPage';

// --- LAZY IMPORTS (Code Splitting) ---
// Dashboard & Core
const DashboardHome = lazy(() => import('./components/DashboardHome'));
const GoalTrackerPage = lazy(() => import('./components/GoalTrackerPage'));

// Jobs & Applications
const JobLibrary = lazy(() => import('./components/JobLibrary'));
const JobDetails = lazy(() => import('./components/JobDetails'));
const AppTrackerPage = lazy(() => import('./components/AppTrackerPage'));
const ApplicationDashboard = lazy(() => import('./components/applications/ApplicationDashboard'));
const RoleCasePage = lazy(() => import('./components/RoleCasePage'));
const SupportingDocStudio = lazy(() => import('./components/applications/SupportingDocStudio'));

// CV Management
const CVLibraryPage = lazy(() => import('./components/CVLibraryPage'));
const CVWorkspace = lazy(() => import('./components/cv/CVWorkspace'));
const CVDashboard = lazy(() => import('./components/cv/CVDashboard'));
const CVSectionWrapper = lazy(() => import('./components/cv/CVSectionWrapper'));
const QuickCVEditor = lazy(() => import('./components/cv/QuickCVEditor'));

// CV Sections (Managers)
// Note: We lazy load these too so opening the "Experience" tab doesn't load "Education" code yet.
const ExperienceManager = lazy(() => import('./components/cv/ExperienceManager'));
const EducationManager = lazy(() => import('./components/cv/EducationManager'));
const ProjectManager = lazy(() => import('./components/cv/ProjectManager'));
const SkillsetManager = lazy(() => import('./components/cv/SkillsetManager'));
const AchievementHub = lazy(() => import('./components/cv/AchievementHub'));
const HobbyManager = lazy(() => import('./components/cv/HobbyManager'));


// --- Loading Fallback Component ---
const PageLoader = () => (
  <div className="vh-100 w-100 d-flex flex-column align-items-center justify-content-center bg-light text-muted">
    <Loader2 className="animate-spin mb-3 text-primary" size={40} />
    <span className="small fw-medium text-uppercase tracking-wide">Loading Workspace...</span>
  </div>
);

function App() {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  
  const location = useLocation();

  useEffect(() => {
    getCurrentUser()
        .then(response => setUser(response.data))
        .catch(() => setUser(null))
        .finally(() => setAuthLoading(false));
  }, []);

  // 1. Initial Auth Check (Blocks everything until we know who you are)
  if (authLoading) {
    return <PageLoader />;
  }

  // 2. Unauthenticated State
  if (!user) return <LoginPage />;

  return (
    // 3. Suspense Boundary: Catches the "loading" state of any Lazy component below
    <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<DashboardHome />} />
          
          {/* --- CV ROUTES --- */}
          <Route path="cvs" element={<CVLibraryPage />} />

          {/* Quick Edit / Preview Route */}
          <Route path="cv/:cvId/quick-edit" element={<QuickCVEditor />} />

          <Route path="cv/:cvId" element={<CVWorkspace />}>
              <Route index element={<CVDashboard />} />
              <Route path="experience" element={<CVSectionWrapper component={ExperienceManager} section="experiences" />} />
              <Route path="education" element={<CVSectionWrapper component={EducationManager} section="education" />} />
              <Route path="projects" element={<CVSectionWrapper component={ProjectManager} section="projects" />} />
              <Route path="skills" element={<CVSectionWrapper component={SkillsetManager} section="skills" />} />
              <Route path="achievements" element={<CVSectionWrapper component={AchievementHub} section="achievements" />} />
              <Route path="hobbies" element={<CVSectionWrapper component={HobbyManager} section="hobbies" />} />
          </Route>
          
          {/* --- JOB ROUTES --- */}
          <Route path="jobs" element={<JobLibrary />} />
          <Route path="/job/:jobId" element={<JobDetails />} />
          
          {/* --- APPLICATION ROUTES --- */}
          <Route path="applications" element={<AppTrackerPage />} />
          <Route path="application/:applicationId" element={<ApplicationDashboard />} />
          <Route path="application/:applicationId/rolecase" element={<RoleCasePage />} />
          
          {/* Tailored CV Workspace */}
          <Route path="application/:applicationId/tailored-cv" element={<QuickCVEditor />} />
          
          <Route path="application/:applicationId/doc/:documentId" element={<SupportingDocStudio />} />
          
          <Route path="goals" element={<GoalTrackerPage />} />
        </Route>
      </Routes>
    </Suspense>
  );
}

export default App;