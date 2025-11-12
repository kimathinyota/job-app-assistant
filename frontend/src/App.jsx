import React, { useState, useEffect } from 'react';
import { fetchAllCVs } from './api/cvClient';
import './App.css'; 

// --- Layout & Components ---
import Layout from './components/Layout';
import DashboardHome from './components/DashboardHome';
import CVManagerPage from './components/CVManagerPage';
import AppTrackerPage from './components/AppTrackerPage';
import GoalTrackerPage from './components/GoalTrackerPage';
import ApplicationWorkspace from './components/applications/ApplicationWorkspace';

const views = {
    'Dashboard': DashboardHome,
    'CV_Manager': CVManagerPage,
    'Application_Tracker': AppTrackerPage,
    'Goal_Tracker': GoalTrackerPage,
};

function App() {
    const [activeView, setActiveView] = useState('Dashboard');
    const [cvs, setCvs] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Workspace State
    const [activeWorkspaceId, setActiveWorkspaceId] = useState(null);
    const [defaultCvId, setDefaultCvId] = useState(null);

    const loadCoreData = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchAllCVs();
            setCvs(data);
            if (data.length > 0) setDefaultCvId(data[0].id);
        } catch (err) {
            setError('Failed to load core data. Ensure backend is running.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadCoreData();
    }, []);

    const ActiveComponent = views[activeView];

    const handleNavigateToWorkspace = (applicationId) => {
        setActiveWorkspaceId(applicationId);
    };

    const handleExitWorkspace = () => {
        setActiveWorkspaceId(null);
        setActiveView('Application_Tracker');
    };

    // --- View Rendering ---
    
    // 1. Workspace View (Full Screen / Focus Mode)
    if (activeWorkspaceId) {
        return (
            <div className="min-h-screen bg-slate-50 dark:bg-slate-950">
                <ApplicationWorkspace
                    key={activeWorkspaceId}
                    applicationId={activeWorkspaceId}
                    onExitWorkspace={handleExitWorkspace}
                />
            </div>
        );
    }

    // 2. Standard Application View (Wrapped in Layout)
    return (
        <Layout activeView={activeView} setActiveView={setActiveView}>
            {loading ? (
                <div className="flex flex-col items-center justify-center min-h-[60vh]">
                    <div className="w-10 h-10 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                    <p className="text-slate-500 font-medium">Initializing RoleCraft...</p>
                </div>
            ) : error ? (
                <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-300">
                    <p className="font-semibold">Connection Error</p>
                    <p>{error}</p>
                </div>
            ) : (
                <ActiveComponent
                    cvs={cvs}
                    setActiveView={setActiveView}
                    reloadData={loadCoreData}
                    defaultCvId={defaultCvId}
                    onNavigateToWorkspace={handleNavigateToWorkspace}
                />
            )}
        </Layout>
    );
}

export default App;