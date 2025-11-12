import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.bundle.min.js'; 
import './index.css'; 

import { fetchAllCVs } from './api/cvClient';

// Components
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

    // Navigation State for Deep Linking
    const [targetCVSection, setTargetCVSection] = useState(null);

    const loadCoreData = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchAllCVs();
            setCvs(data);
            if (data.length > 0) setDefaultCvId(data[0].id);
        } catch (err) {
            setError('Failed to load core data.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadCoreData();
    }, []);

    const ActiveComponent = views[activeView];

    const handleNavigateToWorkspace = (appId) => {
        setActiveWorkspaceId(appId);
    };

    const handleExitWorkspace = () => {
        setActiveWorkspaceId(null);
        setActiveView('Application_Tracker');
    };

    // --- NEW: Deep Link Handler ---
    const handleNavigateToCVSection = (sectionName) => {
        setTargetCVSection(sectionName);
        setActiveView('CV_Manager');
    };

    return (
        <Layout activeView={activeView} setActiveView={setActiveView}>
            {loading ? (
                <div className="d-flex justify-content-center align-items-center" style={{ height: '50vh' }}>
                    <div className="spinner-border text-primary" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                </div>
            ) : error ? (
                <div className="alert alert-danger" role="alert">
                    {error}
                </div>
            ) : activeWorkspaceId ? (
                <ApplicationWorkspace
                    key={activeWorkspaceId}
                    applicationId={activeWorkspaceId}
                    onExitWorkspace={handleExitWorkspace}
                />
            ) : (
                <ActiveComponent
                    cvs={cvs}
                    setActiveView={setActiveView}
                    reloadData={loadCoreData}
                    defaultCvId={defaultCvId}
                    onNavigateToWorkspace={handleNavigateToWorkspace}
                    
                    // Pass deep linking props
                    initialSection={targetCVSection} 
                    onNavigateToCVSection={handleNavigateToCVSection}
                />
            )}
        </Layout>
    );
}

export default App;