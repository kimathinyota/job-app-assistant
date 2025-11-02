// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { fetchAllCVs } from './api/cvClient'; // This is fine
import './App.css'; 

// --- Import Core Components ---
import NavMenu from './components/NavMenu'; 
import DashboardHome from './components/DashboardHome'; 
import CVManagerPage from './components/CVManagerPage';
import AppTrackerPage from './components/AppTrackerPage'; // This is now the tab container
import GoalTrackerPage from './components/GoalTrackerPage'; 
import ApplicationWorkspace from './components/applications/ApplicationWorkspace'; // --- NEW IMPORT ---

// Define the available main views
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

    // --- NEW STATE for Workspace Navigation ---
    const [activeWorkspaceId, setActiveWorkspaceId] = useState(null); // e.g., 'app_123'
    const [defaultCvId, setDefaultCvId] = useState(null); // The CV to use for new apps

    const loadCoreData = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchAllCVs();
            setCvs(data);
            // --- NEW: Set a default CV ---
            if (data.length > 0) {
                setDefaultCvId(data[0].id);
            }
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

    // --- NEW: Handlers for navigation ---
    const handleNavigateToWorkspace = (applicationId) => {
        setActiveWorkspaceId(applicationId);
    };

    const handleExitWorkspace = () => {
        setActiveWorkspaceId(null);
        setActiveView('Application_Tracker'); // Go back to the app list
        // We could also reload data here
    };

    return (
        <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto', textAlign: 'center', backgroundColor: '#f9f9f9', minHeight: '100vh' }}>
            <h1 style={{ fontSize: '2.5em', borderBottom: '1px solid #ddd', paddingBottom: '10px', color: '#333' }}>
                Job Application Assistant
            </h1>
            
            {/* Hide main nav when in workspace */}
            {!activeWorkspaceId && (
                <NavMenu activeView={activeView} setActiveView={setActiveView} />
            )}

            <main>
                {loading ? (
                    <p style={{ fontSize: '1.5em', color: '#007bff' }}>Loading initial data...</p>
                ) : error ? (
                    <p style={{ color: 'red', fontWeight: 'bold' }}>Error: {error}</p>
                ) : activeWorkspaceId ? (
                    // --- NEW: Render Workspace ---
                    <ApplicationWorkspace
                        key={activeWorkspaceId} // Re-mounts component on ID change
                        applicationId={activeWorkspaceId}
                        onExitWorkspace={handleExitWorkspace}
                    />
                ) : (
                    // --- Original View Logic ---
                    <ActiveComponent 
                        cvs={cvs} 
                        setActiveView={setActiveView} 
                        reloadData={loadCoreData} 
                        // --- NEW PROPS ---
                        defaultCvId={defaultCvId}
                        onNavigateToWorkspace={handleNavigateToWorkspace}
                    />
                )}
            </main>
        </div>
    );
}

export default App;