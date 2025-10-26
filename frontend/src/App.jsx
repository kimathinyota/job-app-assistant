// frontend/src/App.jsx
import React, { useState, useEffect } from 'react';
import { fetchAllCVs } from './api/client';
import './App.css'; 

// --- Import Core Components ---
import NavMenu from './components/NavMenu'; 
import DashboardHome from './components/DashboardHome'; 
import CVManagerPage from './components/CVManagerPage';
import AppTrackerPage from './components/AppTrackerPage';
import GoalTrackerPage from './components/GoalTrackerPage'; 


// Define the available views mapping
const views = {
    'Dashboard': DashboardHome,
    'CV_Manager': CVManagerPage,
    'Application_Tracker': AppTrackerPage,
    'Goal_Tracker': GoalTrackerPage,
};


function App() {
    const [activeView, setActiveView] = useState('Dashboard');
    const [cvs, setCvs] = useState([]); // Keep base CV state high up
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Function to load data and refresh state (fetches base CVs for dashboard metric)
    const loadCoreData = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchAllCVs();
            setCvs(data);
        } catch (err) {
            setError('Failed to load core data. Ensure backend is running.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // Load data when the component mounts
    useEffect(() => {
        loadCoreData();
    }, []);

    const ActiveComponent = views[activeView];

    return (
        <div style={{ padding: '20px', maxWidth: '1400px', margin: '0 auto', textAlign: 'center', backgroundColor: '#f9f9f9', minHeight: '100vh' }}>
            <h1 style={{ fontSize: '2.5em', borderBottom: '1px solid #ddd', paddingBottom: '10px', color: '#333' }}>
                Job Application Assistant
            </h1>
            
            <NavMenu activeView={activeView} setActiveView={setActiveView} />

            <main style={{ marginTop: '30px', minHeight: '600px', backgroundColor: 'white', padding: '30px', borderRadius: '10px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                {loading ? (
                    <p style={{ fontSize: '1.5em', color: '#007bff' }}>Loading initial data...</p>
                ) : error ? (
                    <p style={{ color: 'red', fontWeight: 'bold' }}>Error: {error}</p>
                ) : (
                    // Render the active component and pass core data/state functions
                    <ActiveComponent cvs={cvs} setActiveView={setActiveView} reloadData={loadCoreData} />
                )}
            </main>
        </div>
    );
}

export default App;
