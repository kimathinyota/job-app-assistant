// frontend/src/components/AppTrackerPage.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import ApplicationsView from './applications/ApplicationsView';
import { fetchAllCVs } from '../api/cvClient';

const AppTrackerPage = () => {
    // 1. Keep the state for the fetched CV (Restored)
    const [defaultCvId, setDefaultCvId] = useState(null);
    const navigate = useNavigate();

    // 2. Keep the data fetching logic (Restored)
    useEffect(() => {
        const loadData = async () => {
            try {
                const cvs = await fetchAllCVs();
                if (cvs && cvs.length > 0) {
                    setDefaultCvId(cvs[0].id);
                }
            } catch (err) {
                console.error("Failed to load default CV ID:", err);
            }
        };
        loadData();
    }, []);

    // 3. Keep the navigation handler (Restored)
    const handleNavigateToWorkspace = (appId) => {
        navigate(`/application/${appId}`);
    };

    return (
        <div className="text-start h-100 d-flex flex-column">
            {/* 4. Tabs removed, but content preserved */}
            <div className="flex-grow-1 animate-fade-in">
                <ApplicationsView 
                    onNavigateToWorkspace={handleNavigateToWorkspace}
                    onTrackJob={() => navigate('/jobs')} // Redirects to full Job Library since tabs are gone
                    defaultCvId={defaultCvId} // Passed down as requested
                />
            </div>
        </div>
    );
};

export default AppTrackerPage;