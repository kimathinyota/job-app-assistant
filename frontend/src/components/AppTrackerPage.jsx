// frontend/src/components/AppTrackerPage.jsx
import React, { useState } from 'react';
import ApplicationsView from './applications/ApplicationsView';
import SavedJobsView from './applications/SavedJobsView';

// App.jsx passes down these props:
// - defaultCvId: The CV to use when starting a new application
// - onNavigateToWorkspace: The function to call to open the wizard
const AppTrackerPage = ({ defaultCvId, onNavigateToWorkspace }) => {
    const [activeTab, setActiveTab] = useState('applications');

    return (
        <div className="text-start">
            <h2 className="h3">Application Suite</h2>
            
            {/* Bootstrap Nav Pills for the two tabs */}
            <ul className="nav nav-pills mb-3" id="app-suite-tabs" role="tablist">
                <li className="nav-item" role="presentation">
                    <button 
                        className={`nav-link ${activeTab === 'applications' ? 'active' : ''}`}
                        id="apps-tab"
                        onClick={() => setActiveTab('applications')}
                        type="button"
                        role="tab"
                    >
                        🚀 My Applications
                    </button>
                </li>
                <li className="nav-item" role="presentation">
                    <button 
                        className={`nav-link ${activeTab === 'jobs' ? 'active' : ''}`}
                        id="jobs-tab"
                        onClick={() => setActiveTab('jobs')}
                        type="button"
                        role="tab"
                    >
                        🗂️ My Saved Jobs
                    </button>
                </li>
            </ul>

            {/* Conditionally render the content */}
            <div className="tab-content">
                <div className={`tab-pane fade ${activeTab === 'applications' ? 'show active' : ''}`}>
                    {activeTab === 'applications' && (
                        <ApplicationsView 
                            onNavigateToWorkspace={onNavigateToWorkspace} 
                        />
                    )}
                </div>
                <div className={`tab-pane fade ${activeTab === 'jobs' ? 'show active' : ''}`}>
                     {activeTab === 'jobs' && (
                        <SavedJobsView 
                            defaultCvId={defaultCvId}
                            onNavigateToWorkspace={onNavigateToWorkspace} 
                        />
                    )}
                </div>
            </div>
        </div>
    );
};

export default AppTrackerPage;