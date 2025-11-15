// frontend/src/components/AppTrackerPage.jsx
import React, { useState, useEffect } from 'react'; // <-- Added useEffect
import { useNavigate } from 'react-router-dom';   // <-- Added useNavigate
import { Rocket, Briefcase } from 'lucide-react'; 
import ApplicationsView from './applications/ApplicationsView';
import SavedJobsView from './applications/SavedJobsView';
import { fetchAllCVs } from '../api/cvClient'; // <-- Added cvClient import

// 1. Removed all props
const AppTrackerPage = () => {
    const [activeTab, setActiveTab] = useState('applications');
    
    // 2. Added state and hooks to make the component self-contained
    const [defaultCvId, setDefaultCvId] = useState(null);
    const navigate = useNavigate();

    // 3. Fetch data that was previously passed as props
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

    // 4. Re-create the navigation function locally
    const handleNavigateToWorkspace = (appId) => {
        navigate(`/applications/${appId}`);
    };

    // 5. The entire return block (UI) is identical to your original file.
    return (
        <div className="text-start h-100 d-flex flex-column">
            <style>
                {`
                .nav-segment {
                    background-color: #f1f5f9; /* Slate-100 */
                    padding: 4px;
                    border-radius: 12px;
                    display: inline-flex;
                    position: relative;
                }
                .nav-segment .nav-link {
                    border-radius: 8px;
                    padding: 8px 20px;
                    color: #64748b; /* Slate-500 */
                    font-weight: 500;
                    font-size: 0.9rem;
                    transition: all 0.2s ease;
                    border: none;
                }
                .nav-segment .nav-link:hover {
                    color: #0f172a;
                }
                .nav-segment .nav-link.active {
                    background-color: #ffffff;
                    color: #0f172a; /* Slate-900 */
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                    font-weight: 600;
                }
                /* Dark mode support */
                [data-bs-theme="dark"] .nav-segment {
                    background-color: #1e293b;
                }
                [data-bs-theme="dark"] .nav-segment .nav-link {
                    color: #94a3b8;
                }
                [data-bs-theme="dark"] .nav-segment .nav-link.active {
                    background-color: #0f172a;
                    color: #f8fafc;
                }
                `}
            </style>

            <div className="d-flex justify-content-between align-items-center mb-4">
                <div>
                    {/* Optional Title */}
                </div>
                
                {/* Modern Segmented Control */}
                <div className="nav-segment shadow-sm mx-auto mx-md-0">
                    <button 
                        className={`nav-link d-flex align-items-center gap-2 ${activeTab === 'applications' ? 'active' : ''}`}
                        onClick={() => setActiveTab('applications')}
                    >
                        <Rocket size={16} className={activeTab === 'applications' ? 'text-primary' : ''}/>
                        My Applications
                    </button>
                    <button 
                        className={`nav-link d-flex align-items-center gap-2 ${activeTab === 'jobs' ? 'active' : ''}`}
                        onClick={() => setActiveTab('jobs')}
                    >
                        <Briefcase size={16} className={activeTab === 'jobs' ? 'text-primary' : ''}/>
                        Saved Jobs
                    </button>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-grow-1 animate-fade-in">
                {activeTab === 'applications' && (
                    <ApplicationsView 
                        // 6. Pass the locally created function
                        onNavigateToWorkspace={handleNavigateToWorkspace}
                        onTrackJob={() => setActiveTab('jobs')} 
                    />
                )}
                
                {activeTab === 'jobs' && (
                    <div className="animation-slide-up">
                        <SavedJobsView 
                            // 7. Pass the locally fetched state and created function
                            defaultCvId={defaultCvId}
                            onNavigateToWorkspace={handleNavigateToWorkspace} 
                        />
                    </div>
                )}
            </div>
        </div>
    );
};

export default AppTrackerPage;