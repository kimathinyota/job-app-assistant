// frontend/src/components/applications/JobCard.jsx
import React, { useState } from 'react';

const JobCard = ({ 
    job, 
    cvs = [], 
    defaultCvId, 
    application, // This is the full application object or undefined
    onStartApplication, 
    onEdit 
}) => {

    const hasApplication = Boolean(application);

    // This state is only used for the dropdown *before* an app is created
    const [selectedCvId, setSelectedCvId] = useState(
        hasApplication ? application.base_cv_id : defaultCvId
    );

    const handleStartClick = () => {
        if (selectedCvId) {
            onStartApplication(job.id, selectedCvId);
        }
    };
    
    // Helper to find the CV name for display
    const getSelectedCvName = () => {
        if (!application) return "Unknown CV";
        const foundCv = cvs.find(cv => cv.id === application.base_cv_id);
        // Use cv.name for the display name
        return foundCv ? foundCv.name : "Unknown CV"; 
    };

    return (
        // Use card, shadow-sm for depth, and mb-3 for spacing
        <div className="card shadow-sm mb-3 text-start">
            
            {/* Card Header: Contains title, company, and edit button */}
            <div className="card-header bg-white p-3">
                <div className="d-flex justify-content-between align-items-start">
                    <div>
                        <h5 className="card-title mb-0">{job.title}</h5>
                        <small className="text-muted">{job.company}</small>
                    </div>
                    <button
                        className="btn btn-outline-secondary btn-sm"
                        onClick={onEdit}
                        disabled={hasApplication}
                        title={hasApplication ? "Cannot edit a job with an active application" : "Edit Job Details"}
                    >
                        Edit Job
                    </button>
                </div>
            </div>

            {/* Card Body: Contains the CV selector and action button */}
            <div className="card-body p-3">
                <div className="d-flex justify-content-between align-items-end">
                    
                    {/* Column 1: CV Selector or CV "Tag" */}
                    <div className="flex-grow-1 me-3">
                        {hasApplication ? (
                            // If application exists, show a "tag"
                            <div>
                                <label className="form-label small mb-1">
                                    Selected CV:
                                </label>
                                <div 
                                    className="badge bg-light text-dark border p-2 text-start w-100" 
                                    style={{ fontWeight: '500', fontSize: '0.9rem' }}
                                >
                                    {getSelectedCvName()}
                                </div>
                            </div>
                        ) : (
                            // If no application, show the selector
                            <div>
                                <label htmlFor={`cv-select-${job.id}`} className="form-label small mb-1">
                                    Select CV:
                                </label>
                                <select
                                    id={`cv-select-${job.id}`}
                                    className="form-select form-select-sm"
                                    value={selectedCvId || ''} 
                                    onChange={(e) => setSelectedCvId(e.target.value)}
                                    disabled={cvs.length === 0}
                                >
                                    <option value="" disabled>
                                        {cvs.length === 0 ? "No CVs found" : "-- Select a CV --"}
                                    </option>
                                    {cvs.map(cv => (
                                        <option key={cv.id} value={cv.id}>
                                            {cv.name}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        )}
                    </div>
                    
                    {/* Column 2: Action Button or Status Badge */}
                    <div className="flex-shrink-0">
                        {hasApplication ? (
                            <span className="badge bg-success p-2 fs-6">
                                Application Started
                            </span>
                        ) : (
                            <button 
                                className="btn btn-primary"
                                onClick={handleStartClick}
                                disabled={!selectedCvId}
                                style={{ minWidth: '160px' }} // Give button a consistent width
                            >
                                Start Application
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default JobCard;