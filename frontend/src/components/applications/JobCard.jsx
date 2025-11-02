// frontend/src/components/applications/JobCard.jsx
import React from 'react';

const JobCard = ({ job, hasApplication, onStartApplication }) => {
    return (
        <div className="list-group-item d-flex justify-content-between align-items-center">
            <div>
                <h6 className="mb-0">{job.title}</h6>
                <small className="text-muted">{job.company}</small>
            </div>
            <div>
                {hasApplication ? (
                    <span className="badge bg-success">Application Started</span>
                ) : (
                    <button 
                        className="btn btn-primary btn-sm"
                        onClick={onStartApplication}
                    >
                        Start Application
                    </button>
                )}
            </div>
        </div>
    );
};

export default JobCard;