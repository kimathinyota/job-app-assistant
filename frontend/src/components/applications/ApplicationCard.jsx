// frontend/src/components/applications/ApplicationCard.jsx
import React from 'react';

const ApplicationCard = ({ application, job, onClick }) => {
    return (
        <div className="card shadow-sm mb-2" onClick={onClick} style={{ cursor: 'pointer' }}>
            <div className="card-body p-3">
                <h6 className="card-title mb-1">{job.title || 'Loading...'}</h6>
                <p className="card-subtitle small text-muted">{job.company}</p>
            </div>
        </div>
    );
};

export default ApplicationCard;