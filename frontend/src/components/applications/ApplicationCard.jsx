// frontend/src/components/applications/ApplicationCard.jsx
import React from 'react';

const ApplicationCard = ({ application, job, onClick, onDelete }) => {
  // <-- 1. Add onDelete prop

  // --- 2. Add new handler ---
  const handleDeleteClick = (e) => {
    e.stopPropagation(); // Stop it from triggering onClick
    onDelete();
  };

  return (
    <div
      className="card shadow-sm mb-2"
      onClick={onClick}
      style={{ cursor: 'pointer' }}
    >
      <div className="card-body p-3">
        {/* --- 3. Add delete button --- */}
        <button
          type="button"
          className="btn-close"
          onClick={handleDeleteClick}
          style={{ float: 'right', marginLeft: '10px' }}
          title="Delete Application"
        ></button>
        <h6 className="card-title mb-1">{job.title || 'Loading...'}</h6>
        <p className="card-subtitle small text-muted">{job.company}</p>
      </div>
    </div>
  );
};

export default ApplicationCard;