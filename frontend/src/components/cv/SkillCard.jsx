// frontend/src/components/cv/SkillCard.jsx
import React from 'react';
import SkillRelationshipDisplay from './SkillRelationshipDisplay'; // We will create this

const SkillCard = ({ skill, relationships, onEdit, onDelete }) => {
  const hasRelationships = 
      (relationships?.experiences?.size > 0) ||
      (relationships?.education?.size > 0) ||
      (relationships?.projects?.size > 0) ||
      (relationships?.hobbies?.size > 0) ||
      (relationships?.achievements?.size > 0);

  return (
    <div className="card h-100 shadow-sm">
      <div className="card-body">
        {/* Card Header */}
        <div className="d-flex justify-content-between align-items-start">
          <div>
            <h5 className="card-title mb-1">{skill.name}</h5>
            <span className="badge bg-primary me-2 text-capitalize">{skill.category}</span>
            {skill.level && <span className="badge bg-secondary me-2">{skill.level}</span>}
            {skill.importance && <span className="badge bg-info text-dark">Importance: {skill.importance}/5</span>}
          </div>
          {/* Edit/Delete Dropdown */}
          <div className="dropdown">
            <button className="btn btn-sm btn-outline-secondary" type="button" data-bs-toggle="dropdown" aria-expanded="false">
              ...
            </button>
            <ul className="dropdown-menu dropdown-menu-end">
              <li><button className="dropdown-item" onClick={onEdit}>Edit</button></li>
              <li><button className="dropdown-item text-danger" onClick={onDelete}>Delete</button></li>
            </ul>
          </div>
        </div>

        {/* Description */}
        {skill.description && (
          <p className="card-text mt-2 mb-3 fst-italic">{skill.description}</p>
        )}

        {/* Relationships */}
        <div className="mt-3 border-top pt-3">
          <h6 className="card-subtitle mb-2 text-muted">Related To:</h6>
          {hasRelationships ? (
            <SkillRelationshipDisplay relationships={relationships} />
          ) : (
            <p className="small text-muted mb-0">Not linked to any items yet.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default SkillCard;