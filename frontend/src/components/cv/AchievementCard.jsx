// frontend/src/components/cv/AchievementCard.jsx
import React from 'react';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementRelationshipDisplay from './AchievementRelationshipDisplay'; // We will create this

const AchievementCard = ({ achievement, allSkills, relationships, onDelete }) => {
  const hasRelationships = 
      (relationships?.experiences?.length > 0) ||
      (relationships?.education?.length > 0) ||
      (relationships?.projects?.length > 0) ||
      (relationships?.hobbies?.length > 0);

  const hasSkills = achievement.skill_ids && achievement.skill_ids.length > 0;

  return (
    <div className="card h-100 shadow-sm">
      <div className="card-body">
        {/* Card Header */}
        <div className="d-flex justify-content-between align-items-start">
          {/* Text */}
          <div className="pe-3">
            <p className="card-text fw-medium mb-1">{achievement.text}</p>
            {achievement.context && (
                <span className="badge bg-secondary">Context: {achievement.context}</span>
            )}
          </div>
          
          {/* Delete Button */}
          <button 
            className="btn btn-sm btn-outline-danger" 
            type="button"
            onClick={onDelete}
            title="Delete Master Achievement"
          >
            <i className="bi bi-trash-fill"></i>
          </button>
        </div>

        {/* Linked Skills */}
        {hasSkills && (
          <div className="mt-3 border-top pt-3">
            <h6 className="card-subtitle mb-2 text-muted">Linked Skills:</h6>
            <SelectedSkillsDisplay
              allSkills={allSkills}
              selectedSkillIds={achievement.skill_ids}
              pendingSkills={[]}
            />
          </div>
        )}

        {/* Relationships */}
        <div className="mt-3 border-top pt-3">
          <h6 className="card-subtitle mb-2 text-muted">Used In:</h6>
          {hasRelationships ? (
            <AchievementRelationshipDisplay relationships={relationships} />
          ) : (
            <p className="small text-muted mb-0">Not linked to any items yet.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default AchievementCard;