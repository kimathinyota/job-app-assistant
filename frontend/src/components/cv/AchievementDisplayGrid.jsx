// frontend/src/components/cv/AchievementDisplayGrid.jsx
import React from 'react';

const AchievementDisplayGrid = ({
  achievementsToDisplay = [],
  allSkills = [],
  onRemoveAchievement = () => {},
  onEditAchievement = () => {},
  isDisplayOnly = false 
}) => {
  if (!achievementsToDisplay || achievementsToDisplay.length === 0) {
    return <p className="text-muted fst-italic small mt-2">No achievements to display.</p>;
  }

  const getSkillName = (id) => {
    const skill = allSkills.find((s) => s.id === id);
    return skill ? skill.name : id;
  };

  return (
    // Use Bootstrap's grid system
    <div className="row g-3 mt-1">
      {achievementsToDisplay.map((ach, index) => (
        // Each item is a column
        <div key={ach.id || index} className="col-12 col-md-6">
            {/* Use a card for styling */}
            <div className="card h-100 shadow-sm">
                <div className="card-body position-relative">
                    {!isDisplayOnly && (
                        <>
                            <button
                                type="button"
                                onClick={() => onRemoveAchievement(ach.id || index)}
                                className="btn-close position-absolute"
                                style={{ top: '0.75rem', right: '0.75rem' }}
                                title="Remove Achievement"
                            ></button>
                            <button
                                type="button"
                                onClick={() => onEditAchievement(index)}
                                className="btn btn-sm btn-link position-absolute p-0"
                                style={{ top: '0.65rem', right: '2.2rem', textDecoration: 'none', fontSize: '1.1rem' }}
                                title="Edit Achievement"
                            >
                                ✎
                            </button>
                        </>
                    )}

                    <p className="card-text fw-medium mb-2" style={{ paddingRight: isDisplayOnly ? '0' : '2.5rem' }}>
                        {ach.text}
                    </p>

                    {/* Skill tags */}
                    <div className="d-flex flex-wrap gap-1">
                        {(ach.skill_ids || []).map((id) => (
                            <span
                                key={id}
                                className="badge text-bg-info-subtle text-info-emphasis fw-normal"
                            >
                                {getSkillName(id)}
                            </span>
                        ))}
                        {(ach.new_skills || []).map((s, i) => (
                            <span
                                key={i}
                                className="badge text-bg-success-subtle text-success-emphasis fw-normal border border-success-subtle"
                            >
                                +{s.name}
                            </span>
                        ))}
                    </div>
                </div>
            </div>
        </div>
      ))}
    </div>
  );
};

export default AchievementDisplayGrid;