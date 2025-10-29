// frontend/src/components/cv/AchievementDisplayGrid.jsx
import React from 'react';

const AchievementDisplayGrid = ({
  achievementsToDisplay = [], // 👈 **CHANGED:** This is now the primary prop
  allSkills = [],
  onRemoveAchievement = () => {},
  onEditAchievement = () => {},
  isDisplayOnly = false // 👈 **NEW:** To hide buttons
}) => {
  // 💡 **CHANGED:** Check the new prop
  if (!achievementsToDisplay || achievementsToDisplay.length === 0) {
    return <p style={{ color: '#666', fontStyle: 'italic' }}>No achievements to display.</p>;
  }

  // Helper: resolve skill name by ID
  const getSkillName = (id) => {
    const skill = allSkills.find((s) => s.id === id);
    return skill ? skill.name : id; // fallback to id if not found
  };

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '10px',
        marginTop: '10px'
      }}
    >
      {/* 💡 **CHANGED:** Map over achievementsToDisplay */}
      {achievementsToDisplay.map((ach, index) => (
        <div
          key={ach.id || index}
          style={{
            position: 'relative',
            backgroundColor: '#ffffff',
            border: '1px solid #ccc',
            borderRadius: '12px',
            padding: '12px 15px',
            minWidth: '250px',
            maxWidth: '300px',
            boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
            flex: '1 1 calc(50% - 10px)',
            color: '#333'
          }}
        >
          {/* 💡 **CHANGED:** Conditionally show buttons */}
          {!isDisplayOnly && (
            <>
              {/* X button to remove */}
              <button
                type="button"
                onClick={() => onRemoveAchievement(ach.id || index)}
                style={{
                  position: 'absolute',
                  top: '5px',
                  right: '8px',
                  background: 'none',
                  border: 'none',
                  color: 'red',
                  fontWeight: 'bold',
                  cursor: 'pointer'
                }}
                title="Remove Achievement"
              >
                ✕
              </button>

              {/* Edit button */}
              <button
                type="button"
                onClick={() => onEditAchievement(index)}
                style={{
                  position: 'absolute',
                  top: '5px',
                  right: '30px',
                  background: 'none',
                  border: 'none',
                  color: '#007bff',
                  cursor: 'pointer'
                }}
                title="Edit Achievement"
              >
                ✎
              </button>
            </>
          )}

          <p style={{ marginTop: '0', marginBottom: '8px', fontWeight: 'bold' }}>
            {/* This handles both pending (new) and existing (linked) achievements */}
            {ach.text}
          </p>

          {/* Show related skill tags */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
            {(ach.skill_ids || []).map((id) => (
              <span
                key={id}
                style={{
                  backgroundColor: '#d1ecf1',
                  color: '#0c5460',
                  padding: '3px 8px',
                  borderRadius: '10px',
                  fontSize: '0.8em'
                }}
              >
                {getSkillName(id)}
              </span>
            ))}

            {/* This part for pending skills (e.g., in the modal) */}
            {(ach.new_skills || []).map((s, i) => (
              <span
                key={i}
                style={{
                  backgroundColor: '#fefefe',
                  border: '1px dashed #28a745',
                  color: '#155724',
                  padding: '3px 8px',
                  borderRadius: '10px',
                  fontSize: '0.8em'
                }}
              >
                {s.name} ({s.category})
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default AchievementDisplayGrid;