import React from 'react';

const AchievementLinker = ({
  allAchievements = [],
  selectedAchievementIds = [],
  setSelectedAchievementIds,
}) => {
  const handleToggleAchievement = (achId) => {
    if (selectedAchievementIds.includes(achId)) {
      setSelectedAchievementIds(selectedAchievementIds.filter((id) => id !== achId));
    } else {
      setSelectedAchievementIds([...selectedAchievementIds, achId]);
    }
  };

  return (
    <div
      style={{
        marginTop: '15px',
        padding: '15px',
        border: '2px solid #6c757d',
        borderRadius: '8px',
        backgroundColor: '#f8f9fa',
      }}
    >
      <strong style={{ display: 'block', marginBottom: '8px', fontSize: '1.1em', color: '#6c757d' }}>
        Link Existing Achievements:
      </strong>
      <p style={{ fontSize: '0.8em', color: '#666', marginTop: 0 }}>
        {allAchievements.length} master achievements available.
      </p>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px',
          maxHeight: '150px',
          overflowY: 'auto',
          border: '1px solid #eee',
          padding: '5px',
        }}
      >
        {allAchievements.map((ach) => (
          <label
            key={ach.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              fontSize: '0.9em',
              padding: '4px 8px',
              border: '1px solid #ddd',
              borderRadius: '20px',
              cursor: 'pointer',
              backgroundColor: selectedAchievementIds.includes(ach.id)
                ? '#d6d8db'
                : 'white',
            }}
          >
            <input
              type="checkbox"
              checked={selectedAchievementIds.includes(ach.id)}
              onChange={() => handleToggleAchievement(ach.id)}
              style={{ marginRight: '5px' }}
            />
            {ach.text.length > 50 ? ach.text.substring(0, 50) + '...' : ach.text}
          </label>
        ))}
      </div>
    </div>
  );
};

export default AchievementLinker;
