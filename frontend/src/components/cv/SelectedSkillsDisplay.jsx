import React from 'react';

const SelectedSkillsDisplay = ({ allSkills, selectedSkillIds, pendingSkills }) => {
  const selected = allSkills.filter((s) => selectedSkillIds.includes(s.id));

  return (
    <div style={{ marginTop: '8px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
      {selected.map((s) => (
        <span
          key={s.id}
          style={{
            backgroundColor: '#007bff',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '12px',
            fontSize: '0.9em',
          }}
        >
          {s.name}
        </span>
      ))}
      {pendingSkills.map((s, i) => (
        <span
          key={i}
          style={{
            backgroundColor: '#28a745',
            color: 'white',
            padding: '4px 8px',
            borderRadius: '12px',
            fontSize: '0.9em',
          }}
        >
          {s.name}
        </span>
      ))}
    </div>
  );
};

export default SelectedSkillsDisplay;
