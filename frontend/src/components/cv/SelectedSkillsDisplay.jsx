import React from 'react';

const SelectedSkillsDisplay = ({ allSkills, selectedSkillIds, pendingSkills }) => (
  <div style={{ marginTop: '10px', display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
    {selectedSkillIds.map(id => {
      const skill = allSkills.find(s => s.id === id);
      return (
        <span key={id} style={{
          backgroundColor: '#e2e6ea',
          color: '#212529',
          padding: '4px 8px',
          borderRadius: '15px',
          fontSize: '0.85em'
        }}>
          {skill ? skill.name : 'Unknown'}
        </span>
      );
    })}
    {pendingSkills.map((s, i) => (
      <span key={i} style={{
        backgroundColor: '#d4edda',
        color: '#155724',
        padding: '4px 8px',
        borderRadius: '15px',
        fontSize: '0.85em'
      }}>
        +{s.name}
      </span>
    ))}
  </div>
);

export default SelectedSkillsDisplay;
