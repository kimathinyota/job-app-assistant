// frontend/src/components/cv/SelectedSkillsDisplay.jsx
import React from 'react';

const SelectedSkillsDisplay = ({ allSkills, selectedSkillIds, pendingSkills }) => (
  <div className="d-flex flex-wrap gap-2 mt-2">
    {selectedSkillIds.map(id => {
      const skill = allSkills.find(s => s.id === id);
      return (
        <span 
            key={id} 
            className="px-2 py-1 rounded-pill bg-slate-100 text-slate-700 border border-slate-200 fw-medium"
            style={{ fontSize: '0.85em' }}
        >
          {skill ? skill.name : 'Unknown'}
        </span>
      );
    })}
    {pendingSkills.map((s, i) => (
      <span 
        key={i} 
        className="px-2 py-1 rounded-pill bg-emerald-50 text-emerald-700 border border-emerald-200 fw-medium"
        style={{ fontSize: '0.85em' }}
      >
        +{s.name}
      </span>
    ))}
  </div>
);

export default SelectedSkillsDisplay;