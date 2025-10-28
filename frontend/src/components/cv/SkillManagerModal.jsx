import React from 'react';
import SkillLinker from './SkillLinker';

const SkillManagerModal = ({
  isOpen,
  onClose,
  allSkills,
  selectedSkillIds,
  setSelectedSkillIds,
  pendingSkills,
  setPendingSkills
}) => {
  if (!isOpen) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 0, left: 0, width: '100%', height: '100%',
      backgroundColor: 'rgba(0,0,0,0.4)',
      display: 'flex', justifyContent: 'center', alignItems: 'center',
      zIndex: 2000
    }}>
      <div style={{
        backgroundColor: 'white',
        padding: '20px',
        borderRadius: '10px',
        width: '600px',
        maxHeight: '90vh',
        overflowY: 'auto'
      }}>
        <h3 style={{ marginTop: 0 }}>Manage Skills</h3>
        <SkillLinker
          allSkills={allSkills}
          selectedSkillIds={selectedSkillIds}
          setSelectedSkillIds={setSelectedSkillIds}
          pendingSkills={pendingSkills}
          setPendingSkills={setPendingSkills}
        />
        <button onClick={onClose} style={{ marginTop: '10px', padding: '8px 12px', backgroundColor: '#6c757d', color: 'white', borderRadius: '5px' }}>Close</button>
      </div>
    </div>
  );
};

export default SkillManagerModal;
