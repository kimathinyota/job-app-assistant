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
    <div
      style={{
        position: 'fixed',
        top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 9999,
      }}
    >
      <div
        style={{
          backgroundColor: 'white',
          padding: '20px',
          borderRadius: '10px',
          width: '90%',
          maxWidth: '600px',
          maxHeight: '85vh',
          overflowY: 'auto',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
        }}
      >
        <h3 style={{ marginTop: 0, color: '#28a745' }}>Manage Skills</h3>
        <p style={{ fontSize: '0.9em', color: '#666' }}>
          Add new skills (pending) or select existing ones from your master list.  
          All changes will only be saved when you submit the main form.
        </p>

        {/* --- FULL FEATURE SkillLinker (with pending + existing skills) --- */}
        <SkillLinker
          allSkills={allSkills}
          selectedSkillIds={selectedSkillIds}
          setSelectedSkillIds={setSelectedSkillIds}
          pendingSkills={pendingSkills}
          setPendingSkills={setPendingSkills}
        />

        <div style={{ marginTop: '15px', textAlign: 'right' }}>
          <button
            type="button"
            onClick={onClose}
            style={{
              padding: '8px 14px',
              borderRadius: '6px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};

export default SkillManagerModal;
