// frontend/src/components/cv/SkillManagerModal.jsx
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
    // The modal container with overlay
    <div 
        className="modal" 
        style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.4)' }}
        onClick={onClose} // Close on overlay click
    >
      <div 
        className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
        onClick={e => e.stopPropagation()} // Prevent modal click-through
      >
        <div className="modal-content">

          <div className="modal-header">
            <h5 className="modal-title">Manage Skills</h5>
            <button 
                type="button" 
                className="btn-close" 
                onClick={onClose}
            ></button>
          </div>

          <div className="modal-body">
            <p className="text-muted small">
              Link existing skills or add new ones (pending). Changes are saved when you submit the main form.
            </p>

            <SkillLinker
              allSkills={allSkills}
              selectedSkillIds={selectedSkillIds}
              setSelectedSkillIds={setSelectedSkillIds}
              pendingSkills={pendingSkills}
              setPendingSkills={setPendingSkills}
            />
          </div>

          <div className="modal-footer">
            <button 
                type="button" 
                className="btn btn-secondary" 
                onClick={onClose}
            >
             Close
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SkillManagerModal;