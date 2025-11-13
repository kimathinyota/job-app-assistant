// frontend/src/components/cv/SkillManagerModal.jsx
import React from 'react';
import { Layers } from 'lucide-react';
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
        className="modal fade show d-block" 
        style={{ backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(2px)' }}
        onClick={onClose}
    >
      <div 
        className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-content border-0 shadow-lg rounded-xl overflow-hidden">

          {/* Header */}
          <div className="modal-header bg-white border-bottom px-4 py-3">
            <h5 className="modal-title fw-bold d-flex align-items-center gap-2">
                <div className="p-2 bg-emerald-100 text-emerald-600 rounded-circle">
                    <Layers size={20} />
                </div>
                Manage Skills
            </h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>

          {/* Body */}
          <div className="modal-body p-4 bg-slate-50">
            <SkillLinker
              allSkills={allSkills}
              selectedSkillIds={selectedSkillIds}
              setSelectedSkillIds={setSelectedSkillIds}
              pendingSkills={pendingSkills}
              setPendingSkills={setPendingSkills}
            />
          </div>

          {/* Footer */}
          <div className="modal-footer bg-white border-top px-4 py-3">
            <button 
                type="button" 
                className="btn btn-primary px-4" 
                onClick={onClose}
            >
             Done
            </button>
          </div>

        </div>
      </div>
    </div>
  );
};

export default SkillManagerModal;