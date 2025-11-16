import React, { useEffect } from 'react';
import { Layers } from 'lucide-react';
import SkillLinker from './SkillLinker';

const SkillManagerPanel = ({
  isOpen,
  onClose,
  allSkills,
  selectedSkillIds,
  setSelectedSkillIds,
  pendingSkills,
  setPendingSkills,
  sessionSkills
}) => {

  // Effect to lock body scroll
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* 1. BACKDROP */}
      <div 
        style={{
          position: 'fixed',
          inset: '0px',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(2px)',
          zIndex: 1040 
        }}
        onClick={onClose}
      ></div>

      {/* 2. PANEL */}
      <div 
        className="skill-panel" // For animation
        style={{
          position: 'fixed',
          top: '0px',
          bottom: '0px',
          right: '0px',
          // width: '70%',  <-- REMOVED
          // maxWidth: '900px', <-- REMOVED
          zIndex: 1050,
          backgroundColor: '#f8fafc', // bg-slate-50
          boxShadow: '-5px 0 15px rgba(0,0,0,0.1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden', 
        }}
      >
        
        {/* HEADER */}
        <div className="modal-header bg-white border-bottom px-4 py-3">
          <h5 className="modal-title fw-bold d-flex align-items-center gap-2">
              <div className="p-2 bg-emerald-100 text-emerald-600 rounded-circle">
                  <Layers size={20} />
              </div>
              Manage Skills
          </h5>
          <button type="button" className="btn-close" onClick={onClose}></button>
        </div>

        {/* BODY */}
        <div 
          className="modal-body p-3 p-md-4" // Responsive padding
          style={{ flex: 1, overflowY: 'auto' }}
        >
          {/* We can just render the linker directly inside */}
          <SkillLinker
              allSkills={allSkills}
              selectedSkillIds={selectedSkillIds}
              setSelectedSkillIds={setSelectedSkillIds}
              pendingSkills={pendingSkills}
              setPendingSkills={setPendingSkills}
              sessionSkills={sessionSkills}
            />
        </div>

        {/* FOOTER */}
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

      {/* 3. STYLES (UPDATED BLOCK) */}
      <style>{`
        @keyframes slideInFromRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        .skill-panel {
          /* 1. Mobile-first width (make it bigger on small screens) */
          width: 90%; 
          max-width: 100%;

          /* 2. Apply desktop styles only on screens 768px and up */
          @media (min-width: 768px) {
            width: 70%;
            max-width: 900px;
          }

          animation: slideInFromRight 0.3s ease-out;
        }
      `}</style>
    </>
  );
};

export default SkillManagerPanel;