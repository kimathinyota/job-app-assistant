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

  const baseTextColor = '#333'; // Define a base dark color

  return (
    <div style={{
      position: 'fixed',
      top: 0, left: 0, width: '100%', height: '100%',
      backgroundColor: 'rgba(0,0,0,0.4)', // Semi-transparent overlay
      display: 'flex', justifyContent: 'center', alignItems: 'center',
      zIndex: 2000 // Ensure it's above other content
    }}>
      <div style={{
        backgroundColor: 'white', // White background for the modal content
        padding: '25px',
        borderRadius: '10px',
        width: '90%', // Responsive width
        maxWidth: '650px', // Max width
        maxHeight: '90vh',
        overflowY: 'auto',
        boxShadow: '0 4px 15px rgba(0,0,0,0.2)', // Add shadow for depth
        color: baseTextColor // Set default text color for the modal
      }}>
        {/* Removed explicit color style, will inherit baseTextColor */}
        <h3 style={{ marginTop: 0, marginBottom: '10px', borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
          Manage Skills
        </h3>

        {/* Use a slightly lighter color for descriptive text */}
        <p style={{ fontSize: '0.9em', color: '#555', marginBottom: '20px' }}>
          Link existing skills or add new ones (pending). Changes are saved when you submit the main form.
        </p>

        {/* SkillLinker component */}
        <SkillLinker
          allSkills={allSkills}
          selectedSkillIds={selectedSkillIds}
          setSelectedSkillIds={setSelectedSkillIds}
          pendingSkills={pendingSkills}
          setPendingSkills={setPendingSkills}
        />

        {/* Close Button - Keep explicit colors for buttons */}
        <div style={{ textAlign: 'right', marginTop: '20px' }}>
          <button onClick={onClose} style={{
             padding: '8px 15px',
             backgroundColor: '#6c757d', // Grey background
             color: 'white',             // White text
             border: 'none',
             borderRadius: '5px',
             cursor: 'pointer'
             }}>
             Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default SkillManagerModal;