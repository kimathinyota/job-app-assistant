// frontend/src/components/cv/AchievementForm.jsx
import React, { useState, useEffect } from 'react';
import SkillManagerModal from './SkillManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

const AchievementForm = ({ 
  onSubmit, 
  cvId, 
  allSkills, 
  initialData, 
  onCancelEdit 
}) => {
  const [text, setText] = useState('');
  const [selectedSkillIds, setSelectedSkillIds] = useState([]);
  const [pendingSkills, setPendingSkills] = useState([]);
  const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);

  // 1. Use an 'isEditing' flag based on initialData
  const isEditing = Boolean(initialData);

  useEffect(() => {
    if (isEditing) {
      setText(initialData.text || '');
      // This is crucial: populate state from initialData
      setSelectedSkillIds(initialData.skill_ids || initialData.existing_skill_ids || []);
      setPendingSkills(initialData.new_skills || []);
    } else {
      // Reset form if not editing
      setText('');
      setSelectedSkillIds([]);
      setPendingSkills([]);
    }
  }, [initialData, isEditing]); // Depend on initialData

  // 2. This is the NEW, corrected handler
  const handleFormSubmit = () => {
    if (!text.trim()) return;

    // Package the form's *current state*
    const dataToSend = {
      text: text,
      context: 'Global', // This is fine for achievements
      // Use keys that match what the parent expects
      existing_skill_ids: selectedSkillIds, 
      new_skills: pendingSkills
    };

    // When in the CVManagerPage, we also pass the ID
    if (isEditing && initialData.id && !String(initialData.id).startsWith('pending-')) {
      dataToSend.id = initialData.id;
    }

    // Call the modal's "handleFormSubmit" or CVManager's "handleAddOrUpdate"
    // The 'cvId' (which is null here) is just passed along
    onSubmit(cvId, dataToSend, 'Achievement');

    // 3. Reset the form *only if we are not in edit mode*
    // (When editing, the modal handles resetting state)
    if (!isEditing) {
        setText('');
        setSelectedSkillIds([]);
        setPendingSkills([]);
    }
  };

  return (
    <div
      className="card p-3 bg-light-subtle" // Use bootstrap classes
      style={{
        border: '1px solid #6c757d',
      }}
    >
      <h4 className="h5 mt-0 mb-3 text-muted">
        {isEditing ? 'Edit Achievement' : '+ Add New/Pending Achievement'}
      </h4>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Achievement text (e.g., Optimized X by Y%)"
        className="form-control mb-3"
        rows="3"
      />

      {/* Skill Management (unchanged) */}
      <div className="mb-3">
        <strong className="form-label d-block">Related Skills:</strong>
        <button
          type="button"
          onClick={() => setIsSkillModalOpen(true)}
          className="btn btn-secondary btn-sm mb-2"
        >
          Manage Skills
        </button>

        <SelectedSkillsDisplay
          allSkills={allSkills}
          selectedSkillIds={selectedSkillIds}
          pendingSkills={pendingSkills}
        />
      </div>

      <SkillManagerModal
        isOpen={isSkillModalOpen}
        onClose={() => setIsSkillModalOpen(false)}
        allSkills={allSkills}
        selectedSkillIds={selectedSkillIds}
        setSelectedSkillIds={setSelectedSkillIds}
        pendingSkills={pendingSkills}
        setPendingSkills={setPendingSkills}
      />

      {/* 4. Action Buttons (wired to new handler) */}
      <div className="mt-3 border-top pt-3">
        <button
          type="button"
          onClick={handleFormSubmit} // <-- Use the new handler
          className={`btn ${isEditing ? 'btn-warning' : 'btn-primary'} me-2`}
        >
          {isEditing ? 'Save Changes' : 'Add to Pending'}
        </button>
        {isEditing && (
          <button
            type="button"
            onClick={onCancelEdit}
            className="btn btn-outline-secondary"
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};

export default AchievementForm;