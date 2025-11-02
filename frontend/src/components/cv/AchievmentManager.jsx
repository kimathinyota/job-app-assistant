// frontend/src/components/cv/AchievementManager.jsx
import React, { useState } from 'react';
import AchievementForm from './AchievementForm';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

// This is the new component that encapsulates all logic for Master Achievements
const AchievementManager = ({
  cvId,
  achievements = [],
  allSkills = [],
  onSubmit, // This will be `handleAddOrUpdateNestedItem` from the parent
  onDelete, // This will be `handleDeleteNested` from the parent
  onBack    // This will be `() => setActiveSection(null)`
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);

  // --- Handlers to manage local state ---
  const handleAddNewClick = () => {
    setIsCreating(true);
    setEditingId(null);
  };

  const handleEditClick = (itemId) => {
    setEditingId(itemId);
    setIsCreating(false);
  };

  const handleCancel = () => {
    setIsCreating(false);
    setEditingId(null);
  };

  // --- Wrapper handlers to submit and then reset local state ---
  const handleSubmitCreate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType);
    handleCancel();
  };

  const handleSubmitUpdate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType);
    handleCancel();
  };

  const handleDeleteClick = (itemId) => {
    onDelete(cvId, itemId, 'achievements');
  };

  return (
    <div>
      <button onClick={onBack} className="btn btn-secondary mb-3">
        &larr; Back to CV Dashboard
      </button>

      <h3 className="h4 border-bottom pb-2 text-capitalize">
        Master Achievements
      </h3>

      {/* "Add New" Button */}
      {!isCreating && !editingId && (
        <button
          className="btn btn-primary my-3"
          onClick={handleAddNewClick}
        >
          + Add New Master Achievement
        </button>
      )}

      {/* "Create New" Form */}
      {isCreating && (
        <AchievementForm
          key="new-achievement-form"
          onSubmit={handleSubmitCreate}
          cvId={cvId}
          allSkills={allSkills}
          initialData={null}
          onCancelEdit={handleCancel}
        />
      )}

      {/* List of Achievements (Display or Edit) */}
      <ul className="list-group list-group-flush mt-3">
        {achievements.map(item => {
          if (item.id === editingId) {
            // --- RENDER EDIT FORM ---
            return (
              <AchievementForm
                key={item.id}
                onSubmit={handleSubmitUpdate}
                cvId={cvId}
                allSkills={allSkills}
                initialData={item}
                onCancelEdit={handleCancel}
              />
            );
          } else {
            // --- RENDER DISPLAY CARD ---
            return (
              <li
                key={item.id}
                className="list-group-item p-3 mb-3 border shadow-sm rounded"
              >
                {/* Header */}
                <div className="mb-2">
                  <p className="mb-1" style={{ whiteSpace: 'pre-wrap' }}>{item.text || 'Untitled Achievement'}</p>
                  {item.context && (
                    <span className="badge bg-secondary">
                      Context: {item.context}
                    </span>
                  )}
                </div>

                {/* Skills */}
                {(item.skill_ids?.length > 0) && (
                  <div className="mt-2 pt-2 border-top">
                    <strong className="form-label d-block mb-2">Related Skills:</strong>
                    <SelectedSkillsDisplay
                      allSkills={allSkills}
                      selectedSkillIds={item.skill_ids}
                      pendingSkills={[]}
                    />
                  </div>
                )}
                
                {/* Action Buttons */}
                <div className="mt-3 border-top pt-3 text-end">
                  <button
                    onClick={() => handleEditClick(item.id)}
                    className="btn btn-warning btn-sm me-2"
                  >
                    Edit
                  </button>
                  <button
                    onClick={() => handleDeleteClick(item.id)}
                    className="btn btn-danger btn-sm"
                  >
                    Delete
                  </button>
                </div>
              </li>
            );
          }
        })}
      </ul>
    </div>
  );
};

export default AchievementManager;