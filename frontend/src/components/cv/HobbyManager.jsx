// frontend/src/components/cv/HobbyManager.jsx
import React, { useState } from 'react';
import HobbyForm from './HobbyForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

// This is the new component that encapsulates all logic for Hobbies
const HobbyManager = ({
  cvId,
  hobbies = [], // Changed from education
  allSkills = [],
  allAchievements = [],
  onSubmit, // This will be `handleAddOrUpdateNestedItem` from the parent
  onDelete, // This will be `handleDeleteNested` from the parent
  onBack    // This will be `() => setActiveSection(null)`
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);

  // Helper to find achievements for the display card
  const getAchievements = (achievementIds = []) => {
    if (!achievementIds || achievementIds.length === 0) return [];
    return achievementIds
      .map(id => allAchievements.find(a => a.id === id))
      .filter(Boolean);
  };

  // --- Handlers to manage local state (unchanged) ---
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

  // --- Wrapper handlers to submit and then reset local state (unchanged) ---
  const handleSubmitCreate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType); // Call parent handler
    handleCancel(); // Reset local state
  };

  const handleSubmitUpdate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType); // Call parent handler
    handleCancel(); // Reset local state
  };

  const handleDeleteClick = (itemId) => {
    // We pass the full identifiers to the parent delete handler
    onDelete(cvId, itemId, 'hobbies'); // Changed from 'education'
  };

  return (
    <div>
      {/* <button onClick={onBack} className="btn btn-secondary mb-3">
        &larr; Back to CV Dashboard
      </button> */}

      <h3 className="h4 border-bottom pb-2 text-capitalize">
        Hobbies
      </h3>

      {/* "Add New" Button (unchanged) */}
      {!isCreating && !editingId && (
        <button
          className="btn btn-primary my-3"
          onClick={handleAddNewClick}
        >
          + Add New Hobby
        </button>
      )}

      {/* "Create New" Form */}
      {isCreating && (
        <HobbyForm
          key="new-hobby-form"
          onSubmit={handleSubmitCreate} // Use wrapper handler
          cvId={cvId}
          allSkills={allSkills}
          allAchievements={allAchievements}
          initialData={null} // 'create' mode
          onCancelEdit={handleCancel} // Use local handler
        />
      )}

      {/* List of Hobbies (Display or Edit) */}
      <ul className="list-group list-group-flush mt-3">
        {hobbies.map(item => { // Changed from education
          if (item.id === editingId) {
            // --- RENDER EDIT FORM ---
            return (
              <HobbyForm
                key={item.id}
                onSubmit={handleSubmitUpdate} // Use wrapper handler
                cvId={cvId}
                allSkills={allSkills}
                allAchievements={allAchievements}
                initialData={item} // 'edit' mode
                onCancelEdit={handleCancel} // Use local handler
              />
            );
          } else {
            // --- RENDER DISPLAY CARD ---
            const linkedAchievements = getAchievements(item.achievement_ids);
            
            // Calculate aggregated skills for this item
            const allIds = new Set(item.skill_ids || []);
            linkedAchievements.forEach(ach => {
                (ach.skill_ids || []).forEach(id => allIds.add(id));
            });
            const aggregatedSkillIds = Array.from(allIds);

            return (
              <li
                key={item.id}
                className="list-group-item p-3 mb-3 border shadow-sm rounded"
              >
                {/* Header (Modified for Hobby) */}
                <div className="mb-2">
                  <strong className="fs-5 d-block">
                    {item.name || 'Untitled Hobby'}
                  </strong>
                  {item.description && (
                    <p className="mb-2" style={{ whiteSpace: 'pre-wrap' }}>
                        {item.description}
                    </p>
                  )}
                </div>

                {/* Achievements (unchanged) */}
                {linkedAchievements.length > 0 && (
                  <div className="mb-3">
                    <h6 className="small fw-bold mb-0">Key Achievements:</h6>
                    <AchievementDisplayGrid
                      achievementsToDisplay={linkedAchievements}
                      allSkills={allSkills}
                      isDisplayOnly={true}
                    />
                  </div>
                )}

                {/* Skills (unchanged) */}
                {aggregatedSkillIds.length > 0 && (
                  <div className="mt-2 pt-2 border-top">
                    <strong className="form-label d-block mb-2">Related Skills:</strong>
                    <SelectedSkillsDisplay
                      allSkills={allSkills}
                      selectedSkillIds={aggregatedSkillIds}
                      pendingSkills={[]}
                    />
                  </div>
                )}
                
                {/* Action Buttons (unchanged) */}
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

export default HobbyManager;