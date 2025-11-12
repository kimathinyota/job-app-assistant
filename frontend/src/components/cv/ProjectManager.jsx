// frontend/src/components/cv/ProjectManager.jsx
import React, { useState } from 'react';
import ProjectForm from './ProjectForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

// This is the new component that encapsulates all logic for Projects
const ProjectManager = ({
  cvId,
  projects = [],
  // We now need experiences and education to get names for display
  allExperiences = [],
  allEducation = [],
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
  
  // Helpers to find the names of linked items
  const getRelatedExperienceName = (id) => {
    if (!id) return null;
    const exp = allExperiences.find(e => e.id === id);
    return exp ? `${exp.title} @ ${exp.company}` : 'Unknown Experience';
  };
  
  const getRelatedEducationName = (id) => {
    if (!id) return null;
    const edu = allEducation.find(e => e.id === id);
    return edu ? `${edu.degree} @ ${edu.institution}` : 'Unknown Education';
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
    onDelete(cvId, itemId, 'projects'); // Changed
  };

  return (
    <div>
      {/* <button onClick={onBack} className="btn btn-secondary mb-3">
        &larr; Back to CV Dashboard
      </button> */}

      <h3 className="h4 border-bottom pb-2 text-capitalize">
        Projects
      </h3>

      {/* "Add New" Button (unchanged) */}
      {!isCreating && !editingId && (
        <button
          className="btn btn-primary my-3"
          onClick={handleAddNewClick}
        >
          + Add New Project
        </button>
      )}

      {/* "Create New" Form */}
      {isCreating && (
        <ProjectForm
          key="new-project-form"
          onSubmit={handleSubmitCreate} // Use wrapper handler
          cvId={cvId}
          // Pass all data to the form
          allExperiences={allExperiences}
          allEducation={allEducation}
          allSkills={allSkills}
          allAchievements={allAchievements}
          initialData={null} // 'create' mode
          onCancelEdit={handleCancel} // Use local handler
        />
      )}

      {/* List of Projects (Display or Edit) */}
      <ul className="list-group list-group-flush mt-3">
        {projects.map(item => { // Changed
          if (item.id === editingId) {
            // --- RENDER EDIT FORM ---
            return (
              <ProjectForm
                key={item.id}
                onSubmit={handleSubmitUpdate} // Use wrapper handler
                cvId={cvId}
                // Pass all data to the form
                allExperiences={allExperiences}
                allEducation={allEducation}
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
            
            // Get related item names
            const relatedExpName = getRelatedExperienceName(item.related_experience_id);
            const relatedEduName = getRelatedEducationName(item.related_education_id);

            return (
              <li
                key={item.id}
                className="list-group-item p-3 mb-3 border shadow-sm rounded"
              >
                {/* Header (Modified for Project) */}
                <div className="mb-2">
                  <strong className="fs-5 d-block">
                    {item.title || 'Untitled Project'}
                  </strong>
                  
                  {/* NEW: Display for related items */}
                  {relatedExpName && (
                    <span className="badge bg-secondary me-2">
                        Relates to: {relatedExpName}
                    </span>
                  )}
                  {relatedEduName && (
                    <span className="badge bg-info text-dark">
                        Relates to: {relatedEduName}
                    </span>
                  )}
                  
                  {item.description && (
                    <p className="mb-2 mt-2" style={{ whiteSpace: 'pre-wrap' }}>
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

export default ProjectManager;