// frontend/src/components/cv/AchievementManagerModal.jsx
import React, { useState } from 'react';
import AchievementForm from './AchievementForm';
// AchievementDisplayGrid is no longer used here
import AchievementLinker from './AchievementLinker';

// This is the card component, extracted from AchievementDisplayGrid
const AchievementCard = ({
  ach,
  allSkills = [],
  onRemove,
  onEdit,
}) => {
  const getSkillName = (id) => {
    const skill = allSkills.find((s) => s.id === id);
    return skill ? skill.name : id;
  };

  return (
    <div className="card h-100 shadow-sm">
      <div className="card-body position-relative">
        {/* Action Buttons */}
        <button
          type="button"
          onClick={() => onRemove(ach)}
          className="btn-close position-absolute"
          style={{ top: '0.75rem', right: '0.75rem' }}
          title="Remove Achievement"
        ></button>
        <button
          type="button"
          onClick={() => onEdit(ach)}
          className="btn btn-sm btn-link position-absolute p-0"
          style={{ top: '0.65rem', right: '2.2rem', textDecoration: 'none', fontSize: '1.1rem' }}
          title="Edit Achievement"
        >
          ✎
        </button>

        {/* Text */}
        <p className="card-text fw-medium mb-2" style={{ paddingRight: '2.5rem' }}>
          {ach.text}
        </p>

        {/* --- ADDED THIS --- */}
        {ach.context && (
            <span className="badge bg-secondary-subtle text-secondary-emphasis small mb-2">
                Context: {ach.context}
            </span>
        )}
        {/* --- END OF ADDED BLOCK --- */}

        {/* Skill Tags */}
        <div className="d-flex flex-wrap gap-1">
          {(ach.skill_ids || []).map((id) => (
            <span
              key={id}
              className="badge text-bg-info-subtle text-info-emphasis fw-normal"
            >
              {getSkillName(id)}
            </span>
          ))}
          {(ach.new_skills || []).map((s, i) => (
            <span
              key={i}
              className="badge text-bg-success-subtle text-success-emphasis fw-normal border border-success-subtle"
            >
              +{s.name}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};


const AchievementManagerModal = ({
  isOpen,
  onClose,
  allAchievements = [],
  selectedAchievementIds = [],
  setSelectedAchievementIds = () => {},
  pendingAchievements = [],
  setPendingAchievements = () => {},
  allSkills = []
}) => {
  // --- *** NEW STATE *** ---
  const [editingItemId, setEditingItemId] = useState(null); // ID of item being edited in-line
  const [isCreatingNew, setIsCreatingNew] = useState(false); // Toggles the "create new" form

  if (!isOpen) return null;

  // --- *** (HELPER LOGIC - Unchanged) *** ---
  const linkedAchievements = allAchievements
    .filter(a => selectedAchievementIds.includes(a.id))
    .map(a => ({ ...a, isPending: false })); 

  const pendingItemsForGrid = pendingAchievements.map((a, i) => ({
    ...a,
    isPending: true,
    index: i, 
    id: a.id || `pending-${i}`
  }));

  // --- *** (EVENT HANDLERS - Modified) *** ---

  const handleRemove = (item) => {
    if (item.isPending) {
      setPendingAchievements(prev => prev.filter((_, i) => i !== item.index));
    } else {
      const newIdList = selectedAchievementIds.filter(id => id !== item.id);
      setSelectedAchievementIds(newIdList);
    }
    // If we were editing the item we just removed, close the form
    if (editingItemId === item.id) {
      setEditingItemId(null);
    }
  };

  const handleEdit = (item) => {
    setEditingItemId(item.id); // Set the ID of the item to be edited
    setIsCreatingNew(false); // Close the "create" form if it's open
  };
  
  const handleCancelEdit = (itemBeingEdited) => {
    // Check if we were editing a master item
    if (itemBeingEdited && !String(itemBeingEdited.id).startsWith('pending-')) {
      // Re-select the item in the "Linked" list
      const originalId = itemBeingEdited.id;
      if (!selectedAchievementIds.includes(originalId)) {
        const newIdList = [...selectedAchievementIds, originalId];
        setSelectedAchievementIds(newIdList); // Pass the new array
      }
    }
    // Clear the form
    setEditingItemId(null);
    setIsCreatingNew(false);
  };

  const handleFormSubmit = (cvId, achievementData, itemType, originalItemData) => {
    const isEditing = Boolean(originalItemData);
    
    const remappedData = {
      text: achievementData.text,
      context: achievementData.context,
      skill_ids: achievementData.existing_skill_ids || [], 
      new_skills: achievementData.new_skills || [],
      existing_skill_ids: achievementData.existing_skill_ids || [],
    };

    if (isEditing) {
      // We are editing an existing item
      const isPending = String(originalItemData.id).startsWith('pending-');
      
      if (isPending) {
        // Find by index and update
        const updatedList = [...pendingAchievements];
        updatedList[originalItemData.index] = {
          ...updatedList[originalItemData.index], 
          ...remappedData, 
        };
        setPendingAchievements(updatedList);
      } else {
        // This was a "master" achievement. We move it to "pending"
        // (This logic is from the old 'handleFormSubmit')
        const newPendingItem = { 
          ...remappedData, 
          id: `pending-mod-${originalItemData.id}-${Date.now()}`,
          original_id: originalItemData.id
        };
        setPendingAchievements(prev => [...prev, newPendingItem]);
        // Unselect it from the "master" list
        setSelectedAchievementIds(selectedAchievementIds.filter(id => id !== originalItemData.id));
      }
      setEditingItemId(null); // Close the edit form
    } else {
      // We are creating a new item
      const newPendingItem = { 
        ...remappedData, 
        id: `pending-${Date.now()}` 
      };
      setPendingAchievements(prev => [...prev, newPendingItem]);
      setIsCreatingNew(false); // Close the create form
    }
  };

  // --- (Disabling logic - Unchanged) ---
  const pendingOriginalIds = new Set(
    pendingAchievements
      .map(ach => ach.original_id)
      .filter(Boolean)
  );
  if (editingItemId && !String(editingItemId).startsWith('pending-')) {
     pendingOriginalIds.add(editingItemId);
  }
  const disabledAchievementIds = Array.from(pendingOriginalIds);

  return (
    <div 
      className="modal" 
      style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.4)' }}
      onClick={onClose}
    >
      <div 
        className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Manage Achievements</h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>

          <div className="modal-body">
            
            {/* --- 1. Link Master Achievements (Unchanged) --- */}
            <div className="mb-4">
              <strong className="fs-6">Link Master Achievements</strong>
              <AchievementLinker
                allAchievements={allAchievements}
                selectedAchievementIds={selectedAchievementIds}
                setSelectedAchievementIds={setSelectedAchievementIds}
                disabledAchievementIds={disabledAchievementIds}
              />
            </div>

            {/* --- 2. Linked Achievements (NEW RENDER LOGIC) --- */}
            <div className="mb-4">
              <strong className="fs-6">Linked Achievements (Click Edit to copy & modify)</strong>
              
              {linkedAchievements.length === 0 ? (
                 <p className="text-muted fst-italic small mt-2">No master achievements linked.</p>
              ) : (
                <div className="row g-3 mt-1">
                  {linkedAchievements.map((ach) => (
                    <React.Fragment key={ach.id}>
                      {editingItemId === ach.id ? (
                        // RENDER EDIT FORM
                        <div className="col-12">
                          <AchievementForm
                            onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, ach)}
                            cvId={null} 
                            allSkills={allSkills}
                            initialData={ach} 
                            onCancelEdit={() => handleCancelEdit(ach)}
                          />
                        </div>
                      ) : (
                        // RENDER DISPLAY CARD
                        <div className="col-12 col-md-6">
                          <AchievementCard
                            ach={ach}
                            allSkills={allSkills}
                            onRemove={handleRemove}
                            onEdit={handleEdit}
                          />
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              )}
            </div>

            {/* --- 3. New / Modified Achievements (NEW RENDER LOGIC) --- */}
            <div className="mb-4">
              <div className="d-flex justify-content-between align-items-center mb-2">
                <strong className="fs-6">New / Modified Achievements (Local to this experience)</strong>
                {!isCreatingNew && !editingItemId && (
                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => { setIsCreatingNew(true); setEditingItemId(null); }}
                  >
                    + Add New Achievement
                  </button>
                )}
              </div>

              {/* --- "Create New" Form --- */}
              {isCreatingNew && (
                <div className="mb-3">
                  <AchievementForm
                    onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, null)}
                    cvId={null} 
                    allSkills={allSkills}
                    initialData={null} // Create mode
                    onCancelEdit={() => setIsCreatingNew(false)}
                  />
                </div>
              )}

              {/* --- Pending Items Grid --- */}
              {pendingItemsForGrid.length === 0 ? (
                 !isCreatingNew && <p className="text-muted fst-italic small mt-2">No new or modified achievements.</p>
              ) : (
                <div className="row g-3 mt-1">
                  {pendingItemsForGrid.map((ach) => (
                    <React.Fragment key={ach.id}>
                      {editingItemId === ach.id ? (
                        // RENDER EDIT FORM
                        <div className="col-12">
                          <AchievementForm
                            onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, ach)}
                            cvId={null} 
                            allSkills={allSkills}
                            initialData={ach} 
                            onCancelEdit={() => handleCancelEdit(ach)}
                          />
                        </div>
                      ) : (
                        // RENDER DISPLAY CARD
                        <div className="col-12 col-md-6">
                          <AchievementCard
                            ach={ach}
                            allSkills={allSkills}
                            onRemove={handleRemove}
                            onEdit={handleEdit}
                          />
                        </div>
                      )}
                    </React.Fragment>
                  ))}
                </div>
              )}
            </div>

            {/* --- 4. REMOVED Bottom Form --- */}
            
          </div>

          <div className="modal-footer">
            <button type="button" className="btn btn-primary" onClick={onClose}>
             Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AchievementManagerModal;