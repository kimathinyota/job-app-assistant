// frontend/src/components/cv/AchievementManagerModal.jsx
import React, { useState } from 'react';
import AchievementForm from './AchievementForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import AchievementLinker from './AchievementLinker';

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
  const [editingItemData, setEditingItemData] = useState(null); 
  const [formKey, setFormKey] = useState(0); 

  if (!isOpen) return null;

  const linkedAchievements = allAchievements
    .filter(a => selectedAchievementIds.includes(a.id))
    .map(a => ({ ...a, isPending: false })); 

  const pendingItemsForGrid = pendingAchievements.map((a, i) => ({
    ...a,
    isPending: true,
    index: i, 
    id: a.id || `pending-${i}`
  }));

  
  const handleRemove = (item) => {
    if (item.isPending) {
      setPendingAchievements(prev => prev.filter((_, i) => i !== item.index));
    } else {
      const newIdList = selectedAchievementIds.filter(id => id !== item.id);
      setSelectedAchievementIds(newIdList);
    }

    if (editingItemData && (editingItemData.id === item.id)) {
      setEditingItemData(null);
      setFormKey(k => k + 1);
    }
  };

  const handleEdit = (item) => {
    if (item.isPending) {
      setEditingItemData(pendingAchievements[item.index]);
    } else {
      const newIdList = selectedAchievementIds.filter(id => id !== item.id);
      setSelectedAchievementIds(newIdList);
      setEditingItemData(item);
    }
    setFormKey(k => k + 1);
  };

  const handleFormSubmit = (_, achievementData) => {
    const editingPendingIndex = editingItemData ? pendingItemsForGrid.findIndex(p => p.id === editingItemData.id) : -1;

    const remappedData = {
      text: achievementData.text,
      context: achievementData.context,
      skill_ids: achievementData.existing_skill_ids || [], 
      new_skills: achievementData.new_skills || []
    };

    if (editingPendingIndex > -1) {
      const updatedList = [...pendingAchievements];
      updatedList[editingPendingIndex] = {
        ...updatedList[editingPendingIndex], 
        ...remappedData, 
      };
      setPendingAchievements(updatedList);
    } else {
      const newPendingItem = { 
        ...remappedData, 
        id: `pending-${Date.now()}` 
      };

      if (editingItemData && editingItemData.id && !String(editingItemData.id).startsWith('pending-')) {
        newPendingItem.original_id = editingItemData.id; 
      }
      
      setPendingAchievements(prev => [
        ...prev,
        newPendingItem
      ]);
    }
    
    setEditingItemData(null);
    setFormKey(k => k + 1);
  };

  // This logic for disabling tags is correct
  const pendingOriginalIds = new Set(
    pendingAchievements
      .map(ach => ach.original_id)
      .filter(Boolean)
  );

  const currentlyEditingMasterId = 
    editingItemData && !String(editingItemData.id).startsWith('pending-')
      ? editingItemData.id
      : null;

  if (currentlyEditingMasterId) {
    pendingOriginalIds.add(currentlyEditingMasterId);
  }
  
  const disabledAchievementIds = Array.from(pendingOriginalIds);

  
  // --- *** 1. THIS IS THE FIX *** ---
  const handleCancelEdit = () => {
    // Check if we were editing a master item
    if (editingItemData && !String(editingItemData.id).startsWith('pending-')) {
      // Re-select the item in the "Linked" list
      const originalId = editingItemData.id;

      // We must call the prop with a NEW ARRAY, not an updater function.
      // We use the `selectedAchievementIds` prop as the "previous" state.
      if (!selectedAchievementIds.includes(originalId)) {
        const newIdList = [...selectedAchievementIds, originalId];
        setSelectedAchievementIds(newIdList); // Pass the new array
      }
      // If it's already included (which it shouldn't be, but good to check),
      // we don't need to do anything.
    }
    
    // Clear the form
    setEditingItemData(null);
    setFormKey(k => k + 1);
  };
  // --- *** END OF FIX *** ---


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
            
            <div className="mb-4">
              <strong className="fs-6">Link Master Achievements</strong>
              <AchievementLinker
                allAchievements={allAchievements}
                selectedAchievementIds={selectedAchievementIds}
                setSelectedAchievementIds={setSelectedAchievementIds}
                disabledAchievementIds={disabledAchievementIds}
              />
            </div>

            <div className="mb-4">
              <strong className="fs-6">Linked Achievements (Click Edit to copy & modify)</strong>
              <AchievementDisplayGrid
                achievementsToDisplay={linkedAchievements}
                allSkills={allSkills}
                onRemoveAchievement={handleRemove} 
                onEditAchievement={handleEdit}   
                isDisplayOnly={false}
              />
            </div>

            <div className="mb-4">
              <strong className="fs-6">New / Modified Achievements (Local to this experience)</strong>
              <AchievementDisplayGrid
                achievementsToDisplay={pendingItemsForGrid}
                allSkills={allSkills}
                onRemoveAchievement={handleRemove} 
                onEditAchievement={handleEdit}   
                isDisplayOnly={false}
              />
            </div>

            <div className="border-top pt-3">
              <AchievementForm
                key={formKey} 
                onSubmit={handleFormSubmit} 
                cvId={null} 
                allSkills={allSkills}
                initialData={editingItemData} 
                // --- *** 2. WIRE THE CORRECT HANDLER *** ---
                onCancelEdit={handleCancelEdit}
              />
            </div>
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