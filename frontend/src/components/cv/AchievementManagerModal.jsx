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

  /**
   * Called by the AchievementForm inside the modal.
   */
  const handleFormSubmit = (_, achievementData) => {
    // achievementData has { text, existing_skill_ids, new_skills }

    const editingPendingIndex = editingItemData ? pendingItemsForGrid.findIndex(p => p.id === editingItemData.id) : -1;

    // --- *** THIS IS THE FIX *** ---
    // Create a new object that maps 'existing_skill_ids' to 'skill_ids'
    // so that AchievementDisplayGrid can read it correctly.
    const remappedData = {
      text: achievementData.text,
      context: achievementData.context,
      skill_ids: achievementData.existing_skill_ids || [], // <-- Renamed key
      new_skills: achievementData.new_skills || []
    };
    // --- *** END OF FIX *** ---


    if (editingPendingIndex > -1) {
      // We were editing a PENDING item. Update it in-place.
      const updatedList = [...pendingAchievements];
      updatedList[editingPendingIndex] = {
        ...updatedList[editingPendingIndex], // keep original temp ID, original_id
        ...remappedData, // apply new data from form
      };
      setPendingAchievements(updatedList);
    } else {
      // We were either creating a NEW item
      // OR editing a LINKED item (which is now a new pending item).
      const newPendingItem = { 
        ...remappedData, // Use the remapped data
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

  // This logic (for disabling tags) remains correct
  const pendingOriginalIds = pendingAchievements
    .map(ach => ach.original_id) 
    .filter(Boolean); 

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
                disabledAchievementIds={pendingOriginalIds}
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
                onCancelEdit={() => {
                  setEditingItemData(null); 
                  setFormKey(k => k + 1);
                }}
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