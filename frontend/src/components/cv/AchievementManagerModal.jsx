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
  // This state now stores the data for the form *only*.
  // `null` = new item, `{...data}` = editing item.
  const [editingItemData, setEditingItemData] = useState(null); 
  const [formKey, setFormKey] = useState(0); // Used to reset the form

  if (!isOpen) return null;

  // --- 1. DERIVE LISTS (adds 'isPending' flag) ---
  const linkedAchievements = allAchievements
    .filter(a => selectedAchievementIds.includes(a.id))
    .map(a => ({ ...a, isPending: false })); 

  const pendingItemsForGrid = pendingAchievements.map((a, i) => ({
    ...a,
    isPending: true,
    index: i, 
    id: a.id || `pending-${i}`
  }));

  
  // --- 2. HANDLERS ---

  /**
   * Called by "Remove" on EITHER grid.
   * item: The full achievement object (with 'isPending' flag).
   */
  const handleRemove = (item) => {
    if (item.isPending) {
      // Remove from pending list
      setPendingAchievements(prev => prev.filter((_, i) => i !== item.index));
    } else {
      // This is the logic you wanted:
      // Remove from linked list, which will de-select the tag.
      setSelectedAchievementIds(prev => prev.filter(id => id !== item.id));
    }
    // If we are editing the item that just got removed, reset the form
    if (editingItemData && (editingItemData.id === item.id)) {
      setEditingItemData(null);
      setFormKey(k => k + 1);
    }
  };

  /**
   * Called by "Edit" on EITHER grid.
   * item: The full achievement object (with 'isPending' flag).
   */
  const handleEdit = (item) => {
    if (item.isPending) {
      // It's already pending. Set its data in the form.
      // We pass the *original data* from the prop array
      setEditingItemData(pendingAchievements[item.index]);
    } else {
      // --- THIS IS YOUR NEW LOGIC ---
      // "Edit as a Copy"
      // 1. Unlink the master item (de-selects the tag).
      setSelectedAchievementIds(prev => prev.filter(id => id !== item.id));
      
      // 2. Populate the form with its data.
      setEditingItemData(item);
      // --- END NEW LOGIC ---
    }
    // Force the form to re-render with the new 'initialData'
    setFormKey(k => k + 1);
  };

  /**
   * Called by the AchievementForm inside the modal.
   */
  const handleFormSubmit = (_, achievementData) => {
    // Check if the form *was* editing a pending item
    const editingPendingIndex = editingItemData ? pendingItemsForGrid.findIndex(p => p.id === editingItemData.id) : -1;

    if (editingPendingIndex > -1) {
      // We were editing a PENDING item. Update it in-place.
      const updatedList = [...pendingAchievements];
      updatedList[editingPendingIndex] = {
        ...updatedList[editingPendingIndex], // keep original temp ID
        ...achievementData, // apply new data from form
      };
      setPendingAchievements(updatedList);
    } else {
      // We were either creating a NEW item
      // OR editing a LINKED item (which is now a new pending item).
      setPendingAchievements(prev => [
        ...prev,
        { ...achievementData, id: `pending-${Date.now()}` }
      ]);
    }
    
    // Reset the form
    setEditingItemData(null);
    setFormKey(k => k + 1);
  };

  return (
    <div 
      className="modal" 
      style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.4)' }}
      // *** FIX 1: Add onClick handler to the overlay ***
      onClick={onClose}
    >
      <div 
        className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
        // *** FIX 2: Stop click propagation ***
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-content">
          <div className="modal-header">
            <h5 className="modal-title">Manage Achievements</h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>

          <div className="modal-body">
            
            {/* 1. LINKER (TAGS) */}
            <div className="mb-4">
              <strong className="fs-6">Link Master Achievements</strong>
              <AchievementLinker
                allAchievements={allAchievements}
                selectedAchievementIds={selectedAchievementIds}
                setSelectedAchievementIds={setSelectedAchievementIds}
              />
            </div>

            {/* 2. LINKED GRID - Displays toggled items */}
            <div className="mb-4">
              <strong className="fs-6">Linked Achievements (Click Edit to copy & modify)</strong>
              <AchievementDisplayGrid
                achievementsToDisplay={linkedAchievements}
                allSkills={allSkills}
                onRemoveAchievement={handleRemove} // Wired to handleRemove
                onEditAchievement={handleEdit}   // Wired to handleEdit
                isDisplayOnly={false}
              />
            </div>

            {/* 3. PENDING GRID - Displays new/copied items */}
            <div className="mb-4">
              <strong className="fs-6">New / Modified Achievements (Local to this experience)</strong>
              <AchievementDisplayGrid
                achievementsToDisplay={pendingItemsForGrid}
                allSkills={allSkills}
                onRemoveAchievement={handleRemove} // Wired to handleRemove
                onEditAchievement={handleEdit}   // Wired to handleEdit
                isDisplayOnly={false}
              />
            </div>

            {/* 4. FORM - For creating/editing items */}
            <div className="border-top pt-3">
              <AchievementForm
                key={formKey} // Resets form when key changes
                onSubmit={handleFormSubmit} // Submits to modal's state
                cvId={null} // Not saving to DB
                allSkills={allSkills}
                initialData={editingItemData} // Populates form
                onCancelEdit={() => {
                  setEditingItemData(null); // Clear form
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