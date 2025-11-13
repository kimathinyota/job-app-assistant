// frontend/src/components/cv/AchievementManagerModal.jsx
import React, { useState } from 'react';
import { Award, Plus, AlertCircle, CheckCircle2, Copy } from 'lucide-react';
import AchievementForm from './AchievementForm';
import AchievementLinker from './AchievementLinker';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const AchievementManagerModal = ({
  isOpen,
  onClose,
  allAchievements = [],
  selectedAchievementIds = [],
  setSelectedAchievementIds = () => {},
  pendingAchievements = [],
  setPendingAchievements = () => {},
  allSkills = [],
  sessionSkills = [] // <--- NEW: Receive aggregated session skills
}) => {
  const [editingItemId, setEditingItemId] = useState(null); 
  const [isCreatingNew, setIsCreatingNew] = useState(false); 

  if (!isOpen) return null;

  // --- DATA PREP ---
  const linkedAchievements = allAchievements
    .filter(a => selectedAchievementIds.includes(a.id))
    .map(a => ({ ...a, isPending: false })); 

  const pendingItemsForGrid = pendingAchievements.map((a, i) => ({
    ...a,
    isPending: true,
    index: i, 
    id: a.id || `pending-${i}`
  }));

  // --- HANDLERS ---
  const handleRemove = (item) => {
    if (item.isPending) {
      setPendingAchievements(prev => prev.filter((_, i) => i !== item.index));
    } else {
      const newIdList = selectedAchievementIds.filter(id => id !== item.id);
      setSelectedAchievementIds(newIdList);
    }
    if (editingItemId === item.id) setEditingItemId(null);
  };

  const handleEdit = (item) => {
    setEditingItemId(item.id);
    setIsCreatingNew(false);
  };
  
  const handleCancelEdit = (itemBeingEdited) => {
    if (itemBeingEdited && !String(itemBeingEdited.id).startsWith('pending-')) {
      const originalId = itemBeingEdited.id;
      if (!selectedAchievementIds.includes(originalId)) {
        const newIdList = [...selectedAchievementIds, originalId];
        setSelectedAchievementIds(newIdList); 
      }
    }
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
      const isPending = String(originalItemData.id).startsWith('pending-');
      if (isPending) {
        const updatedList = [...pendingAchievements];
        updatedList[originalItemData.index] = { ...updatedList[originalItemData.index], ...remappedData };
        setPendingAchievements(updatedList);
      } else {
        const newPendingItem = { 
          ...remappedData, 
          id: `pending-mod-${originalItemData.id}-${Date.now()}`,
          original_id: originalItemData.id
        };
        setPendingAchievements(prev => [...prev, newPendingItem]);
        setSelectedAchievementIds(selectedAchievementIds.filter(id => id !== originalItemData.id));
      }
      setEditingItemId(null);
    } else {
      const newPendingItem = { ...remappedData, id: `pending-${Date.now()}` };
      setPendingAchievements(prev => [...prev, newPendingItem]);
      setIsCreatingNew(false);
    }
  };

  const pendingOriginalIds = new Set(pendingAchievements.map(ach => ach.original_id).filter(Boolean));
  if (editingItemId && !String(editingItemId).startsWith('pending-')) {
     pendingOriginalIds.add(editingItemId);
  }
  const disabledAchievementIds = Array.from(pendingOriginalIds);

  // --- SPLIT LOGIC FOR DISPLAY ---
  const activeEditingItem = [...linkedAchievements, ...pendingItemsForGrid].find(i => i.id === editingItemId);
  const displayedLinked = linkedAchievements.filter(i => i.id !== editingItemId);
  const displayedPending = pendingItemsForGrid.filter(i => i.id !== editingItemId);

  return (
    <div 
      className="modal fade show d-block" 
      style={{ backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(2px)' }}
      onClick={onClose}
    >
      <div 
        className="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable"
        onClick={e => e.stopPropagation()}
      >
        <div className="modal-content border-0 shadow-lg rounded-xl overflow-hidden">
          
          {/* HEADER */}
          <div className="modal-header bg-white border-bottom px-4 py-3">
            <h5 className="modal-title fw-bold d-flex align-items-center gap-2">
                <div className="p-2 bg-amber-100 text-amber-600 rounded-circle">
                    <Award size={20} />
                </div>
                Manage Achievements
            </h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>

          {/* BODY */}
          <div className="modal-body p-0 bg-slate-50">
            <div className="row g-0 h-100">
                
                {/* LEFT: Linker */}
                <div className="col-lg-4 border-end bg-white p-4 h-100 overflow-auto">
                    <h6 className="fw-bold text-dark mb-3 d-flex align-items-center gap-2">
                        <span className="badge bg-light text-dark border rounded-circle" style={{width: '24px', height: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>1</span>
                        Select from Master List
                    </h6>
                    <div className="alert alert-light border small text-muted mb-3">
                        <CheckCircle2 size={14} className="me-1 text-success" style={{marginTop: '-2px'}}/>
                        Selecting an achievement links it. Editing a linked item creates a unique local copy for this entry.
                    </div>
                    <AchievementLinker
                        allAchievements={allAchievements}
                        selectedAchievementIds={selectedAchievementIds}
                        setSelectedAchievementIds={setSelectedAchievementIds}
                        disabledAchievementIds={disabledAchievementIds}
                    />
                </div>

                {/* RIGHT: Workspace */}
                <div className="col-lg-8 p-4 h-100 overflow-auto bg-slate-50">
                    
                    {/* LINKED ITEMS */}
                    <div className="mb-5">
                        <h6 className="fw-bold text-dark mb-3 pb-2 border-bottom d-flex align-items-center gap-2">
                            <span className="badge bg-light text-dark border rounded-circle" style={{width: '24px', height: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>2</span>
                            Linked Master Achievements
                        </h6>
                        
                        {activeEditingItem && !activeEditingItem.isPending && (
                             <div className="bg-white p-4 rounded-xl shadow-sm border border-primary position-relative mb-3">
                                <div className="d-flex align-items-center gap-2 mb-3 text-primary small fw-bold bg-blue-50 p-2 rounded border border-blue-100">
                                    <Copy size={14}/>
                                    Editing this will detach it and create a local copy.
                                </div>
                                <AchievementForm
                                    onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, activeEditingItem)}
                                    cvId={null} 
                                    allSkills={allSkills}
                                    initialData={activeEditingItem} 
                                    onCancelEdit={() => handleCancelEdit(activeEditingItem)}
                                    sessionSkills={sessionSkills} // <--- PASS HERE
                                />
                            </div>
                        )}

                        <AchievementDisplayGrid 
                            achievementsToDisplay={displayedLinked}
                            allSkills={allSkills}
                            onRemoveAchievement={handleRemove}
                            onEditAchievement={handleEdit}
                        />
                    </div>

                    {/* LOCAL ITEMS */}
                    <div>
                        <div className="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom">
                            <h6 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2">
                                <span className="badge bg-light text-dark border rounded-circle" style={{width: '24px', height: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>3</span>
                                New / Modified (Local)
                            </h6>
                            {!isCreatingNew && !editingItemId && (
                                <button 
                                    className="btn btn-primary btn-sm d-flex align-items-center gap-2"
                                    onClick={() => { setIsCreatingNew(true); setEditingItemId(null); }}
                                >
                                    <Plus size={16}/> Create New
                                </button>
                            )}
                        </div>

                        {isCreatingNew && (
                            <div className="mb-4 p-4 bg-white rounded-xl shadow-sm border">
                                <h6 className="small fw-bold text-primary mb-3 text-uppercase tracking-wide">Creating New Achievement</h6>
                                <AchievementForm
                                    onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, null)}
                                    cvId={null} 
                                    allSkills={allSkills}
                                    initialData={null} 
                                    onCancelEdit={() => setIsCreatingNew(false)}
                                    sessionSkills={sessionSkills} // <--- PASS HERE
                                />
                            </div>
                        )}

                        {activeEditingItem && activeEditingItem.isPending && (
                             <div className="bg-white p-4 rounded-xl shadow-sm border mb-3">
                                <AchievementForm
                                    onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, activeEditingItem)}
                                    cvId={null} 
                                    allSkills={allSkills}
                                    initialData={activeEditingItem} 
                                    onCancelEdit={() => handleCancelEdit(activeEditingItem)}
                                    sessionSkills={sessionSkills} // <--- PASS HERE
                                />
                            </div>
                        )}

                        <AchievementDisplayGrid 
                            achievementsToDisplay={displayedPending}
                            allSkills={allSkills}
                            onRemoveAchievement={handleRemove}
                            onEditAchievement={handleEdit}
                        />
                        
                        {displayedPending.length === 0 && !isCreatingNew && !activeEditingItem?.isPending && (
                            <div className="text-center py-5 bg-white rounded-xl border border-dashed text-muted small">
                                No local achievements created yet.
                            </div>
                        )}
                    </div>

                </div>
            </div>
          </div>

          <div className="modal-footer bg-white border-top px-4 py-3">
            <button type="button" className="btn btn-primary px-4" onClick={onClose}>
             Done
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AchievementManagerModal;