import React, { useState, useEffect } from 'react';
import { Award, Plus, CheckCircle2, Copy, Link as LinkIcon } from 'lucide-react';
import AchievementForm from './AchievementForm';
import AchievementLinker from './AchievementLinker';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const AchievementManagerPanel = ({
  isOpen,
  onClose,
  allAchievements = [],
  selectedAchievementIds = [],
  setSelectedAchievementIds = () => {},
  pendingAchievements = [],
  setPendingAchievements = () => {},
  allSkills = [],
  sessionSkills = []
}) => {
  const [editingItemId, setEditingItemId] = useState(null); 
  const [isCreatingNew, setIsCreatingNew] = useState(false); 
  const [isLinkerVisible, setIsLinkerVisible] = useState(false);

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

  // --- HANDLERS (All unchanged) ---
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

  // --- COMBINED LIST LOGIC ---
  const activeEditingItem = [...linkedAchievements, ...pendingItemsForGrid].find(i => i.id === editingItemId);
  const displayedLinked = linkedAchievements.filter(i => i.id !== editingItemId);
  const displayedPending = pendingItemsForGrid.filter(i => i.id !== editingItemId);
  const allItemsForGrid = [...displayedLinked, ...displayedPending];

  
  // --- Effect to lock body scroll (Unchanged) ---
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'auto';
    }
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      {/* 1. BACKDROP (Unchanged) */}
      <div 
        style={{
          position: 'fixed',
          inset: '0px',
          backgroundColor: 'rgba(0, 0, 0, 0.5)',
          backdropFilter: 'blur(2px)',
          zIndex: 1040 
        }}
        onClick={onClose}
      ></div>

      {/* 2. PANEL */}
      <div 
        className="achievement-panel"
        style={{
          position: 'fixed',
          top: '0px',
          bottom: '0px',
          right: '0px',
          // width: '70%',  <-- REMOVED
          // maxWidth: '900px', <-- REMOVED
          zIndex: 1050,
          backgroundColor: '#f8fafc',
          boxShadow: '-5px 0 15px rgba(0,0,0,0.1)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden', 
        }}
      >
        
        {/* HEADER (Unchanged) */}
        <div className="modal-header bg-white border-bottom px-4 py-3">
          <h5 className="modal-title fw-bold d-flex align-items-center gap-2">
              <div className="p-2 bg-amber-100 text-amber-600 rounded-circle">
                  <Award size={20} />
              </div>
              Manage Achievements
          </h5>
          <button type="button" className="btn-close" onClick={onClose}></button>
        </div>

        {/* BODY (Refactored for mobile) */}
        <div 
          className="modal-body p-3 p-md-4" // Responsive padding
          style={{ flex: 1, overflowY: 'auto' }}
        >
          {/* --- SECTION 1: MASTER LINKER (RESPONSIVE) --- */}
          <div className="mb-4">
              <div className="d-flex justify-content-between align-items-center mb-0 pb-2 border-bottom">
                  <h6 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2">
                      <LinkIcon size={16} className="text-primary"/>
                      Add from Master List
                  </h6>
                  <button 
                      type="button"
                      className="btn btn-light btn-sm d-flex align-items-center gap-2"
                      onClick={() => setIsLinkerVisible(!isLinkerVisible)}
                  >
                      {/* Responsive Text */}
                      <span className="d-none d-md-inline">{isLinkerVisible ? 'Hide List' : 'Show List'}</span>
                      <span className="d-inline d-md-none">{isLinkerVisible ? 'Hide' : 'Show'}</span>
                  </button>
              </div>

              {isLinkerVisible && (
                  <div className="animate-fade-in mt-3 p-3 p-md-4 bg-white border rounded-3 shadow-sm">
                        <div className="alert alert-light border small text-muted mb-3">
                            <CheckCircle2 size={14} className="me-1 text-success" style={{marginTop: '-2px'}}/>
                            Selecting an achievement links it to this item.
                        </div>
                        <AchievementLinker
                            allAchievements={allAchievements}
                            selectedAchievementIds={selectedAchievementIds}
                            setSelectedAchievementIds={setSelectedAchievementIds}
                            disabledAchievementIds={disabledAchievementIds}
                        />
                  </div>
              )}
          </div>

          {/* --- SECTION 2: YOUR ACHIEVEMENTS (RESPONSIVE) --- */}
          <div>
              <div className="d-flex justify-content-between align-items-center mb-3 pb-2 border-bottom">
                  <h6 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2">
                      <Award size={16} className="text-amber-600"/>
                      Your Achievements
                  </h6>
                  {!isCreatingNew && !editingItemId && (
                      <button 
                          type="button" 
                          className="btn btn-primary btn-sm d-flex align-items-center gap-2"
                          onClick={() => { setIsCreatingNew(true); setEditingItemId(null); }}
                      >
                          <Plus size={16}/>
                          {/* Responsive Text */}
                          <span className="d-none d-md-inline">Create New Achievement</span>
                      </button>
                  )}
              </div>

              {/* Create New Form */}
              {isCreatingNew && (
                  <div className="mb-4 p-3 p-md-4 bg-white rounded-xl shadow-sm border">
                      <h6 className="small fw-bold text-primary mb-3 text-uppercase tracking-wide">Creating New Achievement</h6>
                      <AchievementForm
                          onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, null)}
                          cvId={null} 
                          allSkills={allSkills}
                          initialData={null} 
                          onCancelEdit={() => setIsCreatingNew(false)}
                          sessionSkills={sessionSkills}
                      />
                  </div>
              )}

              {/* Edit Form (for *any* item) */}
              {activeEditingItem && (
                    <div className="bg-white p-3 p-md-4 rounded-xl shadow-sm border mb-3">
                      {/* Show the "Modified" banner only when editing a *linked* item */}
                      {activeEditingItem && !activeEditingItem.isPending && (
                        <div className="d-flex align-items-center gap-2 mb-3 text-primary small fw-bold bg-blue-50 p-2 rounded border border-blue-100">
                            <Copy size={14}/>
                            {/* --- CORRECTED TEXT --- */}
                            Editing a linked item will create a pending modification.
                        </div>
                      )}
                      <AchievementForm
                          onSubmit={(cvId, data, type) => handleFormSubmit(cvId, data, type, activeEditingItem)}
                          cvId={null} 
                          allSkills={allSkills}
                          initialData={activeEditingItem} 
                          onCancelEdit={() => handleCancelEdit(activeEditingItem)}
                          sessionSkills={sessionSkills}
                      />
                  </div>
              )}

              {/* Grid of ALL items */}
              <AchievementDisplayGrid 
                  achievementsToDisplay={allItemsForGrid}
                  allSkills={allSkills}
                  onRemoveAchievement={handleRemove}
                  onEditAchievement={handleEdit}
              />
              
              {/* Empty State */}
              {allItemsForGrid.length === 0 && !isCreatingNew && !editingItemId && (
                  <div className="text-center py-4 bg-white rounded-xl border border-dashed text-muted small">
                      No achievements yet. Click "Create New" to add one.
                  </div>
              )}
          </div>

        </div>

        {/* FOOTER (Unchanged) */}
        <div className="modal-footer bg-white border-top px-4 py-3">
          <button 
            type="button" 
            className="btn btn-primary px-4" 
            onClick={onClose}
          >
           Done
          </button>
        </div>
      </div>

      {/* 3. STYLES (UPDATED BLOCK) */}
      <style>{`
        @keyframes slideInFromRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
        .achievement-panel {
          /* 1. Mobile-first width (make it bigger on small screens) */
          width: 90%; 
          max-width: 100%;

          /* 2. Apply desktop styles only on screens 768px and up */
          @media (min-width: 768px) {
            width: 70%;
            max-width: 900px;
          }

          animation: slideInFromRight 0.3s ease-out;
        }
        .animate-fade-in {
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </>
  );
};

export default AchievementManagerPanel;