// frontend/src/components/applications/IntelligentTextAreaModal.jsx
import React, { useState, useEffect, useRef, useMemo } from 'react';
import IntelligentTextArea from './IntelligentTextArea';
// (Make sure this path is correct for your project)
import CVItemDisplayCard from './CVItemDisplayCard'; 

// Helper to get a simple icon for each category
const getCategoryIcon = (type) => {
  switch (type) {
    case 'experiences': return 'bi bi-briefcase';
    case 'skills': return 'bi bi-star';
    case 'projects': return 'bi bi-tools';
    case 'education': return 'bi bi-mortarboard';
    case 'achievements': return 'bi bi-trophy';
    case 'hobbies': return 'bi bi-controller';
    default: return 'bi bi-file-earmark-text';
  }
};

const IntelligentTextAreaModal = ({
  isOpen,
  onClose,
  initialValue,
  onSave,
  cv, // 'cv' here is the *fullCV* object
  onShowPreview, 
  title
}) => {
  const [currentText, setCurrentText] = useState(initialValue || '');
  const editorRef = useRef(null);
  
  // State to manage the *inline* preview
  const [inlinePreview, setInlinePreview] = useState(null); // { item, type }

  useEffect(() => {
    if (isOpen) {
      setCurrentText(initialValue || '');
      setInlinePreview(null); // Reset preview when modal opens
    }
  }, [isOpen, initialValue]);

  // Get the categories from the CV for the shortcut buttons
  const cvCategories = useMemo(() => {
    if (!cv) return [];
    return [
      { type: 'experiences', name: 'Experiences', items: cv.experiences || [] },
      { type: 'projects', name: 'Projects', items: cv.projects || [] },
      { type: 'skills', name: 'Skills', items: cv.skills || [] },
      { type: 'education', name: 'Education', items: cv.education || [] },
      { type: 'achievements', name: 'Achievements', items: cv.achievements || [] },
      { type: 'hobbies', name: 'Hobbies', items: cv.hobbies || [] },
    ];
  }, [cv]);

  // --- Modal Handlers ---
  const handleSaveAndClose = () => {
    onSave(currentText);
    onClose();
  };

  const handleCancel = () => {
    onClose();
  };

  // --- Shortcut Button Handler ---
  const handleShortcutClick = (categoryType) => {
    const category = cvCategories.find(c => c.type === categoryType);
    if (category && editorRef.current) {
        editorRef.current.triggerCategorySearch(category);
    }
  };
  
  // --- This handler "catches" the click from the text area ---
  const handleInlinePreview = (item, type) => {
    setInlinePreview({ item, type });
  };
  
  // --- Helper to close the inline preview ---
  const closeInlinePreview = () => {
    setInlinePreview(null);
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="modal-backdrop fade show" style={{ opacity: 0.5 }}></div>
      {/* --- Using modal-lg for the horizontal split --- */}
      <div className="modal fade show" style={{ display: 'block' }} tabIndex="-1">
        <div className="modal-dialog modal-lg modal-dialog-centered"> 
          <div className="modal-content">
            <div className="modal-header">
              <h5 className="modal-title">Edit: {title}</h5>
              <button type="button" className="btn-close" onClick={handleCancel}></button>
            </div>

            {/* --- flex-column for a top/bottom layout --- */}
            <div className="modal-body" style={{ minHeight: '50vh' }}>
              <div className="d-flex flex-column gap-3">

                {/* --- Section 1: Editor --- */}
                <div style={{ flex: '2' }}>
                  {/* Shortcut Buttons */}
                  <div className="mb-2 d-flex flex-wrap gap-2">
                    <span className='small me-2'>Add CV Reference:</span>
                    {cvCategories.map(cat => (
                      <button 
                        key={cat.type}
                        type="button" 
                        className="btn btn-outline-primary btn-sm"
                        title={`Add ${cat.name}`}
                        onClick={() => handleShortcutClick(cat.type)}
                      >
                        <i className={`${getCategoryIcon(cat.type)} me-1`}></i>
                        {cat.name}
                      </button>
                    ))}
                  </div>

                  {/* The Editor */}
                  <IntelligentTextArea
                    ref={editorRef}
                    initialValue={currentText}
                    cv={cv}
                    manageSaveExternally={true}
                    onLocalTextChange={setCurrentText}
                    onSave={() => {}}
                    onShowPreview={handleInlinePreview} 
                  />
                </div>

                {/* --- Section 2: Preview --- */}
                <div style={{ flex: '1', borderTop: '1px solid var(--bs-border-color)', paddingTop: '1rem', maxHeight: '30vh', overflowY: 'auto' }}>
                  {inlinePreview ? (
                    <div>
                      <div className="d-flex justify-content-between align-items-center mb-2">
                        <h6 className="mb-0 text-muted">Preview</h6>
                        <button type="button" className="btn-close btn-sm" onClick={closeInlinePreview}></button>
                      </div>
                      
                      {/* --- THIS IS THE FIX --- */}
                      {/* By adding a unique 'key', we force React to 
                        unmount the old component and mount a new one
                        every time the item ID changes. This fixes
                        the "double-click" bug.
                      */}
                      <CVItemDisplayCard
                        key={inlinePreview.item.id}
                        item={inlinePreview.item}
                        itemType={inlinePreview.type} 
                        allSkills={cv.skills || []}
                        allAchievements={cv.achievements || []}
                        allExperiences={cv.experiences || []}
                        allEducation={cv.education || []}
                      />
                      {/* --- END OF FIX --- */}

                    </div>
                  ) : (
                    <div className="text-center text-muted" style={{ paddingTop: '2rem' }}>
                      <i className="bi bi-eye-fill fs-3"></i>
                      <p className="mt-2">Click a reference in the editor to preview it here.</p>
                    </div>
                  )}
                </div>

              </div>
            </div>
            {/* --- END MODIFIED LAYAYOUT --- */}

            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={handleCancel}>
                Cancel
              </button>
              <button type="button" className="btn btn-primary" onClick={handleSaveAndClose}>
                Save Changes
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default IntelligentTextAreaModal;