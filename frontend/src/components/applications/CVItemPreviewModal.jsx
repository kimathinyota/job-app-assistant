// frontend/src/components/applications/CVItemPreviewModal.jsx
import React from 'react';
import CVItemDisplayCard from './CVItemDisplayCard';

const CVItemPreviewModal = ({
    isOpen,
    onClose,
    itemToPreview, // This will be an object { item: {...}, type: '...' }
    allSkills,
    allAchievements,
    allExperiences,
    allEducation,
    allHobbies = [] // <--- 1. Accept Prop
}) => {
    if (!isOpen || !itemToPreview) return null;

    const { item, type } = itemToPreview;

    return (
        <div 
            className="modal" 
            style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}
            onClick={onClose}
        >
            <div 
                className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
                onClick={e => e.stopPropagation()}
            >
                <div className="modal-content">
                    <div className="modal-header">
                        <h5 className="modal-title">CV Item Preview</h5>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>
                    <div className="modal-body">
                        <CVItemDisplayCard
                            item={item}
                            itemType={type}
                            allSkills={allSkills}
                            allAchievements={allAchievements}
                            allExperiences={allExperiences}
                            allEducation={allEducation}
                            allHobbies={allHobbies} // <--- 2. Pass Prop
                        />
                    </div>
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CVItemPreviewModal;