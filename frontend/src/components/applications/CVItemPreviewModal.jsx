// frontend/src/components/applications/CVItemPreviewModal.jsx
import React from 'react';
import CVItemDisplayCard from './CVItemDisplayCard';
import { X } from 'lucide-react';

const CVItemPreviewModal = ({
    isOpen,
    onClose,
    itemToPreview, 
    allSkills,
    allAchievements,
    allExperiences,
    allEducation,
    allHobbies = []
}) => {
    // Even if closed, we render the container for animation, but handle visibility
    // Note: Bootstrap offcanvas requires specific classes
    const showClass = isOpen ? 'show' : '';
    const visibility = isOpen ? 'visible' : 'hidden';

    if (!itemToPreview) return null;
    const { item, type } = itemToPreview;

    return (
        <>
            {/* Backdrop */}
            {isOpen && (
                <div 
                    className="offcanvas-backdrop fade show" 
                    onClick={onClose}
                    style={{ zIndex: 1045 }} // Bootstrap standard is 1040, ensure it's above
                ></div>
            )}

            {/* Side Panel (Offcanvas) */}
            <div 
                className={`offcanvas offcanvas-end ${showClass} shadow-lg border-start`} 
                tabIndex="-1" 
                style={{ visibility, width: '500px', zIndex: 1050 }}
            >
                <div className="offcanvas-header border-bottom p-4">
                    <h5 className="offcanvas-title fw-bold text-dark">
                        Item Details
                    </h5>
                    <button 
                        type="button" 
                        className="btn btn-icon btn-sm btn-light rounded-circle" 
                        onClick={onClose}
                    >
                        <X size={20} />
                    </button>
                </div>
                
                <div className="offcanvas-body p-4 bg-light-subtle custom-scroll">
                    <CVItemDisplayCard
                        item={item}
                        itemType={type}
                        allSkills={allSkills}
                        allAchievements={allAchievements}
                        allExperiences={allExperiences}
                        allEducation={allEducation}
                        allHobbies={allHobbies}
                    />
                </div>
                
                <div className="offcanvas-footer p-3 border-top bg-white">
                    <button className="btn btn-outline-secondary w-100" onClick={onClose}>
                        Close Panel
                    </button>
                </div>
            </div>
        </>
    );
};

export default CVItemPreviewModal;