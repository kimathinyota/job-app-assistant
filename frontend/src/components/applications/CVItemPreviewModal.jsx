// frontend/src/components/applications/CVItemPreviewModal.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CVItemDisplayCard from './CVItemDisplayCard';
import { X } from 'lucide-react';

import {fetchItemDetails} from '../../api/cvClient';

const CVItemPreviewModal = ({
    isOpen,
    onClose,
    itemToPreview, 
    allSkills = [], allAchievements = [], allExperiences = [], allEducation = [], allHobbies = [],
    itemId, itemType, highlightText
}) => {
    const [fetchedData, setFetchedData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const showClass = isOpen ? 'show' : '';
    const visibility = isOpen ? 'visible' : 'hidden';

    useEffect(() => {
        if (isOpen && !itemToPreview && itemId && itemType) {
            fetchRichDetails(itemId, itemType);
        } else if (isOpen) {
            setFetchedData(null); 
            setLoading(false);
            setError(null);
        }
    }, [isOpen, itemId, itemType, itemToPreview]);

    const fetchRichDetails = async (id, type) => {
        setLoading(true);
        setError(null);
        try {
            const resp = await fetchItemDetails(id, type);
            setFetchedData(resp);
        } catch (err) {
            console.error(err);
            setError("Could not load full item context.");
        } finally {
            setLoading(false);
        }
    };

    // Determine active data
    let activeItem = itemToPreview ? itemToPreview.item : (fetchedData ? fetchedData.item : null);
    let activeType = itemToPreview ? itemToPreview.type : itemType;
    
    // Use fetched lists if available, else props
    const resolvedSkills = fetchedData ? (fetchedData.skills || []) : allSkills;
    const resolvedAchievements = fetchedData ? (fetchedData.achievements || []) : allAchievements;
    const resolvedExperiences = fetchedData ? (fetchedData.experiences || []) : allExperiences;
    const resolvedEducation = fetchedData ? (fetchedData.education || []) : allEducation;
    const resolvedHobbies = fetchedData ? (fetchedData.hobbies || []) : allHobbies;

    if (!isOpen) return null;

    return (
        <>
            <div className={`offcanvas-backdrop fade ${showClass}`} onClick={onClose} style={{ zIndex: 1045 }}></div>
            <div className={`offcanvas offcanvas-end ${showClass} shadow-lg border-start`} tabIndex="-1" style={{ visibility, width: '600px', zIndex: 1050 }}>
                
                <div className="offcanvas-header border-bottom p-4 bg-white sticky-top">
                    <h5 className="offcanvas-title fw-bold text-dark text-truncate pe-2">
                        {loading ? 'Loading...' : (activeItem?.title || activeItem?.name || 'Details')}
                    </h5>
                    <button type="button" className="btn btn-light rounded-circle p-2" onClick={onClose}><X size={18} /></button>
                </div>
                
                <div className="offcanvas-body p-0 bg-light">
                    {loading && (
                        <div className="d-flex flex-column justify-content-center align-items-center h-100 text-muted">
                            <div className="spinner-border text-primary mb-2" role="status"></div>
                            <small>Fetching context...</small>
                        </div>
                    )}

                    {!loading && !error && activeItem && (
                        <div className="p-4">
                            {highlightText && (
                                <div className="alert alert-warning border-0 bg-warning bg-opacity-10 d-flex align-items-center gap-2 mb-3 py-2 px-3">
                                    <span className="badge bg-warning text-dark">Matched Text</span>
                                    <small className="text-dark fw-medium">"{highlightText}"</small>
                                </div>
                            )}
                            
                            <CVItemDisplayCard
                                item={activeItem}
                                itemType={activeType}
                                allSkills={resolvedSkills}
                                allAchievements={resolvedAchievements}
                                allExperiences={resolvedExperiences}
                                allEducation={resolvedEducation}
                                allHobbies={resolvedHobbies}
                                highlightText={highlightText} // <--- Pass it down
                            />
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};

export default CVItemPreviewModal;