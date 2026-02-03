// frontend/src/components/applications/CVItemPreviewModal.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import CVItemDisplayCard from './CVItemDisplayCard';
import { X } from 'lucide-react';

// Access API base safely
const API_BASE_URL = "http://localhost:8000/api";

const CVItemPreviewModal = ({
    isOpen,
    onClose,
    // 1. Legacy Props (Direct Object + Context Arrays from Parent)
    itemToPreview, 
    allSkills = [],
    allAchievements = [],
    allExperiences = [],
    allEducation = [],
    allHobbies = [],
    
    // 2. New Props (Fetch Mode for Forensics/Search)
    itemId,
    itemType,
    highlightText
}) => {
    // State to hold the "Rich Details" fetched from backend
    const [fetchedData, setFetchedData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Bootstrap Animation Classes
    const showClass = isOpen ? 'show' : '';
    const visibility = isOpen ? 'visible' : 'hidden';

    // --- FETCH LOGIC ---
    useEffect(() => {
        if (isOpen && !itemToPreview && itemId && itemType) {
            fetchRichDetails(itemId, itemType);
        } else if (isOpen) {
            // Reset if opening in legacy mode or closed
            setFetchedData(null); 
            setLoading(false);
            setError(null);
        }
    }, [isOpen, itemId, itemType, itemToPreview]);

    const fetchRichDetails = async (id, type) => {
        setLoading(true);
        setError(null);
        try {
            // Call the new "Hydrated" endpoint we created in the backend
            // This returns { item: {...}, skills: [...], achievements: [...] }
            const response = await axios.get(`${API_BASE_URL}/cv/item-details/${id}?type=${type}`);
            setFetchedData(response.data);
        } catch (err) {
            console.error("Failed to fetch item details:", err);
            setError("Could not load full item context.");
        } finally {
            setLoading(false);
        }
    };

    // --- HIGHLIGHTING LOGIC ---
    const processHighlighting = (originalItem) => {
        if (!highlightText || !originalItem) return originalItem;
        
        // Shallow copy to avoid mutating state
        const itemCopy = { ...originalItem };
        
        const highlightStr = (str) => {
            if (!str) return str;
            // Escape regex special characters
            const safeHighlight = highlightText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`(${safeHighlight})`, 'gi');
            // Use Bootstrap utility classes for the mark
            return str.replace(regex, '<mark class="bg-warning-subtle text-dark fw-bold rounded px-1 border border-warning border-opacity-25">$1</mark>');
        };

        // Apply to standard fields
        if (itemCopy.description) itemCopy.displayed_description = highlightStr(itemCopy.description);
        if (itemCopy.bullets) itemCopy.displayed_bullets = highlightStr(itemCopy.bullets);
        
        return itemCopy;
    };

    // --- DATA RESOLUTION ---
    // Determine which data source to use (Fetched vs Passed)
    
    // 1. The Item
    let activeItem = null;
    let activeType = null;

    if (itemToPreview) {
        activeItem = itemToPreview.item;
        activeType = itemToPreview.type;
    } else if (fetchedData) {
        activeItem = fetchedData.item;
        activeType = itemType;
    }

    // 2. The Context Arrays
    // If we fetched data, use the specific resolved lists from backend.
    // If not, use the global lists passed from parent (Backward Compatibility).
    const resolvedSkills = fetchedData ? (fetchedData.skills || []) : allSkills;
    const resolvedAchievements = fetchedData ? (fetchedData.achievements || []) : allAchievements;
    const resolvedExperiences = fetchedData ? (fetchedData.experiences || []) : allExperiences;
    const resolvedEducation = fetchedData ? (fetchedData.education || []) : allEducation;
    const resolvedHobbies = fetchedData ? (fetchedData.hobbies || []) : allHobbies;

    // 3. Apply Visuals
    const finalItem = activeItem ? processHighlighting(activeItem) : null;

    if (!isOpen) return null;

    return (
        <>
            {/* Backdrop */}
            <div 
                className={`offcanvas-backdrop fade ${showClass}`} 
                onClick={onClose}
                style={{ zIndex: 1045 }} 
            ></div>

            {/* Side Panel (Offcanvas) */}
            <div 
                className={`offcanvas offcanvas-end ${showClass} shadow-lg border-start`} 
                tabIndex="-1" 
                style={{ visibility, width: '600px', zIndex: 1050 }}
            >
                <div className="offcanvas-header border-bottom p-4 bg-white sticky-top">
                    <h5 className="offcanvas-title fw-bold text-dark text-truncate pe-2">
                        {loading ? 'Loading...' : (finalItem?.title || finalItem?.degree || finalItem?.name || 'Item Details')}
                    </h5>
                    <button 
                        type="button" 
                        className="btn btn-light rounded-circle p-2 d-flex align-items-center justify-content-center" 
                        onClick={onClose}
                        style={{ width: '32px', height: '32px' }}
                    >
                        <X size={18} />
                    </button>
                </div>
                
                <div className="offcanvas-body p-0 bg-light">
                    {loading && (
                        <div className="d-flex flex-column justify-content-center align-items-center h-100 text-muted">
                            <div className="spinner-border text-primary mb-2" role="status"></div>
                            <small>Fetching details...</small>
                        </div>
                    )}

                    {error && (
                        <div className="p-5 text-center">
                            <div className="text-danger mb-2 fw-bold">Error</div>
                            <p className="text-muted small">{error}</p>
                            <button className="btn btn-sm btn-outline-secondary mt-2" onClick={() => fetchRichDetails(itemId, itemType)}>Retry</button>
                        </div>
                    )}

                    {!loading && !error && finalItem && (
                        <div className="p-4">
                            {highlightText && (
                                <div className="alert alert-warning border-0 bg-warning bg-opacity-10 d-flex align-items-center gap-2 mb-3 py-2 px-3 shadow-sm">
                                    <span className="badge bg-warning text-dark">Match Found</span>
                                    <small className="text-dark fw-medium text-truncate">"{highlightText}"</small>
                                </div>
                            )}
                            
                            <CVItemDisplayCard
                                item={finalItem}
                                itemType={activeType}
                                
                                // Pass the resolved context lists (either fetched or prop-drilled)
                                allSkills={resolvedSkills}
                                allAchievements={resolvedAchievements}
                                allExperiences={resolvedExperiences}
                                allEducation={resolvedEducation}
                                allHobbies={resolvedHobbies}
                            />
                        </div>
                    )}
                </div>
            </div>
        </>
    );
};

export default CVItemPreviewModal;