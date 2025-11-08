// frontend/src/components/applications/Step1_Mapping.jsx
import React, { useState, useMemo, useEffect, useCallback } from 'react';
import { 
    addMappingPair, 
    deleteMappingPair,
    inferMappingPairs 
} from '../../api/applicationClient.js'; // <-- FIX: Added .js extension
import CVItemPreviewModal from './CVItemPreviewModal.jsx'; // <-- FIX: Added .jsx extension

// A simple loading overlay
const LoadingSuggestions = ({ mode }) => (
    <div className="position-absolute w-100 h-100 d-flex justify-content-center align-items-center" 
         style={{ background: 'rgba(255,255,255,0.8)', zIndex: 10 }}>
        <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
        </div>
        <span className="ms-2">Finding AI suggestions (using "{mode}" mode)...</span>
    </div>
);

// Define the tuning modes in the frontend
const TUNING_MODES = {
    "super_eager": "Super Eager (Find all)",
    "eager_mode": "Eager (Find many)",
    "balanced_default": "Balanced (Recommended)",
    "picky_mode": "Picky (Fewer, better)",
    "super_picky": "Super Picky (Best only)"
};


const Step1_Mapping = ({ job, cv, mapping, onMappingChanged, onNext }) => {
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [previewItem, setPreviewItem] = useState(null); 
    const [isModalOpen, setIsModalOpen] = useState(false);

    const [tuningMode, setTuningMode] = useState('balanced_default'); 
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
    const [allSuggestions, setAllSuggestions] = useState([]);
    const [suggestionError, setSuggestionError] = useState(null);

    const fetchSuggestions = useCallback(async (mode) => {
        setIsLoadingSuggestions(true);
        setSuggestionError(null);
        try {
            const res = await inferMappingPairs(mapping.id, mode);
            setAllSuggestions(res.data);
        } catch (err) {
            console.error("Failed to fetch suggestions:", err);
            setSuggestionError("Could not load AI suggestions.");
        } finally {
            setIsLoadingSuggestions(false);
        }
    }, [mapping.id]); 

    useEffect(() => {
        fetchSuggestions(tuningMode);
    }, [mapping.id, tuningMode, fetchSuggestions]);


    // --- 1. MEMOS for Highlighting ---
    
    // Get all CV items already paired with the selected Req
    const pairedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set();
        return new Set(
            mapping.pairs
                .filter(p => p.feature_id === selectedReqId)
                .map(p => p.context_item_id)
        );
    }, [mapping.pairs, selectedReqId]);

    // Get all CV items suggested for the selected Req
    // *** FIX: Only show suggestions if the *other* column is NOT selected ***
    const suggestedCvItemIds = useMemo(() => {
        // If Col 2 is already selected, don't show new suggestions
        if (!selectedReqId || selectedContextId) return new Set();
        return new Set(
            allSuggestions
                .filter(s => s.feature_id === selectedReqId)
                .map(s => s.context_item_id)
        );
    }, [allSuggestions, selectedReqId, selectedContextId]); // Dependency added

    // Get all Reqs already paired with the selected CV item
    const pairedReqIds = useMemo(() => {
        if (!selectedContextId) return new Set();
        return new Set(
            mapping.pairs
                .filter(p => p.context_item_id === selectedContextId)
                .map(p => p.feature_id)
        );
    }, [mapping.pairs, selectedContextId]);

    // Get all Reqs suggested for the selected CV item
    // *** FIX: Only show suggestions if the *other* column is NOT selected ***
    const suggestedReqIds = useMemo(() => {
        // If Col 1 is already selected, don't show new suggestions
        if (!selectedContextId || selectedReqId) return new Set();
        return new Set(
            allSuggestions
                .filter(s => s.context_item_id === selectedContextId)
                .map(s => s.feature_id)
        );
    }, [allSuggestions, selectedContextId, selectedReqId]); // Dependency added

    
    // Pre-fill annotation (Unchanged, this is correct)
    useEffect(() => {
        if (selectedReqId && selectedContextId) {
            const matchingSuggestion = allSuggestions.find(
                s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId
            );
            if (matchingSuggestion) {
                setAnnotation(matchingSuggestion.annotation || "");
            } else {
                setAnnotation("");
            }
        } else {
            setAnnotation(""); 
        }
    }, [selectedReqId, selectedContextId, allSuggestions]);


    // (Existing Memos: cvEvidenceList, reqTextMap, etc. remain unchanged)
    const cvEvidenceList = useMemo(() => [
        ...cv.experiences.map(item => ({
            id: item.id,
            type: 'experiences',
            text: `${item.title} @ ${item.company}`,
            item: item 
        })),
        ...cv.projects.map(item => ({
            id: item.id,
            type: 'projects',
            text: `${item.title} (Project)`,
            item: item 
        })),
        ...cv.education.map(item => ({
            id: item.id,
            type: 'education',
            text: `${item.degree} @ ${item.institution}`,
            item: item 
        })),
        ...cv.hobbies.map(item => ({
            id: item.id,
            type: 'hobbies',
            text: `${item.name} (Hobby)`,
            item: item 
        })),
    ], [cv.experiences, cv.projects, cv.education, cv.hobbies]);


    // --- 2. SIMPLE Click Handlers (No more toggle) ---
    // (This was correct in the previous version)
    const handleSelectReq = (reqId) => {
        setSelectedReqId(reqId);
    };

    const handleSelectContextItem = (item) => {
        setSelectedContextId(item.id);
        setSelectedContextType(item.type);
    };

    const handleClearSelection = () => {
        setSelectedReqId(null);
        setSelectedContextId(null);
        setSelectedContextType(null);
        setAnnotation("");
    };

    // (Other handlers: handlePreviewClick, handleCloseModal, handleDeletePair)
    const handlePreviewClick = (e, item, type) => {
        e.stopPropagation(); 
        setPreviewItem({ item, type });
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setPreviewItem(null);
    };
    
    const handleDeletePair = async (pairId) => {
        try {
            await deleteMappingPair(mapping.id, pairId);
            await onMappingChanged(); 
        } catch (err) {
             alert(`Failed to delete pair: ${err.response?.data?.detail || err.message}`);
             console.error(err);
        }
    };

    // --- 3. Handle Create Pair (and reset state) ---
    const handleCreatePair = async () => {
        if (!selectedReqId || !selectedContextId || !selectedContextType) return;
        
        const alreadyExists = mapping.pairs.some(
            p => p.feature_id === selectedReqId && 
                 p.context_item_id === selectedContextId && 
                 !p.annotation
        );

        if (alreadyExists && !annotation.trim()) {
            alert("This pair already exists. Please add an annotation to create a duplicate link with a note.");
            return;
        }
        
        setIsSubmitting(true);
        try {
            const matchingSuggestion = allSuggestions.find(
                s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId
            );

            await addMappingPair(
                mapping.id, 
                selectedReqId, 
                selectedContextId, 
                selectedContextType, 
                annotation,
                matchingSuggestion?.feature_text,
                matchingSuggestion?.context_item_text
            );
            await onMappingChanged(); 
            
            // Reset selections after successful creation
            handleClearSelection(); 
        } catch (err) {
            alert(`Failed to create pair: ${err.response?.data?.detail || err.message}`);
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    };


    return (
        <div style={{ position: 'relative' }}>
            {isLoadingSuggestions && <LoadingSuggestions mode={tuningMode} />}
            
            <h4 className="h5">Step 1: Map CV to Job Requirements</h4>
            <p className="text-muted">
                Click a requirement, then click the CV item that proves it. 
                AI suggestions are highlighted with a 💡.
            </p>

            {/* Tuning Mode Selector (Unchanged) */}
            <div className="row mb-3 align-items-center">
                <div className="col-auto">
                    <label htmlFor="tuningModeSelect" className="form-label mb-0">
                        AI Suggestion Mode:
                    </label>
                </div>
                <div className="col-md-4">
                    <select
                        id="tuningModeSelect"
                        className="form-select form-select-sm"
                        value={tuningMode}
                        onChange={(e) => setTuningMode(e.target.value)}
                        disabled={isLoadingSuggestions}
                    >
                        {Object.entries(TUNING_MODES).map(([key, label]) => (
                            <option key={key} value={key}>{label}</option>
                        ))}
                    </select>
                </div>
                <div className="col">
                    {suggestionError && <span className="text-danger small">{suggestionError}</span>}
                </div>
            </div>
            
            <div className="row" style={{ minHeight: '400px', filter: isLoadingSuggestions ? 'blur(2px)' : 'none' }}>
                {/* --- 4. Panel 1: Job Requirements (with new logic) --- */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Job Requirements</h6>
                    <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {job.features.map(req => {
                            const isSelected = selectedReqId === req.id;
                            const isPaired = selectedContextId && pairedReqIds.has(req.id);
                            const isSuggested = selectedContextId && suggestedReqIds.has(req.id);

                            let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                            
                            if (isSelected) {
                                itemClass += ' active'; // This is the selected one
                            } else if (isPaired) {
                                itemClass += ' list-group-item-success'; // Paired with selected CV item
                            } else if (isSuggested) {
                                itemClass += ' list-group-item-info'; // Suggested for selected CV item
                            }
                            // *** FIX: REMOVED ALL 'opacity-50' and 'isDisabled' logic ***
                            
                            return (
                                <button
                                    key={req.id}
                                    type="button"
                                    className={itemClass}
                                    onClick={() => handleSelectReq(req.id)}
                                >
                                    {req.description}
                                    {isSuggested && <span>💡</span>}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* --- 5. Panel 2: Your CV Evidence (with new logic) --- */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                    <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {cvEvidenceList.map(item => {
                            const isSelected = selectedContextId === item.id;
                            const isPaired = selectedReqId && pairedCvItemIds.has(item.id);
                            const isSuggested = selectedReqId && suggestedCvItemIds.has(item.id);

                            let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';

                            if (isSelected) {
                                itemClass += ' active'; // This is the selected one
                            } else if (isPaired) {
                                itemClass += ' list-group-item-success'; // Paired with selected Req
                            } else if (isSuggested) {
                                itemClass += ' list-group-item-info'; // Suggested for selected Req
                            }
                            // *** FIX: REMOVED ALL 'opacity-50' and 'isDisabled' logic ***
                            
                            return (
                                <div
                                    key={item.id}
                                    className={itemClass}
                                    onClick={() => handleSelectContextItem(item)}
                                    style={{ cursor: 'pointer' }} // Always clickable
                                >
                                    <span style={{ flex: 1, marginRight: '10px' }}>
                                        {item.text}
                                        {isSuggested && <span className="ms-2">💡</span>}
                                    </span>
                                    <button
                                        type="button"
                                        className={`btn btn-sm ${isSelected ? 'btn-outline-light' : 'btn-outline-secondary'}`}
                                        onClick={(e) => handlePreviewClick(e, item.item, item.type)}
                                        title="Preview Item"
                                        style={{ zIndex: 5 }} 
                                    >
                                        👁️
                                    </button>
                                </div>
                            );
                        })}
                    </div>
                </div>

                {/* Panel 3: Mapped Pairs */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Manual Pairing</h6>
                    {/* --- 6. NEW: Clear Selection Button (only show if needed) --- */}
                    {(selectedReqId || selectedContextId) && ( // Show if *either* is selected
                         <button 
                            className="btn btn-sm btn-outline-secondary w-100 mb-2"
                            onClick={handleClearSelection}
                         >
                            Clear Selection
                         </button>
                    )}
                    <div className="mb-2">
                        <textarea
                            className="form-control form-control-sm"
                            rows="2"
                            placeholder={
                                selectedReqId && selectedContextId 
                                ? "Annotation (auto-filled for suggestions)" 
                                : "Select a requirement and a CV item"
                            }
                            value={annotation}
                            onChange={(e) => setAnnotation(e.target.value)}
                            disabled={!selectedReqId || !selectedContextId} // Keep this disable logic
                        />
                    </div>
                    <button 
                        className="btn btn-success w-100 mb-3"
                        disabled={!selectedReqId || !selectedContextId || isSubmitting}
                        onClick={handleCreatePair}
                    >
                        {isSubmitting ? 'Pairing...' : 'Create Pair'}
                    </button>
                    
                    <h6 className="border-bottom pb-2">
                        ✅ Mapped Pairs ({mapping.pairs.length})
                    </h6>
                    <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {mapping.pairs.length === 0 && (
                            <p className="small text-muted fst-italic">No pairs mapped yet.</p>
                        )}
                        {mapping.pairs.map(pair => (
                            <div key={pair.id} className="card card-body p-3 mb-2 shadow-sm">
                                <p className="small mb-1">
                                    <strong>Req:</strong> {pair.feature_text}
                                </p>
                                <p className="small mb-2">
                                    <strong>Maps to: </strong> {pair.context_item_text}
                                </p>
                                
                                {pair.annotation && (
                                    <p className="small fst-italic border-top pt-2 mt-2 mb-2">
                                        <strong>Note:</strong> {pair.annotation}
                                    </p>
                                )}
                                
                                <button 
                                    className="btn btn-danger btn-sm"
                                    onClick={() => handleDeletePair(pair.id)}
                                >
                                    Ungroup
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
            
            <div className="text-end mt-4">
                <button className="btn btn-primary" onClick={onNext} disabled={isLoadingSuggestions}>
                    Next: Review CV &gt;
                </button>
            </div>

            {isModalOpen && (
                <CVItemPreviewModal
                    isOpen={isModalOpen}
                    onClose={handleCloseModal}
                    itemToPreview={previewItem}
                    allSkills={cv.skills}
                    allAchievements={cv.achievements}
                    allExperiences={cv.experiences}
                    allEducation={cv.education}
                />
            )}
        </div>
    );
};

export default Step1_Mapping;