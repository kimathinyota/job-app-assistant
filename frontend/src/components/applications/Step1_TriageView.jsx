// frontend/src/components/applications/Step1_TriageView.jsx
import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
    addMappingPair,
    deleteMappingPair,
    inferMappingPairs
} from '../../api/applicationClient';
import CVItemPreviewModal from './CVItemPreviewModal';

// --- (Shared) Helper Component: Loading Overlay ---
const LoadingSuggestions = ({ mode }) => (
    <div className="position-absolute w-100 h-100 d-flex justify-content-center align-items-center"
        style={{ background: 'rgba(255,255,255,0.8)', zIndex: 10, top: 0, left: 0 }}>
        <div className="spinner-border text-primary" role="status">
            <span className="visually-hidden">Loading...</span>
        </div>
        <span className="ms-2">Finding AI suggestions (using "{mode}" mode)...</span>
    </div>
);

// --- (Shared) Tuning Modes ---
const TUNING_MODES = {
    "super_eager": "Super Eager (Find all)",
    "eager_mode": "Eager (Find many)",
    "balanced_default": "Balanced (Recommended)",
    "picky_mode": "Picky (Fewer, better)",
    "super_picky": "Super Pîcky (Best only)"
};

// --- (Triage View) Group Definitions ---
const TRIAGE_GROUPS = [
    { id: 'experiences', title: 'Work Experience' },
    { id: 'projects', title: 'Projects' },
    { id: 'education', title: 'Education' },
    { id: 'hobbies', title: 'Hobbies' }
];

// --- Helper Function to initialize group object ---
const getInitialTriageGroups = () => {
    const initialGroups = {};
    TRIAGE_GROUPS.forEach(g => {
        initialGroups[g.id] = [];
    });
    initialGroups.other = [];
    return initialGroups;
};


// --- *** Triage View Group Card Components *** ---

const SuggestionSubItem = ({ suggestion, onAccept, onIgnore, isAccepting }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [editedAnnotation, setEditedAnnotation] = useState(suggestion.annotation || "Good conceptual match.");

    const handleAcceptClick = () => {
        onAccept(suggestion, editedAnnotation);
    };
    
    return (
        <li className="list-group-item">
            <p className="small mb-1">
                <strong>Requirement:</strong> {suggestion.feature_text}
            </p>
            <div className="alert alert-info p-2 small" role="alert">
                <div className="d-flex justify-content-between align-items-center">
                    <strong className="d-block">💡 AI Reason:</strong>
                    <button 
                        className="btn btn-sm btn-link p-0 text-dark" 
                        onClick={() => setIsEditing(!isEditing)}
                        title="Edit annotation"
                    >
                        {isEditing ? 'Cancel' : '✎ Edit'}
                    </button>
                </div>
                {isEditing ? (
                    <textarea 
                        className="form-control form-control-sm mt-2"
                        value={editedAnnotation}
                        onChange={(e) => setEditedAnnotation(e.target.value)}
                        rows={2}
                    />
                ) : (
                    <em className="d-block mt-1">{editedAnnotation}</em>
                )}
            </div>
            <div className="text-end">
                <button
                    type="button" className="btn btn-sm btn-outline-secondary me-2"
                    onClick={() => onIgnore(suggestion.id)} disabled={isAccepting}
                >
                    Ignore
                </button>
                <button
                    type="button" className="btn btn-sm btn-success"
                    onClick={handleAcceptClick}
                    disabled={isAccepting}
                >
                    {isAccepting ? "Accepting..." : "Accept"}
                </button>
            </div>
        </li>
    );
};

const TriageItemGroupCard = ({ group, onAccept, onIgnore, isAcceptingId, onPreview }) => (
    <div className="card mb-3 shadow-sm">
        <div className="card-header bg-light d-flex justify-content-between align-items-center fw-bold">
            {group.cvItemText}
            <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => onPreview(group.cvItem, group.cvItemType)}
                title="Preview Item"
            >
                👁️
            </button>
        </div>
        <ul className="list-group list-group-flush">
            {group.suggestions.map(sugg => (
                <SuggestionSubItem
                    key={sugg.id}
                    suggestion={sugg}
                    onAccept={onAccept}
                    onIgnore={onIgnore}
                    isAccepting={isAcceptingId === sugg.id}
                />
            ))}
        </ul>
    </div>
);

const AcceptedItemGroupCard = ({ group, onDelete, isDeletingId, onPreview }) => (
    <div className="card mb-3 shadow-sm">
        <div className="card-header bg-light d-flex justify-content-between align-items-center fw-bold">
            {group.cvItemText}
            <button
                type="button"
                className="btn btn-sm btn-outline-secondary"
                onClick={() => onPreview(group.cvItem, group.cvItemType)}
                title="Preview Item"
            >
                👁️
            </button>
        </div>
        <ul className="list-group list-group-flush">
            {group.pairs.map(pair => (
                <li key={pair.id} className="list-group-item d-flex justify-content-between align-items-center">
                    <div>
                        <p className="small mb-0">
                            <strong>Req:</strong> {pair.feature_text}
                        </p>
                        {pair.annotation && (
                            <p className="small fst-italic text-muted mb-0">
                                <strong>Note:</strong> {pair.annotation}
                            </p>
                        )}
                    </div>
                    <button
                        className="btn btn-danger btn-sm"
                        onClick={() => onDelete(pair.id)}
                        disabled={isDeletingId === pair.id}
                    >
                        {isDeletingId === pair.id ? "..." : "Delete"}
                    </button>
                </li>
            ))}
        </ul>
    </div>
);

// --- *** END: Triage Components *** ---


// --- The Main Component ---
const Step1_TriageView = ({ job, cv, mapping, onMappingChanged, onNext }) => {
    // --- Shared State ---
    const [viewMode, setViewMode] = useState('triage');
    const [tuningMode, setTuningMode] = useState('balanced_default');
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
    const [allSuggestions, setAllSuggestions] = useState([]);
    const [suggestionError, setSuggestionError] = useState(null);

    // --- Triage View State ---
    const [isAcceptingId, setIsAcceptingId] = useState(null);
    const [isDeletingId, setIsDeletingId] = useState(null);
    const [activeTriageCategory, setActiveTriageCategory] = useState('experiences');

    // --- Manual View State ---
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [previewItem, setPreviewItem] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [manualCategoryFilter, setManualCategoryFilter] = useState('all');


    // --- Shared Data Fetching ---
    const fetchSuggestions = useCallback(async (mode) => {
        setIsLoadingSuggestions(true);
        setSuggestionError(null);
        try {
            const res = await inferMappingPairs(mapping.id, mode);
            setAllSuggestions(res.data.map((s, i) => ({ ...s, id: `sugg-${i}` })));
        } catch (err) {
            console.error("Failed to fetch suggestions:", err);
            setSuggestionError("Could not load AI suggestions.");
        } finally {
            setIsLoadingSuggestions(false);
        }
    }, [mapping.id]);

    useEffect(() => {
        fetchSuggestions(tuningMode);
    }, [tuningMode, fetchSuggestions]);

    // --- CV Item Lookup Memo ---
    const cvItemLookups = useMemo(() => {
        const lookups = new Map();
        cv.experiences.forEach(item => lookups.set(item.id, { item, type: 'experiences', name: `${item.title} @ ${item.company}` }));
        cv.projects.forEach(item => lookups.set(item.id, { item, type: 'projects', name: `${item.title} (Project)` }));
        cv.education.forEach(item => lookups.set(item.id, { item, type: 'education', name: `${item.degree} @ ${item.institution}` }));
        cv.hobbies.forEach(item => lookups.set(item.id, { item, type: 'hobbies', name: `${item.name} (Hobby)` }));
        return lookups;
    }, [cv]);
    
    // --- Triage View Logic ---
    const acceptedPairs = useMemo(() => mapping.pairs || [], [mapping.pairs]);
    const acceptedPairKeys = useMemo(() => new Set(acceptedPairs.map(p => `${p.feature_id}-${p.context_item_id}`)), [acceptedPairs]);
    const suggestionsToReview = useMemo(() => allSuggestions.filter(s => !acceptedPairKeys.has(`${s.feature_id}-${s.context_item_id}`)), [allSuggestions, acceptedPairKeys]);

    const groupItemsByCvItem = (items, itemKey) => {
        const groups = getInitialTriageGroups();
        const cvItemMap = new Map();

        items.forEach(item => {
            const cvId = item.context_item_id;
            const lookup = cvItemLookups.get(cvId);
            
            const itemType = lookup ? lookup.type : item.context_item_type;
            const itemText = lookup ? lookup.name : item.context_item_text;
            const originalItem = lookup ? lookup.item : null;

            if (!cvItemMap.has(cvId)) {
                cvItemMap.set(cvId, {
                    cvItemId: cvId,
                    cvItem: originalItem,
                    cvItemText: itemText,
                    cvItemType: itemType,
                    [itemKey]: []
                });
            }
            cvItemMap.get(cvId)[itemKey].push(item);
        });

        cvItemMap.forEach(group => {
            if (groups[group.cvItemType]) {
                groups[group.cvItemType].push(group);
            } else {
                groups.other.push(group);
            }
        });
        
        return groups;
    };

    const groupedSuggestionsToReview = useMemo(
        () => groupItemsByCvItem(suggestionsToReview, 'suggestions'), 
        [suggestionsToReview, cvItemLookups]
    );
    
    const groupedAcceptedPairs = useMemo(
        () => groupItemsByCvItem(acceptedPairs, 'pairs'), 
        [acceptedPairs, cvItemLookups]
    );
    
    const allGroupedAcceptedPairs = useMemo(() => {
        return [
            ...groupedAcceptedPairs.experiences,
            ...groupedAcceptedPairs.projects,
            ...groupedAcceptedPairs.education,
            ...groupedAcceptedPairs.hobbies,
            ...groupedAcceptedPairs.other,
        ];
    }, [groupedAcceptedPairs]);

    const manuallyFilteredAcceptedGroups = useMemo(() => {
        if (manualCategoryFilter === 'all') {
            return allGroupedAcceptedPairs;
        }
        return groupedAcceptedPairs[manualCategoryFilter] || [];
    }, [manualCategoryFilter, allGroupedAcceptedPairs, groupedAcceptedPairs]);
    

    const handleTriageAccept = async (suggestion, newAnnotation) => {
        setIsAcceptingId(suggestion.id);
        try {
            await addMappingPair(
                mapping.id, 
                suggestion.feature_id, 
                suggestion.context_item_id, 
                suggestion.context_item_type, 
                newAnnotation, 
                suggestion.feature_text, 
                suggestion.context_item_text
            );
            await onMappingChanged();
        } catch (err) { alert(`Failed to accept pair: ${err.response?.data?.detail || err.message}`); }
        finally { setIsAcceptingId(null); }
    };

    const handleTriageIgnore = (suggestionId) => setAllSuggestions(prev => prev.filter(s => s.id !== suggestionId));
    
    const handleTriageDelete = async (pairId) => {
        setIsDeletingId(pairId);
        try {
            await deleteMappingPair(mapping.id, pairId);
            await onMappingChanged();
        } catch (err) { alert(`Failed to delete pair: ${err.response?.data?.detail || err.message}`); }
        finally { setIsDeletingId(null); }
    };

    const handleTriagePreview = (item, type) => {
        if (!item) {
            alert("Preview is not available for this item as its original data could not be found.");
            return;
        }
        setPreviewItem({ item, type });
        setIsModalOpen(true);
    };

    // --- Logic for Manual View ---
    const cvEvidenceList = useMemo(() => [
        ...cv.experiences.map(item => ({ id: item.id, type: 'experiences', text: `${item.title} @ ${item.company}`, item: item })),
        ...cv.projects.map(item => ({ id: item.id, type: 'projects', text: `${item.title} (Project)`, item: item })),
        ...cv.education.map(item => ({ id: item.id, type: 'education', text: `${item.degree} @ ${item.institution}`, item: item })),
        ...cv.hobbies.map(item => ({ id: item.id, type: 'hobbies', text: `${item.name} (Hobby)`, item: item })),
    ], [cv]);

    // Memos for Column 2 (CV Evidence) highlighting
    const pairedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set();
        return new Set(mapping.pairs.filter(p => p.feature_id === selectedReqId).map(p => p.context_item_id));
    }, [mapping.pairs, selectedReqId]);

    const suggestedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set(); 
        return new Set(allSuggestions.filter(s => s.feature_id === selectedReqId).map(s => s.context_item_id));
    }, [allSuggestions, selectedReqId]); 

    // Memos for Column 1 (Requirements) highlighting
    const pairedReqIds = useMemo(() => {
        if (!selectedContextId) return new Set();
        return new Set(mapping.pairs.filter(p => p.context_item_id === selectedContextId).map(p => p.feature_id));
    }, [mapping.pairs, selectedContextId]);

    const suggestedReqIds = useMemo(() => {
        if (!selectedContextId) return new Set();
        return new Set(allSuggestions.filter(s => s.context_item_id === selectedContextId).map(s => s.feature_id));
    }, [allSuggestions, selectedContextId]);

    useEffect(() => {
        if (selectedReqId && selectedContextId) {
            const matchingSuggestion = allSuggestions.find(s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId);
            setAnnotation(matchingSuggestion?.annotation || "");
        } else {
            setAnnotation("");
        }
    }, [selectedReqId, selectedContextId, allSuggestions]);

    const handleManualSelectReq = (reqId) => {
        setSelectedReqId(prev => (prev === reqId ? null : reqId));
    };
    
    const handleManualSelectContext = (item) => {
        if (selectedContextId === item.id) {
            setSelectedContextId(null);
            setSelectedContextType(null);
        } else {
            setSelectedContextId(item.id);
            setSelectedContextType(item.type);
        }
    };

    const handleManualClear = () => {
        setSelectedReqId(null);
        setSelectedContextId(null);
        setSelectedContextType(null);
        setAnnotation("");
    };
    const handleManualPreview = (e, item, type) => {
        e.stopPropagation();
        setPreviewItem({ item, type });
        setIsModalOpen(true);
    };
    
    const handleManualCreate = async () => {
        if (!selectedReqId || !selectedContextId) return;
        const alreadyExists = mapping.pairs.some(p => p.feature_id === selectedReqId && p.context_item_id === selectedContextId);
        if (alreadyExists) {
            alert("This pair already exists and cannot be added again. Duplicates are not allowed.");
            return;
        }
        setIsSubmitting(true);
        try {
            const matchingSuggestion = allSuggestions.find(s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId);
            await addMappingPair(mapping.id, selectedReqId, selectedContextId, selectedContextType, annotation, matchingSuggestion?.feature_text, matchingSuggestion?.context_item_text);
            await onMappingChanged();
            handleManualClear();
        } catch (err) { alert(`Failed to create pair: ${err.response?.data?.detail || err.message}`); }
        finally { setIsSubmitting(false); }
    };
    
    const handleManualDelete = async (pairId) => {
        handleTriageDelete(pairId);
    };

    const isPairAlreadyMade = useMemo(() => {
        if (!selectedReqId || !selectedContextId) return false;
        return mapping.pairs.some(p => p.feature_id === selectedReqId && p.context_item_id === selectedContextId);
    }, [selectedReqId, selectedContextId, mapping.pairs]);

    // --- Render Function ---
    return (
        <div style={{ position: 'relative' }}>
            {isLoadingSuggestions && <LoadingSuggestions mode={tuningMode} />}

            <h4 className="h5">Step 1: Map CV to Job Requirements</h4>
            
            <div className="d-flex justify-content-between align-items-center mb-3">
                <div className="d-flex align-items-center gap-2">
                    <label htmlFor="tuningModeSelect" className="form-label mb-0">AI Mode:</label>
                    <select
                        id="tuningModeSelect"
                        className="form-select form-select-sm"
                        value={tuningMode}
                        onChange={(e) => setTuningMode(e.target.value)}
                        disabled={isLoadingSuggestions}
                        style={{width: '200px'}}
                    >
                        {Object.entries(TUNING_MODES).map(([key, label]) => (
                            <option key={key} value={key}>{label}</option>
                        ))}
                    </select>
                </div>
                
                <button
                    className="btn btn-outline-secondary"
                    onClick={() => setViewMode(v => (v === 'triage' ? 'manual' : 'triage'))}
                    style={{minWidth: '180px'}}
                >
                    {viewMode === 'triage' ? 'Switch to Manual Mode' : 'Switch to Triage Mode'}
                </button>
            </div>
            {suggestionError && <div className="alert alert-danger small">{suggestionError}</div>}

            {viewMode === 'triage' && (
                <ul className="nav nav-pills nav-fill mx-auto">
                    {TRIAGE_GROUPS.map(groupCat => {
                        const count = groupedSuggestionsToReview[groupCat.id].length;
                        return (
                            <li className="nav-item" key={groupCat.id}>
                                <button 
                                    className={`nav-link ${activeTriageCategory === groupCat.id ? 'active' : ''}`}
                                    onClick={() => setActiveTriageCategory(groupCat.id)}
                                >
                                    {groupCat.title} 
                                    {count > 0 && <span className="badge bg-light text-dark ms-2">{count}</span>}
                                </button>
                            </li>
                        );
                    })}
                </ul>
            )}
            
            {viewMode === 'triage' ? (
                <div className="row" style={{ minHeight: '400px', filter: isLoadingSuggestions ? 'blur(2px)' : 'none' }}>
                    <div className="col-md-7">
                        <h6 className="border-bottom pb-2">
                            To Review: {TRIAGE_GROUPS.find(g => g.id === activeTriageCategory)?.title}
                        </h6>
                        <div className="triage-pane-content" style={{ maxHeight: '500px', overflowY: 'auto', paddingRight: '10px' }}>
                            {(() => {
                                const suggestionGroups = groupedSuggestionsToReview[activeTriageCategory];
                                if (suggestionGroups.length === 0) {
                                    return <p className="small text-muted fst-italic">No new suggestions for this category.</p>;
                                }
                                return suggestionGroups.map(group => (
                                    <TriageItemGroupCard 
                                        key={group.cvItemId} 
                                        group={group}
                                        onAccept={handleTriageAccept}
                                        onIgnore={handleTriageIgnore}
                                        isAcceptingId={isAcceptingId}
                                        onPreview={handleTriagePreview}
                                    />
                                ));
                            })()}
                        </div>
                    </div>
                    
                    <div className="col-md-5">
                        <h6 className="border-bottom pb-2">
                            Mapped: {TRIAGE_GROUPS.find(g => g.id === activeTriageCategory)?.title}
                        </h6>
                        <div className="triage-pane-content" style={{ maxHeight: '500px', overflowY: 'auto', paddingRight: '10px' }}>
                            {(() => {
                                const pairGroups = groupedAcceptedPairs[activeTriageCategory];
                                if (pairGroups.length === 0) {
                                    return <p className="small text-muted fst-italic">No mapped pairs for this category.</p>;
                                }
                                return pairGroups.map(group => (
                                    <AcceptedItemGroupCard 
                                        key={group.cvItemId} 
                                        group={group}
                                        onDelete={handleTriageDelete}
                                        isDeletingId={isDeletingId}
                                        onPreview={handleTriagePreview}
                                    />
                                ));
                            })()}
                        </div>
                    </div>
                </div>
            ) : (
                // --- Manual View ---
                <div className="row" style={{ minHeight: '400px', filter: isLoadingSuggestions ? 'blur(2px)' : 'none' }}>
                    
                    {/* --- *** START: Column 1: Job Requirements (with CORRECTED logic) *** --- */}
                    <div className="col-4">
                        <h6 className="border-bottom pb-2">Job Requirements</h6>
                        <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                            {job.features.map(req => {
                                const isSelected = selectedReqId === req.id;
                                const isPaired = selectedContextId && pairedReqIds.has(req.id);
                                const isSuggested = selectedContextId && suggestedReqIds.has(req.id) && !isPaired;
                                const isAssociated = isPaired || isSuggested; // Is it associated with the *selected* CV item?
                                const isDisabled = selectedContextId && isPaired && !isSelected;

                                let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                                
                                if (isSelected) {
                                    itemClass += ' list-group-item-primary'; // Solid Blue
                                } else if (isAssociated) { // Associated with selected CV item
                                    itemClass += ' list-group-item-success-subtle'; // Light Green
                                }
                                
                                if (isDisabled) {
                                    itemClass += ' disabled';
                                }

                                return (
                                    <button 
                                        key={req.id} 
                                        type="button" 
                                        className={itemClass} 
                                        onClick={() => handleManualSelectReq(req.id)}
                                        disabled={isDisabled}
                                    >
                                        <span className="text-start">{req.description}</span>
                                        {/* Show appropriate icon based on association */}
                                        {isPaired && <span className="badge bg-success-subtle text-success-emphasis rounded-pill">⚭</span>}
                                        {isSuggested && <span className="badge bg-success-subtle text-success-emphasis rounded-pill">💡</span>}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                    {/* --- *** END: Column 1 *** --- */}

                    {/* --- *** START: Column 2: CV Evidence (with CORRECTED logic) *** --- */}
                    <div className="col-4">
                        <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                        <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                            {cvEvidenceList.map(item => {
                                const isSelected = selectedContextId === item.id;
                                const isPaired = selectedReqId && pairedCvItemIds.has(item.id);
                                const isSuggested = selectedReqId && suggestedCvItemIds.has(item.id) && !isPaired;
                                const isAssociated = isPaired || isSuggested; // Is it associated with the *selected* Req?
                                const isDisabled = selectedReqId && isPaired && !isSelected;
                                
                                let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                                
                                if (isSelected) {
                                    itemClass += ' list-group-item-success'; // Solid Green
                                } else if (isAssociated) { // Associated with selected Req
                                    itemClass += ' list-group-item-primary-subtle'; // Light Blue
                                }

                                if (isDisabled) {
                                    itemClass += ' disabled';
                                }
                                
                                return (
                                    <button
                                        key={item.id}
                                        type="button"
                                        className={itemClass} 
                                        onClick={() => handleManualSelectContext(item)} 
                                        style={{ cursor: 'pointer' }}
                                        disabled={isDisabled}
                                    >
                                        <span className="text-start" style={{ flex: 1, marginRight: '10px' }}>
                                            {item.text}
                                            {/* Show appropriate icon based on association */}
                                            {isPaired && <span className="badge bg-primary-subtle text-primary-emphasis rounded-pill ms-2">⚭</span>}
                                            {isSuggested && <span className="badge bg-primary-subtle text-primary-emphasis rounded-pill ms-2">💡</span>}
                                        </span>
                                        <button
                                            type="button" className={`btn btn-sm ${isSelected ? 'btn-outline-light' : 'btn-outline-secondary'}`}
                                            onClick={(e) => handleManualPreview(e, item.item, item.type)} title="Preview Item" style={{ zIndex: 5 }}
                                            disabled={isDisabled}
                                        >👁️</button>
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                    {/* --- *** END: Column 2 *** --- */}

                    {/* Column 3: Pairing Box & Mapped List */}
                    <div className="col-4">
                        <h6 className="border-bottom pb-2">Manual Pairing</h6>
                        {(selectedReqId || selectedContextId) && (
                            <button className="btn btn-sm btn-outline-secondary w-100 mb-2" onClick={handleManualClear}>
                                Clear Selection
                            </button>
                        )}
                        <div className="mb-2">
                            <textarea
                                className="form-control form-control-sm" rows="2"
                                placeholder={selectedReqId && selectedContextId ? "Annotation (auto-filled for suggestions)" : "Select a requirement and a CV item"}
                                value={annotation} 
                                onChange={(e) => setAnnotation(e.target.value)}
                                disabled={!selectedReqId || !selectedContextId}
                            />
                        </div>
                        <button
                            className="btn btn-success w-100 mb-3"
                            disabled={!selectedReqId || !selectedContextId || isSubmitting || isPairAlreadyMade}
                            onClick={handleManualCreate}
                        >
                            {isSubmitting ? 'Pairing...' : (isPairAlreadyMade ? 'Pair Already Exists' : 'Create Pair')}
                        </button>
                        
                        <div className="d-flex justify-content-between align-items-center border-bottom pb-2">
                            <h6 className="mb-0">Mapped Pairs ({mapping.pairs.length})</h6>
                            <select 
                                className="form-select form-select-sm" 
                                style={{width: '150px'}}
                                value={manualCategoryFilter}
                                onChange={(e) => setManualCategoryFilter(e.target.value)}
                            >
                                <option value="all">All Categories</option>
                                {TRIAGE_GROUPS.map(g => (
                                    <option key={g.id} value={g.id}>{g.title}</option>
                                ))}
                            </select>
                        </div>
                        
                        <div style={{ maxHeight: '250px', overflowY: 'auto' }} className="mt-2">
                            {manuallyFilteredAcceptedGroups.length === 0 ? (
                                <p className="small text-muted fst-italic">
                                    {manualCategoryFilter === 'all' 
                                        ? "No pairs mapped yet." 
                                        : "No pairs in this category."}
                                </p>
                            ) : (
                                manuallyFilteredAcceptedGroups.map(group => (
                                    <AcceptedItemGroupCard 
                                        key={group.cvItemId} 
                                        group={group}
                                        onDelete={handleManualDelete}
                                        isDeletingId={isDeletingId}
                                        onPreview={handleTriagePreview}
                                    />
                                ))
                            )}
                        </div>
                    </div>
                </div>
            )}

            {/* --- Shared Navigation --- */}
            <div className="text-end mt-4">
                <button className="btn btn-primary" onClick={onNext} disabled={isLoadingSuggestions}>
                    Next: Review CV &gt;
                </button>
            </div>

            {/* --- (Triage & Manual) Preview Modal --- */}
            {isModalOpen && (
                <CVItemPreviewModal
                    isOpen={isModalOpen}
                    onClose={() => setIsModalOpen(false)}
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

export default Step1_TriageView;