// frontend/src/components/applications/Step1_Mapping.jsx
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

// --- (Triage View) Helper: Suggestion Card ---
const SuggestionCard = ({ suggestion, onAccept, onIgnore, isAccepting }) => (
    <div className="card card-body p-3 mb-2 shadow-sm">
        <p className="small mb-1">
            <strong>Requirement:</strong> {suggestion.feature_text}
        </p>
        <p className="small mb-2">
            <strong>CV Evidence: </strong> {suggestion.context_item_text}
        </p>
        <div className="alert alert-info p-2" role="alert">
            <strong className="d-block small">💡 AI Reason:</strong>
            <p className="small fst-italic mb-0">{suggestion.annotation || "Good conceptual match."}</p>
        </div>
        <div className="d-flex justify-content-end gap-2">
            <button
                type="button" className="btn btn-sm btn-outline-secondary"
                onClick={() => onIgnore(suggestion.id)} disabled={isAccepting}
            >
                Ignore
            </button>
            <button
                type="button" className="btn btn-sm btn-success"
                onClick={() => onAccept(suggestion)} disabled={isAccepting}
            >
                {isAccepting ? "Accepting..." : "Accept"}
            </button>
        </div>
    </div>
);

// --- (Triage View) Helper: Accepted Card ---
const AcceptedPairCard = ({ pair, onDelete, isDeleting }) => (
    <div className="card card-body p-3 mb-2 shadow-sm bg-light">
        <p className="small mb-1"><strong>Req:</strong> {pair.feature_text}</p>
        <p className="small mb-2"><strong>Maps to: </strong> {pair.context_item_text}</p>
        {pair.annotation && (
            <p className="small fst-italic border-top pt-2 mt-2 mb-2">
                <strong>Note:</strong> {pair.annotation}
            </p>
        )}
        <button
            className="btn btn-danger btn-sm"
            onClick={() => onDelete(pair.id)} disabled={isDeleting}
        >
            {isDeleting ? "Deleting..." : "Delete Pair"}
        </button>
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
    { id: 'experience', title: 'Work Experience' },
    { id: 'project', title: 'Projects' },
    { id: 'education', title: 'Education' },
    { id: 'hobby', title: 'Hobbies' }
];

// --- Helper Function to initialize group object ---
const getInitialTriageGroups = () => {
    const initialGroups = {};
    TRIAGE_GROUPS.forEach(g => {
        initialGroups[g.id] = [];
    });
    initialGroups.other = []; // Add a fallback group
    return initialGroups;
};

// --- The Main Component ---
// (FIX: Renamed component to match filename)
const Step1_TriageView = ({ job, cv, mapping, onMappingChanged, onNext }) => {
    // --- Shared State ---
    const [viewMode, setViewMode] = useState('triage'); // 'triage' or 'manual'
    const [tuningMode, setTuningMode] = useState('balanced_default');
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
    const [allSuggestions, setAllSuggestions] = useState([]);
    const [suggestionError, setSuggestionError] = useState(null);

    // --- Triage View State ---
    const [isAcceptingId, setIsAcceptingId] = useState(null);
    const [isDeletingId, setIsDeletingId] = useState(null);

    // --- Manual View State ---
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [previewItem, setPreviewItem] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

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

    // --- Logic for Triage View ---
    const acceptedPairs = useMemo(() => mapping.pairs || [], [mapping.pairs]);
    const acceptedPairKeys = useMemo(() => new Set(acceptedPairs.map(p => `${p.feature_id}-${p.context_item_id}`)), [acceptedPairs]);
    const suggestionsToReview = useMemo(() => allSuggestions.filter(s => !acceptedPairKeys.has(`${s.feature_id}-${s.context_item_id}`)), [allSuggestions, acceptedPairKeys]);

    // --- Grouping logic for Triage View ---
    const groupedSuggestionsToReview = useMemo(() => {
        const groups = getInitialTriageGroups();
        suggestionsToReview.forEach(s => {
            if (groups[s.context_item_type]) {
                groups[s.context_item_type].push(s);
            } else {
                groups.other.push(s);
            }
        });
        return groups;
    }, [suggestionsToReview]);

    const groupedAcceptedPairs = useMemo(() => {
        const groups = getInitialTriageGroups();
        acceptedPairs.forEach(p => {
            if (groups[p.context_item_type]) {
                groups[p.context_item_type].push(p);
            } else {
                groups.other.push(p);
            }
        });
        return groups;
    }, [acceptedPairs]);
    // --- END: Grouping logic ---

    const handleTriageAccept = async (suggestion) => {
        setIsAcceptingId(suggestion.id);
        try {
            await addMappingPair(mapping.id, suggestion.feature_id, suggestion.context_item_id, suggestion.context_item_type, suggestion.annotation, suggestion.feature_text, suggestion.context_item_text);
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

    // --- Logic for Manual View ---
    const cvEvidenceList = useMemo(() => [
        ...cv.experiences.map(item => ({ id: item.id, type: 'experience', text: `${item.title} @ ${item.company}`, item: item })),
        ...cv.projects.map(item => ({ id: item.id, type: 'project', text: `${item.title} (Project)`, item: item })),
        ...cv.education.map(item => ({ id: item.id, type: 'education', text: `${item.degree} @ ${item.institution}`, item: item })),
        ...cv.hobbies.map(item => ({ id: item.id, type: 'hobby', text: `${item.name} (Hobby)`, item: item })),
    ], [cv]);

    const pairedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set();
        return new Set(mapping.pairs.filter(p => p.feature_id === selectedReqId).map(p => p.context_item_id));
    }, [mapping.pairs, selectedReqId]);

    const suggestedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set(); 
        return new Set(allSuggestions.filter(s => s.feature_id === selectedReqId).map(s => s.context_item_id));
    }, [allSuggestions, selectedReqId]); 

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
        try {
            await deleteMappingPair(mapping.id, pairId);
            await onMappingChanged();
        } catch (err) { alert(`Failed to delete pair: ${err.response?.data?.detail || err.message}`); }
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
                >
                    {viewMode === 'triage' ? 'Switch to Manual Mode' : 'Switch to Triage Mode'}
                </button>
            </div>
            {suggestionError && <div className="alert alert-danger small">{suggestionError}</div>}


            {/* --- Conditional View Rendering --- */}
            
            {viewMode === 'triage' ? (
                // --- Triage View (NOW GROUPED) ---
                <div className="row" style={{ minHeight: '400px', filter: isLoadingSuggestions ? 'blur(2px)' : 'none' }}>
                    
                    {/* --- Pane 1: To Review (WITH NEW SCROLLING) --- */}
                    <div className="col-md-7">
                        <h6 className="border-bottom pb-2">To Review ({suggestionsToReview.length})</h6>
                        {/* REMOVED parent scrolling div */}
                        {suggestionsToReview.length === 0 && !isLoadingSuggestions ? (
                            <p className="small text-muted fst-italic">No new suggestions. Try a different tuning mode or switch to manual.</p>
                        ) : (
                            <>
                                {/* --- Render each group with its OWN scroller --- */}
                                {TRIAGE_GROUPS.map(group => {
                                    const groupSuggestions = groupedSuggestionsToReview[group.id];
                                    return (
                                        <div key={group.id} className="mb-3">
                                            <h6 className="text-muted border-bottom pb-1">{group.title} ({groupSuggestions.length})</h6>
                                            {/* ADDED child scrolling div */}
                                            <div style={{ maxHeight: '250px', overflowY: 'auto', paddingRight: '10px' }}>
                                                {groupSuggestions.length === 0 ? (
                                                    <p className="small text-muted fst-italic">No suggestions for this category.</p>
                                                ) : (
                                                    groupSuggestions.map(sugg => (
                                                        <SuggestionCard 
                                                            key={sugg.id} 
                                                            suggestion={sugg} 
                                                            onAccept={handleTriageAccept} 
                                                            onIgnore={handleTriageIgnore} 
                                                            isAccepting={isAcceptingId === sugg.id} 
                                                        />
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    );
                                })}
                                {/* --- Render "Other" group with its OWN scroller --- */}
                                {groupedSuggestionsToReview.other.length > 0 && (
                                    <div key="other" className="mb-3">
                                        <h6 className="text-muted border-bottom pb-1">Other ({groupedSuggestionsToReview.other.length})</h6>
                                        <div style={{ maxHeight: '250px', overflowY: 'auto', paddingRight: '10px' }}>
                                            {groupedSuggestionsToReview.other.map(sugg => (
                                                <SuggestionCard 
                                                    key={sugg.id} 
                                                    suggestion={sugg} 
                                                    onAccept={handleTriageAccept} 
                                                    onIgnore={handleTriageIgnore} 
                                                    isAccepting={isAcceptingId === sugg.id} 
                                                />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                    
                    {/* --- Pane 2: Mapped Pairs (WITH NEW SCROLLING) --- */}
                    <div className="col-md-5">
                        <h6 className="border-bottom pb-2">Mapped Pairs ({acceptedPairs.length})</h6>
                        {/* REMOVED parent scrolling div */}
                        {acceptedPairs.length === 0 ? (
                            <p className="small text-muted fst-italic">No pairs mapped yet.</p>
                        ) : (
                            <>
                                {/* --- Render each group with its OWN scroller --- */}
                                {TRIAGE_GROUPS.map(group => {
                                    const groupPairs = groupedAcceptedPairs[group.id];
                                    return (
                                        <div key={group.id} className="mb-3">
                                            <h6 className="text-muted border-bottom pb-1">{group.title} ({groupPairs.length})</h6>
                                            {/* ADDED child scrolling div */}
                                            <div style={{ maxHeight: '250px', overflowY: 'auto', paddingRight: '10px' }}>
                                                {groupPairs.length === 0 ? (
                                                     <p className="small text-muted fst-italic">No mapped pairs for this category.</p>
                                                ) : (
                                                    groupPairs.map(pair => (
                                                        <AcceptedPairCard 
                                                            key={pair.id} 
                                                            pair={pair} 
                                                            onDelete={handleTriageDelete} 
                                                            isDeleting={isDeletingId === pair.id} 
                                                        />
                                                    ))
                                                )}
                                            </div>
                                        </div>
                                    );
                                })}
                                {/* --- Render "Other" group with its OWN scroller --- */}
                                {groupedAcceptedPairs.other.length > 0 && (
                                    <div key="other" className="mb-3">
                                        <h6 className="text-muted border-bottom pb-1">Other ({groupedAcceptedPairs.other.length})</h6>
                                        <div style={{ maxHeight: '250px', overflowY: 'auto', paddingRight: '10px' }}>
                                            {groupedAcceptedPairs.other.map(pair => (
                                                <AcceptedPairCard 
                                                    key={pair.id} 
                                                    pair={pair} 
                                                    onDelete={handleTriageDelete} 
                                                    isDeleting={isDeletingId === pair.id} 
                                                />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                </div>
            ) : (
                // --- Manual View (Unchanged) ---
                <div className="row" style={{ minHeight: '400px', filter: isLoadingSuggestions ? 'blur(2px)' : 'none' }}>
                    {/* Column 1: Job Requirements */}
                    <div className="col-4">
                        <h6 className="border-bottom pb-2">Job Requirements</h6>
                        <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                            {job.features.map(req => {
                                const isSelected = selectedReqId === req.id;
                                const isPaired = selectedContextId && pairedReqIds.has(req.id);
                                const isSuggested = selectedContextId && suggestedReqIds.has(req.id);
                                const isDisabled = selectedContextId && isPaired && !isSelected;
                                let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                                if (isSelected) itemClass += ' active';
                                else if (isPaired && !isDisabled) itemClass += ' list-group-item-success';
                                else if (isSuggested) itemClass += ' list-group-item-info';
                                if (isDisabled) itemClass += ' disabled';

                                return (
                                    <button 
                                        key={req.id} 
                                        type="button" 
                                        className={itemClass} 
                                        onClick={() => handleManualSelectReq(req.id)}
                                        disabled={isDisabled}
                                    >
                                        <span className="text-start">{req.description}</span>
                                        {isSuggested && <span className="badge bg-info-subtle text-info-emphasis rounded-pill">💡</span>}
                                        {isPaired && !isSelected && <span className="badge bg-secondary ms-2">Paired</span>}
                                    </button>
                                );
                            })}
                        </div>
                    </div>

                    {/* Column 2: CV Evidence */}
                    <div className="col-4">
                        <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                        <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                            {cvEvidenceList.map(item => {
                                const isSelected = selectedContextId === item.id;
                                const isPaired = selectedReqId && pairedCvItemIds.has(item.id);
                                const isSuggested = selectedReqId && suggestedCvItemIds.has(item.id);
                                const isDisabled = selectedReqId && isPaired && !isSelected;
                                let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                                if (isSelected) itemClass += ' active';
                                else if (isPaired && !isDisabled) itemClass += ' list-group-item-success';
                                else if (isSuggested) itemClass += ' list-group-item-info';
                                if (isDisabled) itemClass += ' disabled';
                                
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
                                            {isSuggested && <span className="badge bg-info-subtle text-info-emphasis rounded-pill ms-2">💡</span>}
                                            {isPaired && !isSelected && <span className="badge bg-secondary ms-2">Paired</span>}
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
                        <h6 className="border-bottom pb-2">Mapped Pairs ({mapping.pairs.length})</h6>
                        <div style={{ maxHeight: '250px', overflowY: 'auto' }}>
                            {mapping.pairs.length === 0 && (<p className="small text-muted fst-italic">No pairs mapped yet.</p>)}
                            {mapping.pairs.map(pair => (
                                <div key={pair.id} className="card card-body p-3 mb-2 shadow-sm">
                                    <p className="small mb-1"><strong>Req:</strong> {pair.feature_text}</p>
                                    <p className="small mb-2"><strong>Maps to: E </strong> {pair.context_item_text}</p>
                                    {pair.annotation && (<p className="small fst-italic border-top pt-2 mt-2 mb-2"><strong>Note:</strong> {pair.annotation}</p>)}
                                    <button className="btn btn-danger btn-sm" onClick={() => handleManualDelete(pair.id)}>
                                        Ungroup
                                    </button>
                                </div>
                            ))}
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

            {/* --- (Manual View) Modal --- */}
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

// (FIX: Renamed export to match filename)
export default Step1_TriageView;