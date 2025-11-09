// frontend/src/components/applications/ManualPairModal.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { addMappingPair } from '../../api/applicationClient';

/**
 * This is a "power user" modal for manually creating a pair.
 * It re-creates the 2-column selection UI, complete with highlighting
 * for suggested (💡) and paired (✅) items.
 */
const ManualPairModal = ({
    isOpen,
    onClose,
    job,
    cv,
    mapping,
    allSuggestions, // Pass in the suggestions from the main page
    onPairCreated // Callback to trigger a data reload
}) => {
    // --- Internal State for this modal ---
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    // --- Memos for Highlighting (with safety checks and FIX) ---
    
    // 1. Get all CV items already paired with the *selected Req*
    const pairedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set();
        const pairs = mapping?.pairs || []; 
        return new Set(
            pairs
                .filter(p => p.feature_id === selectedReqId)
                .map(p => p.context_item_id)
        );
    }, [mapping, selectedReqId]);

    // 2. Get all CV items suggested for the *selected Req*
    const suggestedCvItemIds = useMemo(() => {
        // --- THIS IS THE FIX ---
        // Was: if (!selectedReqId || selectedContextId)
        if (!selectedReqId) return new Set(); 
        // --- END OF FIX ---
        
        const suggestions = allSuggestions || []; 
        return new Set(
            suggestions
                .filter(s => s.feature_id === selectedReqId)
                .map(s => s.context_item_id)
        );
    }, [allSuggestions, selectedReqId]); // <-- Removed selectedContextId from dependencies

    // 3. Get all Reqs already paired with the *selected CV item*
    const pairedReqIds = useMemo(() => {
        if (!selectedContextId) return new Set();
        const pairs = mapping?.pairs || [];
        return new Set(
            pairs
                .filter(p => p.context_item_id === selectedContextId)
                .map(p => p.feature_id)
        );
    }, [mapping, selectedContextId]);

    // 4. Get all Reqs suggested for the *selected CV item*
    const suggestedReqIds = useMemo(() => {
        // --- THIS IS THE FIX ---
        // Was: if (!selectedContextId || selectedReqId)
        if (!selectedContextId) return new Set();
        // --- END OF FIX ---
        
        const suggestions = allSuggestions || [];
        return new Set(
            suggestions
                .filter(s => s.context_item_id === selectedContextId)
                .map(s => s.feature_id)
        );
    }, [allSuggestions, selectedContextId]); // <-- Removed selectedReqId from dependencies

    // 5. Memo for the CV Evidence list
    const cvEvidenceList = useMemo(() => [
        ...cv.experiences.map(item => ({
            id: item.id, type: 'experiences', text: `${item.title} @ ${item.company}`
        })),
        ...cv.projects.map(item => ({
            id: item.id, type: 'projects', text: `${item.title} (Project)`
        })),
        ...cv.education.map(item => ({
            id: item.id, type: 'education', text: `${item.degree} @ ${item.institution}`
        })),
        ...cv.hobbies.map(item => ({
            id: item.id, type: 'hobbies', text: `${item.name} (Hobby)`
        })),
    ], [cv]);

    // --- Effect to pre-fill annotation ---
    useEffect(() => {
        if (selectedReqId && selectedContextId) {
            const suggestions = allSuggestions || [];
            const matchingSuggestion = suggestions.find(
                s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId
            );
            setAnnotation(matchingSuggestion?.annotation || "");
        } else {
            setAnnotation("");
        }
    }, [selectedReqId, selectedContextId, allSuggestions]);

    // --- Click Handlers (Using Toggles) ---
    // This logic is correct for this UI: clicking an item selects it,
    // clicking it again deselects it (allowing the *other* column to highlight).
    const handleSelectReq = (reqId) => {
        setSelectedReqId(prev => (prev === reqId ? null : reqId));
        setSelectedContextId(null); // Clear other column on new selection
        setSelectedContextType(null);
    };

    const handleSelectContextItem = (item) => {
        setSelectedReqId(null); // Clear other column on new selection
        if (selectedContextId === item.id) {
            setSelectedContextId(null);
            setSelectedContextType(null);
        } else {
            setSelectedContextId(item.id);
            setSelectedContextType(item.type);
        }
    };
    
    // --- Clear selection button ---
    const handleClearSelection = () => {
        setSelectedReqId(null);
        setSelectedContextId(null);
        setSelectedContextType(null);
        setAnnotation("");
    };


    // --- Submit Handler ---
    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedReqId || !selectedContextId) {
            alert('Please select one item from each list.');
            return;
        }

        setIsSubmitting(true);
        try {
            await addMappingPair(
                mapping.id,
                selectedReqId,
                selectedContextId,
                selectedContextType,
                annotation || `Manually created pair`
            );
            await onPairCreated();
            onClose();

        } catch (err) {
            alert(`Failed to create pair: ${err.response?.data?.detail || err.message}`);
        } finally {
            setIsSubmitting(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div
            className="modal"
            style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}
            onClick={onClose}
        >
            <div
                className="modal-dialog modal-xl modal-dialog-centered"
                onClick={(e) => e.stopPropagation()}
            >
                <form className="modal-content" onSubmit={handleSubmit}>
                    <div className="modal-header">
                        <h5 className="modal-title">Create Manual Pair</h5>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>
                    <div className="modal-body">
                        <p className="small text-muted">
                            Select a requirement (left) or CV item (right) to see pairings and suggestions.
                        </p>
                        
                        {(selectedReqId || selectedContextId) && (
                            <button
                                type="button"
                                className="btn btn-sm btn-outline-secondary w-100 mb-2"
                                onClick={handleClearSelection}
                            >
                                Clear Selection
                            </button>
                        )}
                        
                        <div className="row" style={{ minHeight: '300px' }}>
                            {/* --- Column 1: Job Requirements --- */}
                            <div className="col-md-6">
                                <h6 className="border-bottom pb-2">Job Requirements</h6>
                                <div className="list-group" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                    {job.features.map(req => {
                                        const isSelected = selectedReqId === req.id;
                                        // Highlight if paired with the *selected CV item*
                                        const isPaired = selectedContextId && pairedReqIds.has(req.id);
                                        // Highlight if suggested for the *selected CV item*
                                        const isSuggested = selectedContextId && suggestedReqIds.has(req.id);

                                        let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                                        if (isSelected) itemClass += ' active';
                                        else if (isPaired) itemClass += ' list-group-item-success';
                                        else if (isSuggested) itemClass += ' list-group-item-info';

                                        return (
                                            <button
                                                key={req.id}
                                                type="button"
                                                className={itemClass}
                                                onClick={() => handleSelectReq(req.id)}
                                            >
                                                {req.description}
                                                {isSuggested && <span>💡</span>}
                                                {isPaired && !isSelected && <span>✅</span>}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            {/* --- Column 2: CV Evidence --- */}
                            <div className="col-md-6">
                                <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                                <div className="list-group" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                                    {cvEvidenceList.map(item => {
                                        const isSelected = selectedContextId === item.id;
                                        // Highlight if paired with the *selected Req*
                                        const isPaired = selectedReqId && pairedCvItemIds.has(item.id);
                                        // Highlight if suggested for the *selected Req*
                                        const isSuggested = selectedReqId && suggestedCvItemIds.has(item.id);

                                        let itemClass = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                                        if (isSelected) itemClass += ' active';
                                        else if (isPaired) itemClass += ' list-group-item-success';
                                        else if (isSuggested) itemClass += ' list-group-item-info';

                                        return (
                                            <button
                                                key={item.id}
                                                type="button"
                                                className={itemClass}
                                                onClick={() => handleSelectContextItem(item)}
                                            >
                                                {item.text}
                                                {isSuggested && <span>💡</span>}
                                                {isPaired && !isSelected && <span>✅</span>}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>
                        </div>

                        {/* --- Annotation Section --- */}
                        <div className="mt-3">
                            <label htmlFor="manual-annotation" className="form-label fw-medium">
                                Annotation (Reason):
                            </label>
                            <textarea
                                id="manual-annotation"
                                className="form-control"
                                rows="2"
                                value={annotation}
                                onChange={(e) => setAnnotation(e.target.value)}
                                placeholder="Auto-filled for suggestions, or add your own note..."
                                disabled={!selectedReqId || !selectedContextId}
                            />
                        </div>
                    </div>
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="btn btn-primary"
                            disabled={isSubmitting || !selectedReqId || !selectedContextId}
                        >
                            {isSubmitting ? 'Creating...' : 'Create This Pair'}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ManualPairModal;