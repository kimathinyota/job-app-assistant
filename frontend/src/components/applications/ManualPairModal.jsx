// frontend/src/components/applications/ManualPairModal.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { 
    X, 
    Link2, 
    Wand2, 
    Check, 
    MousePointerClick, 
    Briefcase, 
    Layers,
    Info
} from 'lucide-react';
import { addMappingPair } from '../../api/applicationClient';

const ManualPairModal = ({
    isOpen,
    onClose,
    job,
    cv,
    mapping,
    allSuggestions, 
    onPairCreated
}) => {
    // --- State ---
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    // --- Logic & Memos (Preserved & Safety Checked) ---

    // 1. CV Items already paired with Selected Req
    const pairedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set();
        return new Set(
            (mapping?.pairs || [])
                .filter(p => p.feature_id === selectedReqId)
                .map(p => p.context_item_id)
        );
    }, [mapping, selectedReqId]);

    // 2. CV Items suggested for Selected Req
    const suggestedCvItemIds = useMemo(() => {
        if (!selectedReqId) return new Set(); 
        return new Set(
            (allSuggestions || [])
                .filter(s => s.feature_id === selectedReqId)
                .map(s => s.context_item_id)
        );
    }, [allSuggestions, selectedReqId]);

    // 3. Reqs already paired with Selected Evidence
    const pairedReqIds = useMemo(() => {
        if (!selectedContextId) return new Set();
        return new Set(
            (mapping?.pairs || [])
                .filter(p => p.context_item_id === selectedContextId)
                .map(p => p.feature_id)
        );
    }, [mapping, selectedContextId]);

    // 4. Reqs suggested for Selected Evidence
    const suggestedReqIds = useMemo(() => {
        if (!selectedContextId) return new Set();
        return new Set(
            (allSuggestions || [])
                .filter(s => s.context_item_id === selectedContextId)
                .map(s => s.feature_id)
        );
    }, [allSuggestions, selectedContextId]);

    // 5. Flattened CV List
    const cvEvidenceList = useMemo(() => [
        ...(cv.experiences || []).map(item => ({ id: item.id, type: 'experiences', text: `${item.title} @ ${item.company}` })),
        ...(cv.projects || []).map(item => ({ id: item.id, type: 'projects', text: `${item.title} (Project)` })),
        ...(cv.education || []).map(item => ({ id: item.id, type: 'education', text: `${item.degree} @ ${item.institution}` })),
        ...(cv.hobbies || []).map(item => ({ id: item.id, type: 'hobbies', text: `${item.name} (Hobby)` })),
    ], [cv]);

    // --- Auto-fill Annotation ---
    useEffect(() => {
        if (selectedReqId && selectedContextId) {
            const match = (allSuggestions || []).find(
                s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId
            );
            setAnnotation(match?.annotation || "");
        } else {
            setAnnotation("");
        }
    }, [selectedReqId, selectedContextId, allSuggestions]);

    // --- Handlers ---
    const handleSelectReq = (reqId) => {
        setSelectedReqId(prev => (prev === reqId ? null : reqId));
        // Optional: Clear evidence selection if you want strict step-by-step
        // setSelectedContextId(null); 
        // setSelectedContextType(null);
    };

    const handleSelectContextItem = (item) => {
        if (selectedContextId === item.id) {
            setSelectedContextId(null);
            setSelectedContextType(null);
        } else {
            setSelectedContextId(item.id);
            setSelectedContextType(item.type);
        }
    };
    
    const handleClearSelection = () => {
        setSelectedReqId(null);
        setSelectedContextId(null);
        setSelectedContextType(null);
        setAnnotation("");
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedReqId || !selectedContextId) return;

        setIsSubmitting(true);
        try {
            await addMappingPair(
                mapping.id,
                selectedReqId,
                selectedContextId,
                selectedContextType,
                annotation || "Manually created pair"
            );
            await onPairCreated();
            onClose();
        } catch (err) {
            alert(`Failed to create pair: ${err.response?.data?.detail || err.message}`);
        } finally {
            setIsSubmitting(false);
        }
    };

    const isAlreadyPaired = (mapping?.pairs || []).some(
        p => p.feature_id === selectedReqId && p.context_item_id === selectedContextId
    );

    if (!isOpen) return null;

    return (
        <div className="modal show d-block" style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }} tabIndex="-1">
            <style>{`
                .custom-scroll::-webkit-scrollbar { width: 6px; }
                .custom-scroll::-webkit-scrollbar-track { background: rgba(0,0,0,0.02); }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 10px; }
                .modal-xl-custom { max-width: 1140px; }
            `}</style>

            <div className="modal-dialog modal-xl modal-xl-custom modal-dialog-centered">
                <div className="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                    
                    {/* Header */}
                    <div className="modal-header border-bottom-0 bg-white pb-0">
                        <div className="d-flex align-items-center gap-3">
                            <div className="bg-primary bg-opacity-10 p-2 rounded-circle text-primary">
                                <Link2 size={20} />
                            </div>
                            <div>
                                <h5 className="modal-title fw-bold text-dark mb-0">Create Manual Connection</h5>
                                <p className="text-muted small mb-0">Link a requirement to evidence from your CV.</p>
                            </div>
                        </div>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>

                    <div className="modal-body p-4">
                        <div className="row g-4">
                            
                            {/* COL 1: Requirements */}
                            <div className="col-md-4">
                                <div className="card shadow-sm h-100 border-0 bg-light bg-opacity-50">
                                    <div className="card-header bg-white border-bottom fw-bold small text-primary d-flex align-items-center justify-content-between">
                                        <span>1. Select Requirement</span>
                                        <Briefcase size={14} className="opacity-50"/>
                                    </div>
                                    <div className="list-group list-group-flush overflow-auto custom-scroll" style={{height: '450px'}}>
                                        {(job.features || []).map(req => {
                                            const isSelected = selectedReqId === req.id;
                                            const isPaired = selectedContextId && pairedReqIds.has(req.id);
                                            const isSuggested = selectedContextId && suggestedReqIds.has(req.id) && !isPaired;
                                            const isDisabled = selectedContextId && isPaired && !isSelected;

                                            let itemClass = 'bg-white text-dark border-bottom-0 mb-1 rounded-3 border shadow-sm mx-2 mt-2';
                                            if (isSelected) itemClass = 'bg-primary text-white border-primary';
                                            else if (isPaired) itemClass = 'bg-light text-muted fst-italic opacity-75';
                                            else if (isSuggested) itemClass = 'bg-success-subtle text-success-emphasis border-success-subtle'; 

                                            return (
                                                <button 
                                                    key={req.id}
                                                    type="button"
                                                    className={`list-group-item list-group-item-action p-3 small transition-all ${itemClass} ${isDisabled ? 'opacity-50' : ''}`}
                                                    onClick={() => handleSelectReq(req.id)}
                                                    disabled={isDisabled}
                                                >
                                                    <div className="d-flex justify-content-between align-items-start gap-2">
                                                        <span className="text-start lh-sm">{req.description}</span>
                                                        <div className="d-flex flex-column gap-1 mt-1">
                                                            {isPaired && <Link2 size={14}/>}
                                                            {isSuggested && <Wand2 size={14} className="animate-pulse"/>}
                                                            {isSelected && <Check size={14}/>}
                                                        </div>
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>

                            {/* COL 2: Evidence */}
                            <div className="col-md-4">
                                <div className="card shadow-sm h-100 border-0 bg-light bg-opacity-50">
                                    <div className="card-header bg-white border-bottom fw-bold small text-success d-flex align-items-center justify-content-between">
                                        <span>2. Select Evidence</span>
                                        <Layers size={14} className="opacity-50"/>
                                    </div>
                                    <div className="list-group list-group-flush overflow-auto custom-scroll" style={{height: '450px'}}>
                                        {cvEvidenceList.map(item => {
                                            const isSelected = selectedContextId === item.id;
                                            const isPaired = selectedReqId && pairedCvItemIds.has(item.id);
                                            const isSuggested = selectedReqId && suggestedCvItemIds.has(item.id) && !isPaired;
                                            const isDisabled = selectedReqId && isPaired && !isSelected;

                                            let itemClass = 'bg-white text-dark border-bottom-0 mb-1 rounded-3 border shadow-sm mx-2 mt-2';
                                            if (isSelected) itemClass = 'bg-success text-white border-success';
                                            else if (isPaired) itemClass = 'bg-light text-muted fst-italic opacity-75';
                                            else if (isSuggested) itemClass = 'bg-primary-subtle text-primary-emphasis border-primary-subtle';

                                            return (
                                                <button 
                                                    key={item.id}
                                                    type="button"
                                                    className={`list-group-item list-group-item-action p-3 small transition-all ${itemClass} ${isDisabled ? 'opacity-50' : ''}`}
                                                    onClick={() => handleSelectContextItem(item)}
                                                    disabled={isDisabled}
                                                >
                                                    <div className="d-flex justify-content-between align-items-center">
                                                        <div className="d-flex flex-column align-items-start text-start overflow-hidden">
                                                            <span className="text-truncate w-100 fw-medium">{item.text}</span>
                                                            <span className="badge bg-white bg-opacity-25 border border-white border-opacity-25 fw-normal mt-1" style={{fontSize: '0.65rem'}}>
                                                                {item.type.toUpperCase()}
                                                            </span>
                                                        </div>
                                                        <div className="d-flex gap-2 align-items-center ms-2">
                                                            {isPaired && <Link2 size={14}/>}
                                                            {isSuggested && <Wand2 size={14} className="animate-pulse"/>}
                                                            {isSelected && <Check size={14}/>}
                                                        </div>
                                                    </div>
                                                </button>
                                            );
                                        })}
                                    </div>
                                </div>
                            </div>

                            {/* COL 3: Action Panel */}
                            <div className="col-md-4">
                                <div className="card shadow-sm h-100 border rounded-4 bg-white">
                                    <div className="card-header bg-white border-bottom fw-bold small p-3 text-dark d-flex align-items-center gap-2">
                                        <span className="badge bg-dark rounded-circle text-white d-flex align-items-center justify-content-center" style={{width:20, height:20}}>3</span>
                                        Review & Connect
                                    </div>
                                    <div className="card-body d-flex flex-column p-4">
                                        
                                        {/* Info Box */}
                                        <div className="alert alert-light border mb-3 d-flex align-items-start gap-2 p-2">
                                            <Info size={16} className="text-primary mt-1 flex-shrink-0"/>
                                            <p className="small text-muted mb-0" style={{lineHeight: '1.3'}}>
                                                Select one item from each column to create a connection.
                                            </p>
                                        </div>

                                        <label className="form-label small fw-bold text-muted text-uppercase mb-2">Reasoning</label>
                                        <textarea 
                                            className="form-control bg-light border-0 shadow-sm text-dark mb-3 flex-grow-1" 
                                            placeholder={selectedReqId && selectedContextId ? "Why is this a match? (Optional)" : "Waiting for selection..."}
                                            value={annotation}
                                            onChange={(e) => setAnnotation(e.target.value)}
                                            disabled={!selectedReqId || !selectedContextId}
                                            style={{resize: 'none', minHeight: '120px'}}
                                        />
                                        
                                        {(selectedReqId || selectedContextId) && (
                                            <button 
                                                type="button"
                                                className="btn btn-sm btn-outline-secondary w-100 mb-3" 
                                                onClick={handleClearSelection}
                                            >
                                                Clear Selection
                                            </button>
                                        )}

                                        <button 
                                            type="button"
                                            className={`btn w-100 py-3 fw-bold shadow-sm ${isAlreadyPaired ? 'btn-secondary' : 'btn-primary'}`}
                                            disabled={!selectedReqId || !selectedContextId || isSubmitting || isAlreadyPaired}
                                            onClick={handleSubmit}
                                        >
                                            {isSubmitting ? 'Connecting...' : 
                                             isAlreadyPaired ? 'Already Connected' : 
                                             <><MousePointerClick size={18} className="me-2"/> Create Link</>}
                                        </button>
                                    </div>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ManualPairModal;