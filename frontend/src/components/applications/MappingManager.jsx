// frontend/src/components/applications/MappingManager.jsx
import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    Wand2, Check, X, Trash2, Eye, Edit3, ArrowRight, ArrowLeft,
    Briefcase, GraduationCap, FolderGit2, Heart, Layers, 
    Lightbulb, Link2, MousePointerClick, MessageSquare, 
    Search, GitMerge, Settings2, AlertCircle, ArrowUpDown, Filter,
    ChevronDown
} from 'lucide-react';
import {
    fetchApplicationDetails, fetchJobDetails, fetchMappingDetails,
    addMappingPair, deleteMappingPair, inferMappingPairs
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient';
import CVItemPreviewModal from './CVItemPreviewModal';

// --- CONSTANTS ---
const TUNING_MODES = {
    "super_eager": "Super Eager (Find all)",
    "eager_mode": "Eager (Find many)",
    "balanced_default": "Balanced (Recommended)",
    "picky_mode": "Picky (Fewer, better)",
    "super_picky": "Super Picky (Best only)"
};

const TRIAGE_GROUPS = [
    { id: 'experiences', title: 'Experience', icon: Briefcase },
    { id: 'projects', title: 'Projects', icon: FolderGit2 },
    { id: 'education', title: 'Education', icon: GraduationCap },
    { id: 'hobbies', title: 'Hobbies', icon: Heart }
];

// --- SUB-COMPONENTS ---

const LoadingOverlay = ({ mode }) => (
    <div className="position-absolute w-100 h-100 d-flex flex-column justify-content-center align-items-center bg-white bg-opacity-90"
        style={{ zIndex: 50, top: 0, left: 0, backdropFilter: 'blur(4px)' }}>
        <div className="spinner-border text-primary mb-3" role="status" style={{width: '3rem', height: '3rem'}}></div>
        <h6 className="fw-bold text-dark d-flex align-items-center gap-2">
            <Wand2 className="text-primary animate-pulse" size={24}/> AI Analysis Running...
        </h6>
        <p className="text-muted small mb-0">Using strategy: <span className="fw-medium text-dark">"{mode.replace(/_/g, ' ')}"</span></p>
    </div>
);

const SuggestionItem = ({ suggestion, onAccept, onIgnore, isProcessing }) => {
    const [note, setNote] = useState(suggestion.annotation || "");
    const [isEditing, setIsEditing] = useState(false);

    return (
        <div className="p-4 border-bottom hover-bg-white transition-all">
            <div className="mb-3">
                <span className="badge bg-primary bg-opacity-10 text-primary border border-primary-subtle mb-2 text-uppercase" style={{fontSize: '0.65rem'}}>Match Found</span>
                <p className="mb-0 fw-medium text-dark lh-sm text-wrap">{suggestion.feature_text}</p>
            </div>
            
            <div className="bg-white border rounded-3 p-3 mb-3 shadow-sm">
                <div className="d-flex justify-content-between align-items-center mb-2">
                    <small className="fw-bold text-muted d-flex align-items-center gap-1 text-uppercase" style={{fontSize: '0.7rem'}}>
                        <Lightbulb size={12} className="text-warning"/> AI Reasoning
                    </small>
                    <button className="btn btn-link p-0 text-decoration-none small" style={{fontSize:'0.75rem'}} onClick={() => setIsEditing(!isEditing)}>
                        {isEditing ? 'Done' : 'Edit'}
                    </button>
                </div>
                {isEditing ? (
                    <textarea className="form-control form-control-sm bg-light border-0" rows={2} value={note} onChange={(e) => setNote(e.target.value)} autoFocus/>
                ) : (
                    <p className="mb-0 small text-muted fst-italic">{note}</p>
                )}
            </div>

            <div className="d-flex justify-content-end gap-2">
                <button className="btn btn-sm btn-white border shadow-sm text-muted px-3 hover-bg-light" onClick={() => onIgnore(suggestion.id)}>
                    <X size={14} className="me-1"/> Ignore
                </button>
                <button className="btn btn-sm btn-primary shadow-sm px-3" onClick={() => onAccept(suggestion, note)} disabled={isProcessing}>
                    {isProcessing ? <span className="spinner-border spinner-border-sm"/> : <><Check size={14} className="me-1"/> Accept Match</>}
                </button>
            </div>
        </div>
    );
};

const MappedItem = ({ pair, onDelete, isDeleting }) => (
    <div className="list-group-item border-0 border-bottom p-3 d-flex justify-content-between align-items-start hover-bg-light transition-all">
        <div className="d-flex gap-3 min-w-0">
            <div className="mt-1 text-success flex-shrink-0"><Check size={16}/></div>
            <div className="min-w-0">
                <p className="mb-1 small fw-bold text-dark text-wrap lh-sm">{pair.feature_text}</p>
                {pair.annotation && <p className="mb-0 small text-muted fst-italic text-wrap">{pair.annotation}</p>}
            </div>
        </div>
        <button 
            className="btn btn-link text-danger p-1 opacity-50 hover-opacity-100 flex-shrink-0 ms-2 rounded-circle hover-bg-danger-subtle" 
            onClick={() => onDelete(pair.id)} 
            disabled={isDeleting}
        >
            <Trash2 size={14}/>
        </button>
    </div>
);

const TriageModeView = ({ 
    groupedSuggestions, 
    groupedPairs, 
    activeCategory, 
    setActiveCategory, 
    onAccept, 
    onIgnore, 
    onDelete, 
    onPreview,
    isProcessingId,
    isDeletingId
}) => {
    
    const renderTriageButton = (group, isDropdownItem = false) => {
        const count = groupedSuggestions[group.id]?.length || 0;
        const isActive = activeCategory === group.id;

        if (isDropdownItem) {
            return (
                <button 
                    key={group.id}
                    onClick={() => setActiveCategory(group.id)}
                    className={`dropdown-item d-flex justify-content-between align-items-center ${isActive ? 'active' : ''}`}
                >
                    <span className="d-flex align-items-center gap-2">
                        <group.icon size={16} className={isActive ? '' : 'opacity-50'}/>
                        {group.title}
                    </span>
                    {count > 0 && (
                        <span className="badge bg-danger rounded-pill ms-1 shadow-sm">{count}</span>
                    )}
                </button>
            );
        }

        return (
            <button 
                key={group.id}
                onClick={() => setActiveCategory(group.id)}
                className={`btn btn-sm rounded-pill px-4 d-flex align-items-center gap-2 border-0 transition-all ${isActive ? 'bg-white text-primary shadow-sm fw-bold' : 'text-muted hover-text-dark'}`}
                style={{minWidth: '140px', justifyContent: 'center'}}
            >
                <group.icon size={16} className={isActive ? 'text-primary' : 'opacity-50'}/>
                {group.title}
                {count > 0 && (
                    <span className="badge bg-danger rounded-pill ms-1 shadow-sm" style={{fontSize: '0.65rem', padding: '0.35em 0.6em'}}>{count}</span>
                )}
            </button>
        );
    };

    const activeGroup = TRIAGE_GROUPS.find(g => g.id === activeCategory) || TRIAGE_GROUPS[0];
    const activeCount = groupedSuggestions[activeGroup.id]?.length || 0;

    return (
        <div className="animate-fade-in">
            {/* Category Tabs */}
            <div className="d-flex justify-content-center mb-5">
                {/* Desktop */}
                <div className="bg-light p-1 rounded-pill border d-none d-lg-inline-flex shadow-sm flex-nowrap">
                    {TRIAGE_GROUPS.map(group => renderTriageButton(group, false))}
                </div>

                {/* Mobile */}
                <div className="dropdown d-lg-none w-100">
                    <button 
                        className="btn btn-white border shadow-sm d-flex w-100 justify-content-between align-items-center py-2 px-3 rounded-3" 
                        type="button" 
                        data-bs-toggle="dropdown" 
                        aria-expanded="false"
                    >
                        <span className="d-flex align-items-center gap-2 fw-bold text-primary">
                            <activeGroup.icon size={18}/>
                            {activeGroup.title}
                            {activeCount > 0 && (
                                <span className="badge bg-danger rounded-pill ms-1 shadow-sm">{activeCount}</span>
                            )}
                        </span>
                        <ChevronDown size={18} className="text-muted"/>
                    </button>
                    <ul className="dropdown-menu shadow-lg border-0 w-100 p-2 rounded-3">
                        {TRIAGE_GROUPS.map(group => (
                            <li key={group.id}>{renderTriageButton(group, true)}</li>
                        ))}
                    </ul>
                </div>
            </div>

            <div className="row g-4">
                {/* LEFT: Suggestions */}
                <div className="col-lg-6">
                    <div className="d-flex align-items-center justify-content-between mb-3 px-1">
                        <h6 className="text-uppercase text-muted small fw-bold mb-0 d-flex align-items-center gap-2">
                            <div className="p-1 bg-primary bg-opacity-10 rounded text-primary"><Wand2 size={14}/></div>
                            AI Suggestions
                        </h6>
                        <span className="badge bg-light text-muted border">{(groupedSuggestions[activeCategory] || []).reduce((acc, g) => acc + g.items.length, 0)} Pending</span>
                    </div>
                    
                    <div className="custom-scroll pe-2" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                        {(groupedSuggestions[activeCategory] || []).length === 0 ? (
                            <div className="text-center py-5 bg-light bg-opacity-50 rounded-4 border border-dashed">
                                <div className="text-muted opacity-25 mb-3"><Check size={48}/></div>
                                <h6 className="fw-bold text-muted">All Caught Up!</h6>
                                <p className="small text-muted mb-0">No pending suggestions for this category.</p>
                            </div>
                        ) : (
                            groupedSuggestions[activeCategory].map(group => (
                                <div key={group.cvItemId} className="card border-0 shadow-sm mb-4 overflow-hidden rounded-4">
                                    <div className="card-header bg-white border-bottom p-3 d-flex justify-content-between align-items-center">
                                        <span className="fw-bold text-dark small d-flex align-items-center gap-2 text-truncate min-w-0">
                                            <Layers size={16} className="text-primary flex-shrink-0"/> 
                                            <span className="text-truncate" title={group.cvItemText}>{group.cvItemText}</span>
                                        </span>
                                        <button className="btn btn-sm btn-light border hover-bg-light flex-shrink-0 ms-2 rounded-circle p-1" onClick={() => onPreview(group.cvItem, group.cvItemType)}>
                                            <Eye size={14}/>
                                        </button>
                                    </div>
                                    <div className="card-body p-0 bg-light bg-opacity-10">
                                        {group.items.map(sugg => (
                                            <SuggestionItem 
                                                key={sugg.id} 
                                                suggestion={sugg} 
                                                onAccept={onAccept} 
                                                onIgnore={onIgnore}
                                                isProcessing={isProcessingId === sugg.id}
                                            />
                                        ))}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>

                {/* RIGHT: Confirmed */}
                <div className="col-lg-6">
                    <div className="d-flex align-items-center justify-content-between mb-3 px-1">
                        <h6 className="text-uppercase text-muted small fw-bold mb-0 d-flex align-items-center gap-2">
                            <div className="p-1 bg-success bg-opacity-10 rounded text-success"><Check size={14}/></div>
                            Confirmed Maps
                        </h6>
                        <span className="badge bg-light text-muted border">{(groupedPairs[activeCategory] || []).reduce((acc, g) => acc + g.items.length, 0)} Active</span>
                    </div>

                    <div className="custom-scroll pe-2" style={{ maxHeight: '600px', overflowY: 'auto' }}>
                        {(groupedPairs[activeCategory] || []).length === 0 ? (
                            <div className="text-center py-5 bg-light bg-opacity-50 rounded-4 border border-dashed">
                                <p className="small text-muted mb-0">No mappings confirmed yet.</p>
                            </div>
                        ) : (
                            groupedPairs[activeCategory].map(group => (
                                <div key={group.cvItemId} className="card border shadow-sm mb-3 overflow-hidden rounded-3">
                                    <div className="card-header bg-white p-2 px-3 border-bottom fw-bold small text-muted d-flex justify-content-between align-items-center">
                                        <span className="text-truncate">{group.cvItemText}</span>
                                        <button className="btn btn-sm p-0 text-muted ms-2" onClick={() => onPreview(group.cvItem, group.cvItemType)}>
                                            <Eye size={12}/>
                                        </button>
                                    </div>
                                    <div className="list-group list-group-flush">
                                        {group.items.map(pair => (
                                            <MappedItem 
                                                key={pair.id} 
                                                pair={pair} 
                                                onDelete={onDelete}
                                                isDeleting={isDeletingId === pair.id}
                                            />
                                        ))}
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

const ManualModeView = ({ 
    job, 
    cvEvidenceList, 
    mapping,
    allSuggestions,
    cvItemLookups, 
    onAddPair, 
    onDeletePair, 
    onPreview,
    isSubmitting 
}) => {
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    
    const [pairSearch, setPairSearch] = useState('');
    const [filter, setFilter] = useState('all');
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

    const [reqSearch, setReqSearch] = useState('');
    const [evidenceSearch, setEvidenceSearch] = useState('');
    const [showReqSearch, setShowReqSearch] = useState(false);
    const [showEvidenceSearch, setShowEvidenceSearch] = useState(false);
    const [mobileStep, setMobileStep] = useState(1);

    const reqSearchInputRef = useRef(null);
    const evidenceSearchInputRef = useRef(null);

    useEffect(() => {
        if (showReqSearch && reqSearchInputRef.current) reqSearchInputRef.current.focus();
    }, [showReqSearch]);

    useEffect(() => {
        if (showEvidenceSearch && evidenceSearchInputRef.current) evidenceSearchInputRef.current.focus();
    }, [showEvidenceSearch]);

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

    const selectedReqText = useMemo(() => {
        return job.features.find(f => f.id === selectedReqId)?.description;
    }, [selectedReqId, job.features]);

    const selectedEvidenceItem = useMemo(() => {
        const item = cvEvidenceList.find(e => e.id === selectedContextId);
        if (!item) return null;
        const group = TRIAGE_GROUPS.find(g => g.id === item.type);
        return { ...item, icon: group ? group.icon : Layers };
    }, [selectedContextId, cvEvidenceList]);

    useEffect(() => {
        if (selectedReqId && selectedContextId) {
            const match = allSuggestions.find(s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId);
            setAnnotation(match?.annotation || "");
        } else {
            setAnnotation("");
        }
    }, [selectedReqId, selectedContextId, allSuggestions]);

    const handleManualClear = useCallback(() => {
        setSelectedReqId(null);
        setSelectedContextId(null);
        setSelectedContextType(null);
        setAnnotation("");
        setMobileStep(1);
    }, []);

    useEffect(() => {
        const mql = window.matchMedia("(max-width: 991.98px)");
        const handleChange = (e) => { if (e.matches) handleManualClear(); };
        mql.addEventListener('change', handleChange);
        return () => mql.removeEventListener('change', handleChange);
    }, [handleManualClear]);

    const handleCreate = () => {
        if (!selectedReqId || !selectedContextId) return;
        const sugg = allSuggestions.find(s => s.feature_id === selectedReqId && s.context_item_id === selectedContextId);
        onAddPair(selectedReqId, selectedContextId, selectedContextType, annotation, sugg);
        handleManualClear();
    };

    const isAlreadyPaired = mapping.pairs.some(p => p.feature_id === selectedReqId && p.context_item_id === selectedContextId);

    const filteredPairs = useMemo(() => {
        let pairs = [...mapping.pairs];
        pairs = pairs.filter(pair => {
            const lookup = cvItemLookups.get(pair.context_item_id);
            const evidenceText = lookup?.name || pair.context_item_text || '';
            const matchesSearch = ((pair.feature_text || '').toLowerCase().includes(pairSearch.toLowerCase()) ||
                (evidenceText).toLowerCase().includes(pairSearch.toLowerCase()) ||
                (pair.annotation || '').toLowerCase().includes(pairSearch.toLowerCase()));
            const matchesCategory = filter === 'all' || pair.context_item_type === filter;
            return matchesSearch && matchesCategory;
        });

        if (sortConfig.key) {
            pairs.sort((a, b) => {
                let aVal = a[sortConfig.key] || '', bVal = b[sortConfig.key] || '';
                if (sortConfig.key === 'context_item_text') {
                    aVal = cvItemLookups.get(a.context_item_id)?.name || a.context_item_text || '';
                    bVal = cvItemLookups.get(b.context_item_id)?.name || b.context_item_text || '';
                }
                if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
                if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
                return 0;
            });
        }
        return pairs;
    }, [mapping.pairs, pairSearch, filter, sortConfig, cvItemLookups]);

    const requestSort = (key) => {
        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') direction = 'desc';
        setSortConfig({ key, direction });
    };

    const handleTablePreview = (pair) => {
        const lookup = cvItemLookups.get(pair.context_item_id);
        if (lookup && lookup.item) onPreview(lookup.item, pair.context_item_type);
    };

    const filteredFeatures = useMemo(() => (job.features || []).filter(req => req.description.toLowerCase().includes(reqSearch.toLowerCase())), [job.features, reqSearch]);
    const filteredEvidence = useMemo(() => cvEvidenceList.filter(item => item.text.toLowerCase().includes(evidenceSearch.toLowerCase())), [cvEvidenceList, evidenceSearch]);

    const RequirementColumn = ({ isMobile = false, inputRef }) => (
        <div className="card shadow-sm h-100 border-0 rounded-4 overflow-hidden">
            <div className="card-header bg-primary bg-opacity-10 text-primary fw-bold small border-bottom p-3 d-flex justify-content-between align-items-center">
                <span>1. Select Requirement</span>
                <button className={`btn btn-sm p-0 ${showReqSearch ? 'text-primary' : 'text-muted hover-text-primary'}`} onClick={() => setShowReqSearch(!showReqSearch)}>
                    <Search size={14}/>
                </button>
            </div>
            {showReqSearch && (
                <div className="p-2 bg-light border-bottom">
                    <input ref={inputRef} type="text" className="form-control form-control-sm border-0 shadow-none bg-white" placeholder="Filter requirements..." value={reqSearch} onChange={(e) => setReqSearch(e.target.value)}/>
                </div>
            )}
            <div className="list-group list-group-flush overflow-auto custom-scroll" style={{height: '400px'}}>
                {filteredFeatures.map(req => {
                    const isSelected = selectedReqId === req.id;
                    const isPaired = selectedContextId && pairedReqIds.has(req.id);
                    const isSuggested = selectedContextId && suggestedReqIds.has(req.id) && !isPaired;
                    const isDisabled = selectedContextId && isPaired && !isSelected;
                    let itemClass = isSelected ? 'bg-primary text-white border-primary' : isPaired ? 'bg-light text-muted fst-italic' : isSuggested ? 'bg-primary-subtle text-primary-emphasis fw-medium border-primary-subtle' : 'bg-white text-dark';

                    return (
                        <button key={req.id} className={`list-group-item list-group-item-action border-0 border-bottom small p-3 transition-all ${itemClass} ${isDisabled ? 'opacity-50' : ''}`}
                            onClick={() => setSelectedReqId(prev => {
                                const newId = prev === req.id ? null : req.id;
                                if (isMobile && newId) setMobileStep(2);
                                return newId;
                            })}
                            disabled={isDisabled}
                        >
                            <div className="d-flex justify-content-between align-items-start gap-2">
                                <span className="text-start text-wrap lh-sm">{req.description}</span>
                                <div className="d-flex flex-column gap-1 mt-1">{isPaired && <Link2 size={14}/>}{isSuggested && <Wand2 size={14} className="animate-pulse"/>}</div>
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );

    const EvidenceColumn = ({ isMobile = false, inputRef }) => (
        <div className="card shadow-sm h-100 border-0 rounded-4 overflow-hidden">
            <div className="card-header bg-success bg-opacity-10 text-success fw-bold small border-bottom p-3 d-flex justify-content-between align-items-center">
                {isMobile && <button className="btn btn-sm p-0 text-success me-2" onClick={() => setMobileStep(1)}><ArrowLeft size={18} /></button>}
                <span className="flex-grow-1">2. Select Evidence</span>
                <button className={`btn btn-sm p-0 ${showEvidenceSearch ? 'text-success' : 'text-muted hover-text-success'}`} onClick={() => setShowEvidenceSearch(!showEvidenceSearch)}><Search size={14}/></button>
            </div>
            {showEvidenceSearch && (
                <div className="p-2 bg-light border-bottom">
                    <input ref={inputRef} type="text" className="form-control form-control-sm border-0 shadow-none bg-white" placeholder="Filter evidence..." value={evidenceSearch} onChange={(e) => setEvidenceSearch(e.target.value)}/>
                </div>
            )}
            <div className="list-group list-group-flush overflow-auto custom-scroll" style={{height: '400px'}}>
                {filteredEvidence.map(item => {
                    const isSelected = selectedContextId === item.id;
                    const isPaired = selectedReqId && pairedCvItemIds.has(item.id);
                    const isSuggested = selectedReqId && suggestedCvItemIds.has(item.id) && !isPaired;
                    const isDisabled = selectedReqId && isPaired && !isSelected;
                    let itemClass = isSelected ? 'bg-success text-white border-success' : isPaired ? 'bg-light text-muted fst-italic' : isSuggested ? 'bg-primary-subtle text-primary-emphasis fw-medium border-primary-subtle' : 'bg-white text-dark';

                    return (
                        <button key={item.id} className={`list-group-item list-group-item-action border-0 border-bottom small p-3 transition-all ${itemClass} ${isDisabled ? 'opacity-50' : ''}`}
                            onClick={() => {
                                if (isSelected) { setSelectedContextId(null); setSelectedContextType(null); }
                                else { setSelectedContextId(item.id); setSelectedContextType(item.type); if (isMobile) setMobileStep(3); }
                            }}
                            disabled={isDisabled}
                        >
                            <div className="d-flex justify-content-between align-items-center">
                                <div className="d-flex flex-column align-items-start text-start overflow-hidden">
                                    <span className="text-truncate w-100 fw-medium">{item.text}</span>
                                    <span className="badge bg-white bg-opacity-25 border border-white border-opacity-25 fw-normal" style={{fontSize: '0.65rem'}}>{item.type.toUpperCase()}</span>
                                </div>
                                <div className="d-flex gap-2 align-items-center ms-2">
                                    {isPaired && <Link2 size={14}/>}
                                    {isSuggested && <Wand2 size={14} className="animate-pulse"/>}
                                    <div role="button" onClick={(e) => { e.stopPropagation(); onPreview(item.item, item.type); }} className="p-1 rounded hover-bg-white-25"><Eye size={14}/></div>
                                </div>
                            </div>
                        </button>
                    );
                })}
            </div>
        </div>
    );

    const ConnectColumn = ({ isMobile = false, reqText, evidenceItem, onPreview }) => (
        <div className="card shadow-sm h-100 border-0 rounded-4 overflow-hidden">
            <div className="card-header bg-dark bg-opacity-10 text-dark fw-bold small border-bottom p-3 d-flex align-items-center gap-2">
                {isMobile && <button className="btn btn-sm p-0 text-dark me-2" onClick={() => setMobileStep(2)}><ArrowLeft size={18} /></button>}
                <span className="flex-grow-1">3. Connect & Link</span>
            </div>
            <div className="card-body d-flex flex-column p-4 bg-white h-100">
                <div className="mb-3">
                    <label className="form-label small fw-bold text-muted text-uppercase mb-2">Your Selection</label>
                    {(!reqText && !evidenceItem) ? (
                        <div className="text-center py-4 bg-light rounded-3 border-dashed"><p className="small text-muted fst-italic mb-0">Select items to begin</p></div>
                    ) : (
                        <div className="d-flex flex-column gap-2">
                            {reqText && (
                                <div className="d-flex align-items-start p-3 bg-white rounded-3 border border-primary-subtle shadow-sm">
                                    <Layers size={16} className="text-primary me-2 flex-shrink-0 mt-1" />
                                    <span className="small fw-medium text-dark text-wrap" title={reqText}>{reqText}</span>
                                </div>
                            )}
                            {evidenceItem && (
                                <div className="d-flex justify-content-between align-items-start p-3 bg-white rounded-3 border border-success-subtle shadow-sm">
                                    <div className="d-flex align-items-start me-2" style={{minWidth: 0}}>
                                        <evidenceItem.icon size={16} className="text-success me-2 flex-shrink-0 mt-1" />
                                        <span className="small fw-medium text-dark text-wrap" title={evidenceItem.text}>{evidenceItem.text}</span>
                                    </div>
                                    <button className="btn btn-sm btn-link text-success p-0 flex-shrink-0 ms-auto" onClick={() => onPreview(evidenceItem.item, evidenceItem.type)} title="Preview Evidence"><Eye size={16} /></button>
                                </div>
                            )}
                        </div>
                    )}
                </div>
                <div className="mt-auto"> 
                    <label className="form-label small fw-bold text-muted text-uppercase mb-2">Reasoning (Optional)</label>
                    <textarea className="form-control bg-light border-0 shadow-sm text-dark mb-3" placeholder={selectedReqId && selectedContextId ? "Add your reasoning..." : "Select items to add reasoning"} value={annotation} onChange={(e) => setAnnotation(e.target.value)} disabled={!selectedReqId || !selectedContextId} rows={3} style={{resize: 'none', minHeight: '80px'}}/>
                    {(selectedReqId || selectedContextId) && <button className="btn btn-sm btn-outline-secondary w-100 mb-3" onClick={handleManualClear}>Clear Selection</button>}
                    <button className={`btn w-100 py-3 fw-bold shadow-sm ${isAlreadyPaired ? 'btn-secondary' : 'btn-primary'}`} disabled={!selectedReqId || !selectedContextId || isSubmitting || isAlreadyPaired} onClick={handleCreate}>
                        {isSubmitting ? 'Linking...' : isAlreadyPaired ? 'Already Linked' : <><MousePointerClick size={18} className="me-2"/> Create Link</>}
                    </button>
                </div>
            </div>
        </div>
    );

    return (
        <div className="animate-fade-in">
            <div className="d-none d-lg-flex row g-4 mb-5">
                <div className="col-lg-4"><RequirementColumn isMobile={false} inputRef={reqSearchInputRef} /></div>
                <div className="col-lg-4"><EvidenceColumn isMobile={false} inputRef={evidenceSearchInputRef} /></div>
                <div className="col-lg-4"><ConnectColumn isMobile={false} reqText={selectedReqText} evidenceItem={selectedEvidenceItem} onPreview={onPreview}/></div>
            </div>
            <div className="d-lg-none mb-5">
                {mobileStep === 1 && <RequirementColumn isMobile={true} inputRef={reqSearchInputRef} />}
                {mobileStep === 2 && <EvidenceColumn isMobile={true} inputRef={evidenceSearchInputRef} />}
                {mobileStep === 3 && <ConnectColumn isMobile={true} reqText={selectedReqText} evidenceItem={selectedEvidenceItem} onPreview={onPreview}/>}
            </div>
            <div className="card border-0 shadow-sm rounded-4 overflow-hidden mt-5">
                <div className="card-header bg-white py-3 px-4 border-bottom">
                    <div className="d-flex justify-content-between align-items-center flex-wrap gap-3">
                        <h6 className="fw-bold mb-0 text-dark d-flex align-items-center gap-2"><GitMerge size={18} className="text-primary"/> Mapped Pairs <span className="badge bg-light text-dark border ms-1 rounded-pill">{filteredPairs.length}</span></h6>
                        <div className="d-flex gap-2 flex-column flex-sm-row">
                            <div className="position-relative flex-grow-1">
                                <Search size={14} className="position-absolute top-50 start-0 translate-middle-y ms-3 text-muted"/>
                                <input type="text" className="form-control form-control-sm ps-5 rounded-pill bg-light border-0" placeholder="Search mappings..." value={pairSearch} onChange={(e) => setPairSearch(e.target.value)}/>
                            </div>
                            <div className="position-relative" style={{minWidth: '160px'}}>
                                <Filter size={14} className="position-absolute top-50 start-0 translate-middle-y ms-3 text-muted" style={{zIndex:10}}/>
                                <select className="form-select form-select-sm rounded-pill border-0 bg-light ps-5 pe-4 cursor-pointer" value={filter} onChange={(e) => setFilter(e.target.value)}>
                                    <option value="all">All Types</option>
                                    {TRIAGE_GROUPS.map(g => <option key={g.id} value={g.id}>{g.title}</option>)}
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="card-body p-0 custom-scroll custom-scroll-x" style={{maxHeight: '400px', overflowY: 'auto', overflowX: 'auto'}}>
                    <table className="table table-hover mb-0 align-middle">
                        <thead className="bg-light text-muted small text-uppercase sticky-top" style={{zIndex: 5}}>
                            <tr>
                                <th className="ps-4 py-3 border-0 fw-bold cursor-pointer" style={{width: '35%'}} onClick={() => requestSort('feature_text')}>Requirement <ArrowUpDown size={12} className="ms-1 opacity-50"/></th>
                                <th className="py-3 border-0 fw-bold cursor-pointer" style={{width: '30%'}} onClick={() => requestSort('context_item_text')}>CV Evidence <ArrowUpDown size={12} className="ms-1 opacity-50"/></th>
                                <th className="py-3 border-0 fw-bold" style={{width: '25%'}}>Annotation</th>
                                <th className="pe-4 py-3 border-0 text-end" style={{width: '10%'}}>Action</th>
                            </tr>
                        </thead>
                        <tbody className="border-top-0">
                            {filteredPairs.map(pair => {
                                const lookup = cvItemLookups.get(pair.context_item_id);
                                const displayEvidence = lookup?.name || pair.context_item_text || "Unknown Item";
                                return (
                                    <tr key={pair.id} className="transition-all">
                                        <td className="ps-4 py-3"><div className="text-wrap small text-dark" style={{lineHeight: '1.5', wordBreak: 'break-word'}}>{pair.feature_text}</div></td>
                                        <td className="py-3" style={{cursor: 'pointer'}} onClick={() => handleTablePreview(pair)}>
                                            <div className="d-flex align-items-center gap-2" title="Click to preview evidence">
                                                {pair.context_item_type && <span className="badge bg-white text-secondary border fw-normal" style={{fontSize: '0.65rem'}}>{pair.context_item_type.slice(0,3).toUpperCase()}</span>}
                                                <span className="text-wrap fw-medium text-primary small text-decoration-underline-hover">{displayEvidence}</span>
                                            </div>
                                        </td>
                                        <td className="py-3"><div className="d-flex align-items-center gap-2 text-muted">{pair.annotation && <MessageSquare size={14} className="opacity-50 flex-shrink-0"/>}<span className="text-wrap fst-italic small opacity-75" style={{maxWidth: '250px'}}>{pair.annotation || '-'}</span></div></td>
                                        <td className="text-end pe-4 py-3"><button className="btn btn-icon btn-sm text-danger bg-danger bg-opacity-10 hover-bg-danger hover-text-white rounded-circle transition-all" onClick={() => onDeletePair(pair.id)} title="Remove Mapping" style={{width: '32px', height: '32px'}}><Trash2 size={14}/></button></td>
                                    </tr>
                                );
                            })}
                            {filteredPairs.length === 0 && <tr><td colSpan="4" className="text-center py-5 text-muted fst-italic bg-light bg-opacity-25"><div className="mb-2"><Search size={24} className="opacity-25"/></div>No matching pairs found.</td></tr>}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};


// --- MAIN COMPONENT ---

const MappingManager = () => {
    const { applicationId } = useParams();
    const navigate = useNavigate();

    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    
    const [viewMode, setViewMode] = useState('triage');
    const [tuningMode, setTuningMode] = useState('balanced_default');
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);
    const [allSuggestions, setAllSuggestions] = useState([]);
    const [error, setError] = useState(null);
    
    const [isProcessingId, setIsProcessingId] = useState(null);
    const [isDeletingId, setIsDeletingId] = useState(null);
    const [activeTriageCategory, setActiveTriageCategory] = useState('experiences');
    const [previewItem, setPreviewItem] = useState(null);
    const [isModalOpen, setIsModalOpen] = useState(false);

    // Load Data
    const loadData = useCallback(async () => {
        try {
            const app = (await fetchApplicationDetails(applicationId)).data;
            
            // Fetch all dependencies in parallel
            const [jobRes, cvData, mappingRes] = await Promise.all([
                fetchJobDetails(app.job_id),
                fetchCVDetails(app.base_cv_id),
                fetchMappingDetails(app.mapping_id)
            ]);
            
            setData({ 
                app, 
                job: jobRes.data, 
                cv: cvData, // cvClient returns data directly
                mapping: mappingRes.data 
            });
        } catch (err) {
            console.error(err);
            setError("Failed to load application data.");
        } finally {
            setLoading(false);
        }
    }, [applicationId]);

    useEffect(() => { loadData(); }, [loadData]);

    // Reload Mapping specific helper
    const handleReloadMapping = async () => {
        const res = await fetchMappingDetails(data.app.mapping_id);
        setData(prev => ({ ...prev, mapping: res.data }));
    };

    // Suggestions Logic
    const fetchSuggestions = useCallback(async (mode) => {
        if (!data?.mapping?.id) return;
        setIsLoadingSuggestions(true);
        try {
            const res = await inferMappingPairs(data.mapping.id, mode);
            setAllSuggestions(res.data.map((s, i) => ({ ...s, id: `sugg-${i}` })));
        } catch (err) {
            console.error("AI Suggestion Error:", err);
        } finally {
            setIsLoadingSuggestions(false);
        }
    }, [data?.mapping?.id]);

    useEffect(() => {
        if (data?.mapping?.id) fetchSuggestions(tuningMode);
    }, [data?.mapping?.id, tuningMode]);

    // --- DATA TRANSFORMATION HELPERS ---

    const cvItemLookups = useMemo(() => {
        const lookups = new Map();
        if (!data?.cv) return lookups;
        
        (data.cv.experiences || []).forEach(i => lookups.set(i.id, { item: i, type: 'experiences', name: `${i.title} @ ${i.company}` }));
        (data.cv.projects || []).forEach(i => lookups.set(i.id, { item: i, type: 'projects', name: i.title }));
        (data.cv.education || []).forEach(i => lookups.set(i.id, { item: i, type: 'education', name: `${i.degree} @ ${i.institution}` }));
        (data.cv.hobbies || []).forEach(i => lookups.set(i.id, { item: i, type: 'hobbies', name: i.name }));
        return lookups;
    }, [data?.cv]);

    const groupItems = (items) => {
        const groups = { experiences: [], projects: [], education: [], hobbies: [], other: [] };
        const map = new Map();

        (items || []).forEach(item => {
            const cvId = item.context_item_id;
            if (!map.has(cvId)) {
                const lookup = cvItemLookups.get(cvId);
                map.set(cvId, {
                    cvItemId: cvId,
                    cvItem: lookup?.item,
                    cvItemText: lookup?.name || item.context_item_text || "Unknown Item",
                    cvItemType: lookup?.type || item.context_item_type || "other",
                    items: []
                });
            }
            map.get(cvId).items.push(item);
        });

        map.forEach(g => {
            if (groups[g.cvItemType]) groups[g.cvItemType].push(g);
            else groups.other.push(g);
        });
        return groups;
    };

    const acceptedPairs = data?.mapping?.pairs || [];
    const acceptedKeys = new Set(acceptedPairs.map(p => `${p.feature_id}-${p.context_item_id}`));
    const suggestionsToReview = allSuggestions.filter(s => !acceptedKeys.has(`${s.feature_id}-${s.context_item_id}`));

    const groupedSuggestions = useMemo(() => groupItems(suggestionsToReview), [suggestionsToReview, cvItemLookups]);
    const groupedPairs = useMemo(() => groupItems(acceptedPairs), [acceptedPairs, cvItemLookups]);

    const cvEvidenceList = useMemo(() => {
        if (!data?.cv) return [];
        return [
            ...(data.cv.experiences || []).map(i => ({ id: i.id, type: 'experiences', text: `${i.title} @ ${i.company}`, item: i })),
            ...(data.cv.projects || []).map(i => ({ id: i.id, type: 'projects', text: i.title, item: i })),
            ...(data.cv.education || []).map(i => ({ id: i.id, type: 'education', text: `${i.degree} @ ${i.institution}`, item: i })),
            ...(data.cv.hobbies || []).map(i => ({ id: i.id, type: 'hobbies', text: i.name, item: i })),
        ];
    }, [data?.cv]);


    // --- HANDLERS ---

    const handleAccept = async (sugg, note) => {
        setIsProcessingId(sugg.id);
        try {
            await addMappingPair(data.mapping.id, sugg.feature_id, sugg.context_item_id, sugg.context_item_type, note, sugg.feature_text, sugg.context_item_text);
            handleReloadMapping();
        } catch (e) { alert(e.message); } 
        finally { setIsProcessingId(null); }
    };

    const handleDelete = async (id) => {
        setIsDeletingId(id);
        try { 
            await deleteMappingPair(data.mapping.id, id); 
            handleReloadMapping(); 
        } catch (e) { alert(e.message); } 
        finally { setIsDeletingId(null); }
    };

    const handleManualAdd = async (reqId, ctxId, ctxType, note, sugg) => {
        try {
            await addMappingPair(data.mapping.id, reqId, ctxId, ctxType, note, sugg?.feature_text, sugg?.context_item_text);
            handleReloadMapping();
        } catch (e) { alert(e.message); }
    };

    const handlePreview = (item, type) => {
        setPreviewItem({item, type});
        setIsModalOpen(true);
    };


    if (loading) return <div className="vh-100 d-flex align-items-center justify-content-center">Loading...</div>;
    if (!data) return <div>Error loading data.</div>;

    return (
        <div className="container-xl py-4">
            <style>{`
                .hover-bg-light:hover { background-color: #f8f9fa; }
                .hover-bg-white:hover { background-color: white; }
                .hover-bg-white-25:hover { background-color: rgba(255,255,255,0.25); cursor: pointer; }
                .hover-text-dark:hover { color: #212529 !important; }
                .hover-opacity-100:hover { opacity: 1 !important; }
                .hover-bg-danger:hover { background-color: #dc3545 !important; }
                .hover-text-white:hover { color: white !important; }
                .nav-segment button { transition: all 0.2s ease; }
                .text-decoration-underline-hover:hover { text-decoration: underline; }
                .group-hover-opacity-100:hover { opacity: 1 !important; }
                .group:hover .group-hover-opacity-100 { opacity: 1 !important; }
                
                .custom-scroll::-webkit-scrollbar { width: 5px; }
                .custom-scroll::-webkit-scrollbar-track { background: transparent; }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 10px; }
                .custom-scroll::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }

                .custom-scroll-x::-webkit-scrollbar { height: 4px; }
                .custom-scroll-x::-webkit-scrollbar-track { background: transparent; }
                .custom-scroll-x::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 10px; }
                .custom-scroll-x::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }

                .border-primary-subtle { border-color: rgba(var(--bs-primary-rgb), 0.3) !important; }
                .border-success-subtle { border-color: rgba(var(--bs-success-rgb), 0.3) !important; }
                .border-dashed { border-style: dashed !important; border-width: 2px !important; border-color: rgba(0,0,0,0.1) !important; }
            `}</style>

            {isLoadingSuggestions && <LoadingOverlay mode={tuningMode} />}

            {/* --- HEADER --- */}
            <div className="d-flex flex-column flex-lg-row justify-content-between align-items-lg-center mb-5 gap-4">
                <div className="d-flex align-items-center">
                    <div>
                        <h2 className="fw-bold text-dark mb-0 h4 d-flex align-items-center gap-2">
                            <GitMerge size={20} className="text-primary"/>
                            Evidence Mapping
                        </h2>
                        <div className="d-flex gap-2 align-items-center">
                             <button onClick={() => navigate(`/application/${applicationId}`)} className="btn btn-link p-0 text-primary small fw-bold text-uppercase tracking-wide text-decoration-none">
                                Back to Dashboard
                            </button>
                            <span className="text-muted small">•</span>
                            <p className="text-muted small mb-0">Link your career history to job requirements.</p>
                        </div>
                    </div>
                </div>
                
                <div className="d-flex flex-column flex-sm-row align-items-stretch align-items-sm-center gap-2">
                    {/* AI Settings */}
                    <div className="dropdown">
                        <button className="btn btn-white border shadow-sm d-flex align-items-center justify-content-center gap-2 px-3 py-2 rounded-3" type="button" data-bs-toggle="dropdown" aria-expanded="false">
                            <Settings2 size={16} className="text-primary"/>
                            <span className="small fw-medium">{TUNING_MODES[tuningMode].split('(')[0]}</span>
                        </button>
                        <ul className="dropdown-menu shadow-lg border-0 p-2 rounded-3" style={{minWidth: '240px'}}>
                            <li className="px-3 py-1 small text-muted fw-bold text-uppercase">AI Sensitivity</li>
                            {Object.entries(TUNING_MODES).map(([k,v]) => (
                                <li key={k}>
                                    <button className={`dropdown-item rounded-2 small py-2 ${tuningMode === k ? 'active' : ''}`} onClick={() => setTuningMode(k)}>
                                        {v}
                                    </button>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* View Toggle */}
                    <div className="bg-light p-1 rounded-3 border d-inline-flex">
                        <button 
                            className={`btn btn-sm px-3 rounded-2 border-0 fw-medium ${viewMode==='triage'?'bg-white shadow-sm text-dark':'text-muted'}`} 
                            onClick={()=>setViewMode('triage')}
                        >
                            AI Triage
                        </button>
                        <button 
                            className={`btn btn-sm px-3 rounded-2 border-0 fw-medium ${viewMode==='manual'?'bg-white shadow-sm text-dark':'text-muted'}`} 
                            onClick={()=>setViewMode('manual')}
                        >
                            Manual
                        </button>
                    </div>
                </div>
            </div>

            {/* --- CONTENT --- */}
            {error && (
                <div className="alert alert-danger d-flex align-items-center gap-2 shadow-sm border-0 rounded-3 mb-4">
                    <AlertCircle size={18}/> {error}
                </div>
            )}

            {viewMode === 'triage' ? (
                <TriageModeView 
                    groupedSuggestions={groupedSuggestions}
                    groupedPairs={groupedPairs}
                    activeCategory={activeTriageCategory}
                    setActiveCategory={setActiveTriageCategory}
                    onAccept={handleAccept}
                    onIgnore={(id) => setAllSuggestions(prev => prev.filter(x => x.id !== id))}
                    onDelete={handleDelete}
                    onPreview={handlePreview}
                    isProcessingId={isProcessingId}
                    isDeletingId={isDeletingId}
                />
            ) : (
                <ManualModeView
                    job={data.job}
                    cvEvidenceList={cvEvidenceList}
                    mapping={data.mapping}
                    allSuggestions={allSuggestions}
                    cvItemLookups={cvItemLookups}
                    onAddPair={handleManualAdd}
                    onDeletePair={handleDelete}
                    onPreview={handlePreview}
                    isSubmitting={false} 
                />
            )}

            {/* Modal Logic */}
            {isModalOpen && (
                <CVItemPreviewModal
                    isOpen={isModalOpen}
                    onClose={() => setIsModalOpen(false)}
                    itemToPreview={previewItem}
                    allSkills={data.cv.skills}
                    allAchievements={data.cv.achievements}
                    allExperiences={data.cv.experiences}
                    allEducation={data.cv.education}
                    allHobbies={data.cv.hobbies}
                />
            )}
        </div>
    );
};

export default MappingManager;