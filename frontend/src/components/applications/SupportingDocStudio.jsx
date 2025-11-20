// frontend/src/components/applications/SupportingDocStudio.jsx
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom'; 
import {
    fetchCoverLetterDetails,
    updateCoverLetterIdea,
    updateCoverLetterParagraph,
    addCoverLetterIdea,
    deleteCoverLetterIdea,
    addCoverLetterParagraph,
    autofillCoverLetter,
    generateCoverLetterPrompt,
    fetchApplicationDetails, 
    fetchJobDetails,         
    fetchMappingDetails,
    deleteCoverLetterParagraph      
} from '../../api/applicationClient.js';
import { fetchCVDetails } from '../../api/cvClient.js'; 

import PromptModal from './PromptModal.jsx';
import CVItemPreviewModal from './CVItemPreviewModal.jsx';
import ParagraphStudio from './ParagraphStudio.jsx';
import { 
    Wand2, Loader2, Sparkles, Plus, 
    Layout, Lock, Edit3, ArrowLeft, Trash2, Eye, Menu, X, ChevronLeft, GripVertical 
} from 'lucide-react';

// Import the window size hook for robust mobile detection
import { useWindowSize } from '../../hooks/useWindowSize';

const SectionDivider = ({ index, onInsert, disabled }) => {
    if (disabled) return <div style={{height: '24px'}}></div>;
    return (
        <div className="position-relative section-divider-zone" style={{ height: '24px', zIndex: 5 }}>
            <div className="divider-line position-absolute top-50 start-0 w-100" style={{height: '2px', background: '#e2e8f0', opacity: 0, transition: 'opacity 0.2s'}}></div>
            <button 
                className="divider-btn btn btn-primary rounded-circle p-0 position-absolute start-50 top-50 translate-middle shadow-sm"
                style={{ width: '32px', height: '32px', opacity: 0, transform: 'translate(-50%, -50%) scale(0.8)', transition: 'all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)' }}
                onClick={() => onInsert(index)}
                title="Insert New Section"
            >
                <Plus size={18} strokeWidth={3} />
            </button>
        </div>
    );
};

const SupportingDocStudio = ({
    documentId: propDocId, 
    job: propJob,
    mapping: propMapping,
    fullCV: propCV,
    isLocked: propIsLocked, 
    onBack
}) => {
    const params = useParams();
    const navigate = useNavigate();
    
    // --- MOBILE RESPONSIVE LOGIC ---
    const { width } = useWindowSize();
    const isMobile = width < 992; 
    const [showMobileMenu, setShowMobileMenu] = useState(false);

    const effectiveDocId = propDocId || params.documentId;
    const applicationId = params.applicationId; 
    
    const [doc, setDoc] = useState(null);
    const [data, setData] = useState(null); 
    const [isLoading, setIsLoading] = useState(true);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [strategy, setStrategy] = useState('standard');
    const [isReorderMode, setIsReorderMode] = useState(false);
    const [docName, setDocName] = useState("");
    const [error, setError] = useState(null);
    
    const [clPromptJson, setClPromptJson] = useState('');
    const [isPromptModalOpen, setIsPromptModalOpen] = useState(false);
    const [previewItem, setPreviewItem] = useState(null);

    const [isDocumentPreviewOpen, setIsDocumentPreviewOpen] = useState(false); 

    const activeJob = propJob || data?.job;
    const activeCV = propCV || data?.cv;
    const activeMapping = propMapping || data?.mapping;
    const activeIsLocked = propIsLocked !== undefined ? propIsLocked : data?.app?.is_locked;

    const loadDoc = useCallback(async (silent = false) => {
        if (!effectiveDocId || effectiveDocId === "undefined" || effectiveDocId === "null") return;
        if (!silent) setIsLoading(true);
        try {
            const res = await fetchCoverLetterDetails(effectiveDocId);
            const docData = res.data;
            
            let contextData = {};

            if (applicationId) {
                const appRes = await fetchApplicationDetails(applicationId);
                const app = appRes.data;

                const [jobRes, cvData, mappingRes] = await Promise.all([
                    fetchJobDetails(app.job_id),
                    fetchCVDetails(app.base_cv_id),
                    fetchMappingDetails(app.mapping_id)
                ]);
                
                contextData = {
                    app,
                    job: jobRes.data,
                    cv: cvData, 
                    mapping: mappingRes.data
                };
                setData(contextData);
            }
            
            if (docData.paragraphs.length === 0 && !contextData.app?.is_locked) {
                const filled = await autofillCoverLetter(effectiveDocId, 'standard');
                setDoc(filled.data);
                setDocName(filled.data.name);
            } else {
                setDoc(docData);
                setDocName(docData.name);
            }
        } catch (err) { 
            console.error(err);
            setError("Failed to load document data.");
        } finally { 
            setIsLoading(false); 
        }
    }, [effectiveDocId, applicationId]);

    useEffect(() => { loadDoc(); }, [loadDoc]);

    const cvLookups = useMemo(() => {
        const map = new Map();
        if (!activeCV) return map;
        const register = (arr, type, nameFn) => (arr || []).forEach(item => map.set(item.id, { ...item, _displayName: nameFn(item), _type: type }));
        register(activeCV.experiences, 'experiences', i => `${i.title} @ ${i.company}`);
        register(activeCV.projects, 'projects', i => i.title);
        register(activeCV.education, 'education', i => `${i.degree} @ ${i.institution}`);
        register(activeCV.hobbies, 'hobbies', i => i.name);
        register(activeCV.skills, 'skills', i => i.name);
        register(activeCV.achievements, 'achievements', i => i.text.substring(0, 50) + "...");
        return map;
    }, [activeCV]);

    const { ideaMap, pairMap } = useMemo(() => {
        const ideas = doc?.ideas || [];
        const pairs = activeMapping?.pairs || []; 
        const iMap = new Map(ideas.map(i => [i.id, i]));
        const pMap = new Map(pairs.map(p => {
            const cvItem = cvLookups.get(p.context_item_id);
            return [p.id, {
                ...p,
                context_item_text: cvItem?._displayName || p.context_item_text || "Unknown Evidence",
                context_item_type: cvItem?._type || p.context_item_type || "misc"
            }];
        }));
        return { ideaMap: iMap, pairMap: pMap };
    }, [doc, activeMapping, cvLookups]);

    const handleShowPreview = (id, typeHint) => {
        if (!activeCV) return;

        let targetId = id;
        let targetType = typeHint;

        if (typeof id === 'string' && id.startsWith('ev-')) {
            const pairId = id.replace('ev-', '');
            const pair = pairMap.get(pairId);
            if (pair) {
                targetId = pair.context_item_id;
                if (!targetType || targetType === 'other') {
                     targetType = pair.context_item_type;
                }
            } else {
                console.warn("Could not resolve evidence pair:", pairId);
                return;
            }
        }
        
        const collections = {
            experiences: activeCV.experiences || [],
            education: activeCV.education || [],
            projects: activeCV.projects || [],
            hobbies: activeCV.hobbies || [],
            skills: activeCV.skills || [],
            achievements: activeCV.achievements || []
        };

        let foundItem = null;

        if (targetType && collections[targetType]) {
            foundItem = collections[targetType].find(i => i.id === targetId);
        }

        if (!foundItem) {
            for (const [key, list] of Object.entries(collections)) {
                const match = list.find(i => i.id === targetId);
                if (match) {
                    foundItem = match;
                    targetType = key;
                    break;
                }
            }
        }

        if (foundItem) {
            setPreviewItem({ item: foundItem, type: targetType });
        } else {
            console.warn("Could not find CV item:", targetId);
        }
    };

    const handleRename = async () => {
        if(activeIsLocked) return;
        try {
            // await client.patch(`/coverletter/${effectiveDocId}`, { name: docName });
            console.log("Renaming doc to", docName);
        } catch(err) { console.error("Failed to rename", err); } 
    };

    const handleInsertParagraph = async (index) => {
        console.log("👇 INSERTING PARAGRAPH AT VISUAL INDEX (ORDER):", index);
        if(activeIsLocked) return;
        setIsSubmitting(true);
        try {
            // Pass 'index' as the last argument (order)
            const res = await addCoverLetterParagraph(doc.id, [], "Untitled Section", "", "user", index);
            const newPara = res.data;
            
            setDoc(prev => {
                const oldParas = [...prev.paragraphs];
                // Insert locally to match backend state
                oldParas.splice(index, 0, newPara);
                // Re-normalize local orders (though backend has already shifted them, this ensures UI consistency)
                const reordered = oldParas.map((p, idx) => ({ ...p, order: idx }));
                return { ...prev, paragraphs: reordered };
            });
        } catch(err) {
            console.error("Failed to insert paragraph:", err);
            // If insertion fails, reloading ensures we don't show a desynced state
            loadDoc(true);
        } finally { 
            setIsSubmitting(false); 
        }
    };

    const handleDeleteParagraph = async (paraId) => {
        if (!window.confirm("Are you sure you want to delete this paragraph section and its content?")) return;
        if(activeIsLocked) return;
        setIsSubmitting(true);
        try {
            await deleteCoverLetterParagraph(doc.id, paraId);
            setDoc(prev => {
                const newParas = prev.paragraphs.filter(p => p.id !== paraId);
                const reordered = newParas.map((p, idx) => ({ ...p, order: idx }));
                return { ...prev, paragraphs: reordered };
            });
        } catch (err) { 
            console.error("Failed to delete paragraph", err);
            loadDoc(true); 
        } finally { 
            setIsSubmitting(false); 
        }
    };
    
    const handleToggleReorder = () => {
        setIsReorderMode(prev => !prev);
        setShowMobileMenu(false); 
    };

    const handleDocumentPreview = () => {
        setIsDocumentPreviewOpen(true);
        setShowMobileMenu(false); 
    };

    const handleGeneratePromptClick = async () => {
        setIsSubmitting(true);
        const res = await generateCoverLetterPrompt(activeMapping.id);
        setClPromptJson(JSON.stringify(res.data, null, 2));
        setIsPromptModalOpen(true);
        setIsSubmitting(false);
        setShowMobileMenu(false); 
    };

    const handleGlobalUpdate = async (type, id, updates) => {
        if (type === 'idea') {
            setDoc(prev => ({ ...prev, ideas: prev.ideas.map(i => i.id === id ? { ...i, ...updates } : i) }));
            await updateCoverLetterIdea(doc.id, id, updates);
        } else if (type === 'paragraph') {
             setDoc(prev => ({ ...prev, paragraphs: prev.paragraphs.map(p => p.id === id ? { ...p, ...updates } : p) }));
            await updateCoverLetterParagraph(doc.id, id, updates);
        }
    };
    
    const handleAddArgument = async (paraId, classification) => {
        if (isSubmitting || activeIsLocked) return;
        setIsSubmitting(true);
        try {
            const newIdeaRes = await addCoverLetterIdea(doc.id, "New Argument", [], "", classification);
            const newIdea = newIdeaRes.data;
            const para = doc.paragraphs.find(p => p.id === paraId);
            const newIdeaIds = [...para.idea_ids, newIdea.id];
            await updateCoverLetterParagraph(doc.id, paraId, { idea_ids: newIdeaIds });
            await loadDoc(true);
        } catch (err) { console.error(err); } 
        finally { setIsSubmitting(false); }
    };

    const handleDeleteIdea = async (idea, para) => {
        if (!window.confirm(`Delete argument: "${idea.title}"?`)) return;
        if (isSubmitting || activeIsLocked) return;
        setIsSubmitting(true);
        try {
            const newIdeaIds = para.idea_ids.filter(id => id !== idea.id);
            await updateCoverLetterParagraph(doc.id, para.id, { idea_ids: newIdeaIds });
            await deleteCoverLetterIdea(doc.id, idea.id);
            await loadDoc(true);
        } catch (err) { console.error(err); } 
        finally { setIsSubmitting(false); }
    };

    const handleRevertIdea = async (id) => {
        if (activeIsLocked) return;
        await updateCoverLetterIdea(doc.id, id, { owner: 'autofill' });
        loadDoc(true);
    };

    // --- NEW: MOVE PARAGRAPH LOGIC (Replaces DragEnd) ---
    const sortedParagraphs = useMemo(() => 
        [...(doc?.paragraphs || [])].sort((a, b) => (a.order || 0) - (b.order || 0)),
    [doc?.paragraphs]);

    const handleMoveParagraph = async (index, direction) => {
        if (activeIsLocked) return;
        
        const newIndex = direction === 'up' ? index - 1 : index + 1;
        // Boundary Check
        if (newIndex < 0 || newIndex >= sortedParagraphs.length) return;

        // 1. Create a copy and swap elements
        const newParagraphs = [...sortedParagraphs];
        [newParagraphs[index], newParagraphs[newIndex]] = [newParagraphs[newIndex], newParagraphs[index]];
        
        // 2. Re-assign order property based on new array index
        const reorderedWithCorrectIndices = newParagraphs.map((p, idx) => ({ ...p, order: idx }));
        
        // 3. Optimistic UI update
        setDoc(prev => ({ ...prev, paragraphs: reorderedWithCorrectIndices }));

        // 4. Identify the two items that changed and persist them
        const p1 = reorderedWithCorrectIndices[index];   // Item moved from newIndex -> index
        const p2 = reorderedWithCorrectIndices[newIndex]; // Item moved from index -> newIndex

        setIsSubmitting(true);
        try {
            await Promise.all([
                updateCoverLetterParagraph(doc.id, p1.id, { order: p1.order }),
                updateCoverLetterParagraph(doc.id, p2.id, { order: p2.order })
            ]);
        } catch (err) {
            console.error("Failed to update order:", err);
            alert("Failed to save order. Reloading...");
            loadDoc(true);
        } finally {
            setIsSubmitting(false);
        }
    };

    if (!effectiveDocId) return <div className="alert alert-danger m-4">Document ID missing.</div>;
    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin text-primary" /></div>;
    if (!doc) return <div className="alert alert-warning m-4">Initializing document...</div>;

    return (
        <div className="container-xl py-2 py-lg-4" style={{ maxWidth: '1100px' }}>
            
            {activeIsLocked && (
                <div className="alert alert-warning d-flex align-items-center gap-2 mb-4 shadow-sm border-warning">
                    <Lock size={16} />
                    <strong>Snapshot Mode:</strong> This document is locked because the application has been submitted.
                </div>
            )}

            {/* HEADER SECTION (Integrated Back Button) */}
            <div 
                className="bg-white pt-2 pb-2 pt-lg-3 pb-lg-3 z-3" 
                style={{
                    // Static on mobile (scrolls away), Sticky on Desktop
                    position: isMobile ? 'relative' : 'sticky',
                    top: 0,
                    backdropFilter: 'blur(12px)', 
                    background: 'rgba(255,255,255,0.85)'
                }}
            >
                {/* 1. UNIFIED HEADER ROW (Back + Title + Actions) */}
                <div className={`p-3 p-lg-4 rounded-4 bg-light-subtle border border-light shadow-lg d-flex flex-column flex-lg-row justify-content-between align-items-start align-items-lg-center ${isMobile ? 'mb-3' : 'mb-4'}`}>
                    
                    {/* LEFT: Back Button + Title */}
                    <div className="d-flex align-items-center gap-2 flex-grow-1 w-100 mb-2 mb-lg-0 me-lg-4">
                        
                        {/* MODERN BACK BUTTON (Integrated) */}
                        <button 
                            onClick={onBack ? onBack : () => navigate(`/application/${applicationId}`)}
                            className="btn btn-white border shadow-sm rounded-circle d-flex align-items-center justify-content-center me-2 flex-shrink-0 hover-lift"
                            style={{width: '42px', height: '42px'}}
                            title="Back to Dashboard"
                        >
                            <ChevronLeft size={22} className="text-secondary" />
                        </button>

                        {/* Document Title */}
                        <div className="flex-grow-1">
                            {activeIsLocked ? (
                                <h2 className="fw-bold text-dark mb-0 tracking-tight" style={{fontSize: isMobile ? '1.25rem' : '2rem', lineHeight: 1.2}}>{docName}</h2>
                            ) : (
                                <input 
                                    type="text" 
                                    className="form-control form-control-lg border-0 p-0 fw-bold text-dark shadow-none bg-transparent"
                                    style={{fontSize: isMobile ? '1.25rem' : '2rem', letterSpacing: '-0.03em'}}
                                    value={docName}
                                    onChange={(e) => setDocName(e.target.value)}
                                    onBlur={handleRename}
                                />
                            )}
                        </div>

                        {!activeIsLocked && !isMobile && <Edit3 size={16} className="text-muted opacity-50 ms-2" />}
                        
                        {/* Mobile Menu Toggle (Absolute right on small screens) */}
                        {isMobile && !activeIsLocked && (
                            <button 
                                className="btn btn-light border ms-2 flex-shrink-0" 
                                onClick={() => setShowMobileMenu(!showMobileMenu)}
                                style={{width: '42px', height: '42px'}}
                            >
                                {showMobileMenu ? <X size={20} /> : <Menu size={20} />}
                            </button>
                        )}
                    </div>

                    {/* DESKTOP ACTIONS (Visible on lg+) */}
                    {!activeIsLocked && !isMobile && (
                        <div className="d-flex gap-3 flex-wrap align-items-center flex-shrink-0">
                            <button 
                                className="btn btn-outline-secondary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift" 
                                onClick={handleDocumentPreview} 
                                title="Preview Document"
                            >
                                <Eye size={16}/> <span className="fw-bold">Preview</span>
                            </button>
                            
                            <button 
                                className={`btn btn-outline-secondary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift ${isReorderMode ? 'active border-primary text-primary bg-primary-subtle' : ''}`} 
                                onClick={handleToggleReorder} 
                                title={isReorderMode ? "Exit Reorder Mode" : "Enter Reorder Mode"}
                            >
                                <GripVertical size={16}/> <span className="fw-bold">{isReorderMode ? 'Done' : 'Reorder'}</span>
                            </button>

                            <button 
                                className="btn btn-primary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift" 
                                onClick={handleGeneratePromptClick} 
                                disabled={isSubmitting}
                            >
                                {isSubmitting ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>} 
                                <span className="fw-bold">AI Prompt</span>
                            </button>
                        </div>
                    )}

                    {/* MOBILE ACTIONS MENU (Visible < lg when toggled) */}
                    {isMobile && showMobileMenu && !activeIsLocked && (
                        <div className="w-100 mt-3 pt-3 border-top animate-fade-in">
                            <div className="d-flex flex-column gap-2">
                                <button 
                                    className="btn btn-outline-secondary w-100 d-flex align-items-center justify-content-between px-3 py-2" 
                                    onClick={handleDocumentPreview}
                                >
                                    <span className="d-flex align-items-center gap-2"><Eye size={18}/> Preview Document</span>
                                </button>

                                <button 
                                    className={`btn w-100 d-flex align-items-center justify-content-between px-3 py-2 ${isReorderMode ? 'btn-primary-subtle text-primary border-primary' : 'btn-outline-secondary'}`} 
                                    onClick={handleToggleReorder}
                                >
                                    <span className="d-flex align-items-center gap-2"><GripVertical size={18}/> {isReorderMode ? 'Exit Reorder Mode' : 'Reorder Sections'}</span>
                                </button>

                                <button 
                                    className="btn btn-primary w-100 d-flex align-items-center justify-content-between px-3 py-2" 
                                    onClick={handleGeneratePromptClick} 
                                    disabled={isSubmitting}
                                >
                                    <span className="d-flex align-items-center gap-2">
                                        {isSubmitting ? <Loader2 size={18} className="animate-spin"/> : <Wand2 size={18}/>} 
                                        Generate AI Prompt
                                    </span>
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
            
            {!activeIsLocked && (
                <div className="p-3 p-lg-4 mb-4 rounded-4 bg-light-subtle border border-light shadow-sm position-relative overflow-hidden">
                     <div className="d-flex flex-wrap align-items-center gap-3 position-relative z-1">
                        <div className="d-flex align-items-center justify-content-center bg-white rounded-circle shadow-sm text-primary" style={{width: 48, height: 48}}>
                            <Sparkles size={24} />
                        </div>
                        <div className="flex-grow-1">
                            <label className="small text-muted fw-bold text-uppercase mb-1">Narrative Strategy</label>
                            <select 
                                className="form-select border-0 bg-white shadow-sm fw-semibold py-2" 
                                style={{borderRadius: '12px', cursor: 'pointer'}}
                                value={strategy} 
                                onChange={async (e) => {
                                    setIsSubmitting(true);
                                    await autofillCoverLetter(doc.id, e.target.value, 'reset').then(res => setDoc(res.data));
                                    setStrategy(e.target.value);
                                    setIsSubmitting(false);
                                }}
                            >
                                <option value="standard">Standard (Pro → Personal → Company)</option>
                                <option value="mission_driven">Mission-Driven (Company → Pro → Personal)</option>
                                <option value="specialist">Specialist (Focus on Hard Skills)</option>
                            </select>
                        </div>
                    </div>
                </div>
            )}

            {/* LIST OF PARAGRAPHS (Now Simple List, No DnD) */}
            <div className="d-flex flex-column">
                <SectionDivider index={0} onInsert={handleInsertParagraph} disabled={activeIsLocked} />
                
                {sortedParagraphs.map((para, index) => (
                    <React.Fragment key={para.id}>
                        <div className="mb-0">
                            <ParagraphStudio
                                paragraph={para}
                                jobFeatures={activeJob?.features || []}
                                fullCV={activeCV}
                                ideaMap={ideaMap}
                                pairMap={pairMap}
                                isSubmitting={isSubmitting}
                                isReorderMode={isReorderMode}
                                onUpdate={handleGlobalUpdate}
                                onAddArgument={(pid, classif) => handleAddArgument(pid, classif)} 
                                onDeleteIdea={(idea, p) => handleDeleteIdea(idea, p)}
                                onDeleteParagraph={handleDeleteParagraph} 
                                onRevertIdea={(id) => handleRevertIdea(id)}
                                onShowPreview={handleShowPreview} 
                                readOnly={activeIsLocked}
                                // Move Props
                                onMoveUp={() => handleMoveParagraph(index, 'up')}
                                onMoveDown={() => handleMoveParagraph(index, 'down')}
                                isFirst={index === 0}
                                isLast={index === sortedParagraphs.length - 1}
                            />
                        </div>
                        <SectionDivider index={index + 1} onInsert={handleInsertParagraph} disabled={activeIsLocked} />
                    </React.Fragment>
                ))}
            </div>

            <PromptModal isOpen={isPromptModalOpen} jsonString={clPromptJson} onClose={() => setIsPromptModalOpen(false)} />
            
            <PromptModal 
                isOpen={isDocumentPreviewOpen} 
                jsonString={`// Full Document Preview:\n// The document compiler is not yet fully implemented in this studio. The final document would be generated here by compiling all visible paragraphs (and their content/headings).\n\n// First paragraph content start:\n${sortedParagraphs[0]?.draft_text || '...'}`} 
                onClose={() => setIsDocumentPreviewOpen(false)} 
                title="Full Document Preview (Draft)" 
            />
            
            {previewItem && (
                <CVItemPreviewModal 
                    isOpen={!!previewItem} 
                    onClose={() => setPreviewItem(null)} 
                    itemToPreview={previewItem} 
                    allSkills={activeCV?.skills}
                    allAchievements={activeCV?.achievements}
                    allExperiences={activeCV?.experiences}
                    allEducation={activeCV?.education}
                    allHobbies={activeCV?.hobbies}
                />
            )}

            <style>{`
                .section-divider-zone:hover .divider-line { opacity: 1 !important; }
                .section-divider-zone:hover .divider-btn { opacity: 1 !important; transform: translate(-50%, -50%) scale(1) !important; }
                .section-divider-zone:active .divider-line { opacity: 1 !important; }
                .hover-lift:hover { transform: translateY(-1px); }
            `}</style>
        </div>
    );
};

export default SupportingDocStudio;