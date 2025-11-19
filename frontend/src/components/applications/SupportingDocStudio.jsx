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
    fetchMappingDetails      
} from '../../api/applicationClient.js';
import { fetchCVDetails } from '../../api/cvClient.js'; 


import PromptModal from './PromptModal.jsx';
import CVItemPreviewModal from './CVItemPreviewModal.jsx';
import ParagraphStudio from './ParagraphStudio.jsx';
import { 
    Wand2, Loader2, BrainCircuit, Sparkles, Plus, 
    Layout, GripVertical, Lock, Edit3, ArrowLeft, Trash2, Eye // Added Trash2, Eye
} from 'lucide-react';

import { 
    DndContext, closestCenter, KeyboardSensor, PointerSensor, 
    useSensor, useSensors, DragOverlay 
} from '@dnd-kit/core';
import { 
    arrayMove, SortableContext, sortableKeyboardCoordinates, 
    verticalListSortingStrategy 
} from '@dnd-kit/sortable';

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
    const [activeId, setActiveId] = useState(null);

    // NEW STATE: For Document Preview (Placeholder for a more complex modal)
    const [isDocumentPreviewOpen, setIsDocumentPreviewOpen] = useState(false); 

    const sensors = useSensors(
        useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
        useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
    );

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

    // --- FIX: PREVIEW HANDLER (remains the same) ---
    const handleShowPreview = (id, typeHint) => {
        if (!activeCV) return;

        let targetId = id;
        let targetType = typeHint;
        // ... rest of handleShowPreview logic ...

        // 1. Resolve "Evidence ID" (ev-...) to "CV Item ID" (exp_...)
        if (typeof id === 'string' && id.startsWith('ev-')) {
            const pairId = id.replace('ev-', '');
            const pair = pairMap.get(pairId);
            if (pair) {
                targetId = pair.context_item_id;
                // Also refine the type hint from the pair data
                if (!targetType || targetType === 'other') {
                     targetType = pair.context_item_type;
                }
            } else {
                console.warn("Could not resolve evidence pair:", pairId);
                return;
            }
        }
        
        // 2. Find the actual object in the CV
        const collections = {
            experiences: activeCV.experiences || [],
            education: activeCV.education || [],
            projects: activeCV.projects || [],
            hobbies: activeCV.hobbies || [],
            skills: activeCV.skills || [],
            achievements: activeCV.achievements || []
        };

        let foundItem = null;

        // A. Try Type Hint first
        if (targetType && collections[targetType]) {
            foundItem = collections[targetType].find(i => i.id === targetId);
        }

        // B. Fallback: Brute Force Search
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

    // RENAME FIX (Line 229) - Assumes client is imported
    const handleRename = async () => {
        if(activeIsLocked) return;
        try {
            await client.patch(`/coverletter/${effectiveDocId}`, { name: docName });
        } catch(err) { console.error("Failed to rename", err); } // Added error logging
    };

    const handleInsertParagraph = async (index) => {
        if(activeIsLocked) return;
        setIsSubmitting(true);
        try {
            const res = await addCoverLetterParagraph(doc.id, [], "Untitled Section", "", "user");
            const newPara = res.data;
            setDoc(prev => {
                const oldParas = [...prev.paragraphs];
                oldParas.splice(index, 0, newPara);
                const reordered = oldParas.map((p, idx) => ({ ...p, order: idx }));
                return { ...prev, paragraphs: reordered };
            });
        } finally { setIsSubmitting(false); }
    };

    // NEW FUNCTION: Handle Paragraph Deletion
    const handleDeleteParagraph = async (paraId) => {
        if (!window.confirm("Are you sure you want to delete this paragraph section and its content?")) return;
        if(activeIsLocked) return;
        setIsSubmitting(true);
        try {
            // Assuming the API client has a direct delete method for a paragraph
            await client.delete(`/coverletter/${doc.id}/paragraph/${paraId}`); 
            
            setDoc(prev => {
                const newParas = prev.paragraphs.filter(p => p.id !== paraId);
                const reordered = newParas.map((p, idx) => ({ ...p, order: idx }));
                return { ...prev, paragraphs: reordered };
            });
        } catch (err) { 
            console.error("Failed to delete paragraph", err);
            loadDoc(true); // Attempt a silent reload to correct state on error
        } finally { 
            setIsSubmitting(false); 
        }
    };
    
    // NEW FUNCTION: Toggle Reorder Mode
    const handleToggleReorder = () => {
        setIsReorderMode(prev => !prev);
    };

    // NEW FUNCTION: Document Preview Trigger
    const handleDocumentPreview = () => {
        // In a real application, logic to aggregate all paragraph.draft_text 
        // using RichTextEditor's logic would go here.
        setIsDocumentPreviewOpen(true);
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
    
    // ... rest of event handlers remain the same ...
    
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

    const handleDragEnd = (event) => {
        const { active, over } = event;
        setActiveId(null);
        if (active.id !== over.id) {
            setDoc((items) => {
                const oldIndex = items.paragraphs.findIndex(p => p.id === active.id);
                const newIndex = items.paragraphs.findIndex(p => p.id === over.id);
                const newOrder = arrayMove(items.paragraphs, oldIndex, newIndex);
                newOrder.forEach((p, idx) => updateCoverLetterParagraph(items.id, p.id, { order: idx }));
                return { ...items, paragraphs: newOrder };
            });
        }
    };
    
    // ... loading checks remain the same ...

    if (!effectiveDocId) return <div className="alert alert-danger m-4">Document ID missing.</div>;
    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin text-primary" /></div>;
    if (!doc) return <div className="alert alert-warning m-4">Initializing document...</div>;


    const sortedParagraphs = [...(doc?.paragraphs || [])].sort((a, b) => (a.order || 0) - (b.order || 0));

    return (
        <div className="container-xl py-4" style={{ maxWidth: '1100px' }}>
            
            {activeIsLocked && (
                <div className="alert alert-warning d-flex align-items-center gap-2 mb-4 shadow-sm border-warning">
                    <Lock size={16} />
                    <strong>Snapshot Mode:</strong> This document is locked because the application has been submitted.
                </div>
            )}

            {/* NEW A+ HEADER SECTION */}
            <div className="position-sticky top-0 bg-white pt-3 pb-3 z-3" style={{backdropFilter: 'blur(12px)', background: 'rgba(255,255,255,0.85)'}}>
                
                {/* 1. BACK/CONTEXT ROW */}
                <div className="d-flex align-items-center gap-2 text-primary mb-3">
                    <BrainCircuit size={20} />
                    <button 
                        onClick={onBack ? onBack : () => navigate(`/application/${applicationId}`)} 
                        className="btn btn-link p-0 text-primary small fw-bold text-uppercase tracking-wide text-decoration-none d-flex align-items-center gap-1"
                    >
                        <ArrowLeft size={14}/> Back to Dashboard
                    </button>
                </div>

                {/* 2. TITLE/RENAME & ACTIONS ROW (A+ Look with Padding) */}
                <div className="p-4 mb-4 rounded-4 bg-light-subtle border border-light shadow-lg d-flex flex-column flex-lg-row justify-content-between align-items-start align-items-lg-center">
                    
                    {/* Title/Rename Area */}
                    <div className="d-flex align-items-center gap-2 flex-grow-1 mb-3 mb-lg-0 me-lg-4">
                        {activeIsLocked ? (
                            <h2 className="fw-bold text-dark mb-0 tracking-tight">{docName}</h2>
                        ) : (
                            <input 
                                type="text" 
                                className="form-control form-control-lg border-0 p-0 fw-bold text-dark shadow-none bg-transparent"
                                style={{fontSize: '2rem', letterSpacing: '-0.03em'}}
                                value={docName}
                                onChange={(e) => setDocName(e.target.value)}
                                onBlur={handleRename}
                            />
                        )}
                        {!activeIsLocked && <Edit3 size={16} className="text-muted opacity-50" />}
                    </div>

                    {/* Action Buttons (New Layout) */}
                    {!activeIsLocked && (
                        <div className="d-flex gap-3 flex-wrap align-items-center flex-shrink-0">
                            
                            {/* Preview Button */}
                            <button 
                                className="btn btn-outline-secondary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift" 
                                onClick={handleDocumentPreview} 
                                title="Preview Document"
                            >
                                <Eye size={16}/> 
                                <span className="fw-bold d-none d-md-inline">Preview</span>
                            </button>
                            
                            {/* Reorder Button (Toggle) */}
                            <button 
                                className={`btn btn-outline-secondary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift ${isReorderMode ? 'active border-primary text-primary bg-primary-subtle' : ''}`} 
                                onClick={handleToggleReorder} 
                                title={isReorderMode ? "Exit Reorder Mode" : "Enter Reorder Mode"}
                            >
                                <GripVertical size={16}/> 
                                <span className="fw-bold d-none d-md-inline">{isReorderMode ? 'Reordering' : 'Reorder'}</span>
                            </button>

                            {/* AI Prompt Button (Text Changed) */}
                            <button className="btn btn-primary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift" onClick={async () => {
                                setIsSubmitting(true);
                                const res = await generateCoverLetterPrompt(activeMapping.id);
                                setClPromptJson(JSON.stringify(res.data, null, 2));
                                setIsPromptModalOpen(true);
                                setIsSubmitting(false);
                            }} disabled={isSubmitting}>
                                {isSubmitting ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>} 
                                <span className="fw-bold">AI Prompt</span> {/* Shortened text */}
                            </button>
                        </div>
                    )}
                </div>
            </div>
            
            {!activeIsLocked && (
                <div className="p-4 mb-5 rounded-4 bg-light-subtle border border-light shadow-sm position-relative overflow-hidden">
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

            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragStart={(e) => setActiveId(e.active.id)} onDragEnd={(e) => !activeIsLocked && handleDragEnd(e)}>
                <SortableContext items={sortedParagraphs.map(p => p.id)} strategy={verticalListSortingStrategy} disabled={!isReorderMode || activeIsLocked}>
                    
                    <SectionDivider index={0} onInsert={handleInsertParagraph} disabled={activeIsLocked} />

                    <div className="d-flex flex-column">
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
                                        onDeleteParagraph={handleDeleteParagraph} // New prop for deletion
                                        onRevertIdea={(id) => handleRevertIdea(id)}
                                        onShowPreview={handleShowPreview} 
                                        readOnly={activeIsLocked} 
                                    />
                                </div>
                                <SectionDivider index={index + 1} onInsert={handleInsertParagraph} disabled={activeIsLocked} />
                            </React.Fragment>
                        ))}
                    </div>
                </SortableContext>
                <DragOverlay>
                    {activeId ? <div className="bg-white p-3 rounded-4 shadow-lg border border-primary-subtle"><GripVertical className="d-inline me-2"/> Moving...</div> : null}
                </DragOverlay>
            </DndContext>

            <PromptModal isOpen={isPromptModalOpen} jsonString={clPromptJson} onClose={() => setIsPromptModalOpen(false)} />
            
            {/* Placeholder Modal for Full Document Preview */}
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