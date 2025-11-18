// frontend/src/components/applications/SupportingDocStudio.jsx
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom'; 
import {
    fetchApplicationDetails, fetchJobDetails, fetchMappingDetails, // Added these imports
    fetchCoverLetterDetails,
    updateCoverLetterIdea,
    updateCoverLetterParagraph,
    addCoverLetterIdea,
    deleteCoverLetterIdea,
    addCoverLetterParagraph,
    autofillCoverLetter,
    generateCoverLetterPrompt
} from '../../api/applicationClient.js';
import { fetchCVDetails } from '../../api/cvClient.js'; // Added this import


import PromptModal from './PromptModal.jsx';
import CVItemPreviewModal from './CVItemPreviewModal.jsx';
import ParagraphStudio from './ParagraphStudio.jsx';
import { 
    Wand2, Loader2, BrainCircuit, Sparkles, Plus, 
    Layout, GripVertical, Lock, Edit3, ArrowLeft 
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

// --- MAIN COMPONENT (Now Self-Contained) ---
const SupportingDocStudio = () => {
    const { applicationId, documentId } = useParams(); 
    const navigate = useNavigate();

    // --- STATE ---
    const [data, setData] = useState(null); // Holds { app, job, cv, mapping }
    const [doc, setDoc] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [strategy, setStrategy] = useState('standard');
    const [isReorderMode, setIsReorderMode] = useState(false);
    const [docName, setDocName] = useState("");
    const [error, setError] = useState(null);
    
    // Modal States
    const [clPromptJson, setClPromptJson] = useState('');
    const [isPromptModalOpen, setIsPromptModalOpen] = useState(false);
    const [previewItem, setPreviewItem] = useState(null);
    const [activeId, setActiveId] = useState(null);

    // --- 1. LOAD ALL DATA ---
    const loadData = useCallback(async (silent = false) => {
        if (!silent) setIsLoading(true);
        try {
            // A. Fetch the Application Context (for Job/CV/Mapping IDs)
            const appRes = await fetchApplicationDetails(applicationId);
            const app = appRes.data;

            // B. Parallel Fetch Context & The Document
            // NOTE: We must handle potential nulls if IDs are missing (though unlikely for a valid app)
            const [jobRes, cvData, mappingRes, docRes] = await Promise.all([
                fetchJobDetails(app.job_id),
                fetchCVDetails(app.base_cv_id), // Returns data directly
                fetchMappingDetails(app.mapping_id),
                fetchCoverLetterDetails(documentId)
            ]);

            const loadedDoc = docRes.data;

            // C. Smart Start (Autofill if empty)
            if (loadedDoc.paragraphs.length === 0 && !app.is_locked) {
                const filled = await autofillCoverLetter(documentId, 'standard');
                setDoc(filled.data);
                setDocName(filled.data.name);
            } else {
                setDoc(loadedDoc);
                setDocName(loadedDoc.name);
            }

            setData({ 
                app, 
                job: jobRes.data, 
                cv: cvData, 
                mapping: mappingRes.data 
            });

        } catch (err) {
            console.error(err);
            setError("Failed to load document workspace.");
        } finally {
            setIsLoading(false);
        }
    }, [applicationId, documentId]);

    useEffect(() => { loadData(); }, [loadData]);

    // --- 2. MEMOIZED LOOKUPS (Dependent on fetched data) ---
    const cvLookups = useMemo(() => {
        const map = new Map();
        if (!data?.cv) return map;
        const register = (arr, type, nameFn) => (arr || []).forEach(item => map.set(item.id, { ...item, _displayName: nameFn(item), _type: type }));
        
        register(data.cv.experiences, 'experiences', i => `${i.title} @ ${i.company}`);
        register(data.cv.projects, 'projects', i => i.title);
        register(data.cv.education, 'education', i => `${i.degree} @ ${i.institution}`);
        register(data.cv.hobbies, 'hobbies', i => i.name);
        register(data.cv.skills, 'skills', i => i.name);
        register(data.cv.achievements, 'achievements', i => i.text.substring(0, 50) + "...");
        return map;
    }, [data?.cv]);

    const { ideaMap, pairMap } = useMemo(() => {
        const ideas = doc?.ideas || [];
        const pairs = data?.mapping?.pairs || [];

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
    }, [doc, data?.mapping, cvLookups]);

    const isLocked = data?.app?.is_locked || false;
    const sensors = useSensors(
        useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
        useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
    );

    // --- 3. HANDLERS ---

    const handleRename = async () => {
        if(isLocked) return;
        try {
            await client.patch(`/coverletter/${documentId}`, { name: docName });
        } catch(err) { console.error("Failed to rename"); }
    };

    const handleInsertParagraph = async (index) => {
        if(isLocked) return;
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
        if (isSubmitting || isLocked) return;
        setIsSubmitting(true);
        try {
            const newIdeaRes = await addCoverLetterIdea(doc.id, "New Argument", [], "", classification);
            const newIdea = newIdeaRes.data;
            const para = doc.paragraphs.find(p => p.id === paraId);
            const newIdeaIds = [...para.idea_ids, newIdea.id];
            await updateCoverLetterParagraph(doc.id, paraId, { idea_ids: newIdeaIds });
            await loadData(true); // Silent reload
        } catch (err) { console.error(err); } 
        finally { setIsSubmitting(false); }
    };

    const handleDeleteIdea = async (idea, para) => {
        if (!window.confirm(`Delete argument: "${idea.title}"?`)) return;
        if (isSubmitting || isLocked) return;
        setIsSubmitting(true);
        try {
            const newIdeaIds = para.idea_ids.filter(id => id !== idea.id);
            await updateCoverLetterParagraph(doc.id, para.id, { idea_ids: newIdeaIds });
            await deleteCoverLetterIdea(doc.id, idea.id);
            await loadData(true);
        } catch (err) { console.error(err); } 
        finally { setIsSubmitting(false); }
    };

    const handleRevertIdea = async (id) => {
        if (isLocked) return;
        await updateCoverLetterIdea(doc.id, id, { owner: 'autofill' });
        loadData(true);
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

    // --- 7. RENDER ---
    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin text-primary" /></div>;
    if (!doc || !data) return <div className="alert alert-danger m-4">{error || "Failed to load data."}</div>;

    const sortedParagraphs = [...(doc?.paragraphs || [])].sort((a, b) => (a.order || 0) - (b.order || 0));

    return (
        <div className="container-xl py-4" style={{ maxWidth: '1100px' }}>
            
            {/* Locked Banner */}
            {isLocked && (
                <div className="alert alert-warning d-flex align-items-center gap-2 mb-4 shadow-sm border-warning">
                    <Lock size={16} />
                    <strong>Snapshot Mode:</strong> This document is locked because the application has been submitted.
                </div>
            )}

            {/* Header */}
            <div className="d-flex flex-wrap justify-content-between align-items-end mb-4 position-sticky top-0 bg-white pt-3 pb-3 border-bottom z-3" style={{backdropFilter: 'blur(12px)', background: 'rgba(255,255,255,0.85)'}}>
                <div className="mb-2 mb-lg-0 flex-grow-1">
                    <div className="d-flex align-items-center gap-2 text-primary mb-1">
                        <BrainCircuit size={20} />
                        <button 
                            onClick={() => navigate(`/application/${applicationId}`)} 
                            className="btn btn-link p-0 text-primary small fw-bold text-uppercase tracking-wide text-decoration-none d-flex align-items-center gap-1"
                        >
                            <ArrowLeft size={14}/> Back to Dashboard
                        </button>
                    </div>
                    <div className="d-flex align-items-center gap-2">
                        {isLocked ? (
                            <h2 className="fw-bold text-dark mb-0 tracking-tight">{docName}</h2>
                        ) : (
                            <input 
                                type="text" 
                                className="form-control form-control-lg border-0 px-0 fw-bold text-dark shadow-none bg-transparent"
                                style={{fontSize: '2rem', letterSpacing: '-0.03em'}}
                                value={docName}
                                onChange={(e) => setDocName(e.target.value)}
                                onBlur={handleRename}
                            />
                        )}
                        {!isLocked && <Edit3 size={16} className="text-muted opacity-50" />}
                    </div>
                </div>
                
                {!isLocked && (
                    <div className="d-flex gap-2 flex-wrap align-items-center">
                        <button className={`btn btn-sm d-flex align-items-center gap-2 rounded-pill px-3 ${isReorderMode ? 'btn-dark' : 'btn-light text-muted'}`} onClick={() => setIsReorderMode(!isReorderMode)}>
                            <Layout size={16} /> <span className="fw-medium">{isReorderMode ? 'Done' : 'Organize'}</span>
                        </button>
                         <button className="btn btn-sm btn-primary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift" onClick={async () => {
                            setIsSubmitting(true);
                            const res = await generateCoverLetterPrompt(data.mapping.id);
                            setClPromptJson(JSON.stringify(res.data, null, 2));
                            setIsPromptModalOpen(true);
                            setIsSubmitting(false);
                        }} disabled={isSubmitting}>
                            {isSubmitting ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>} 
                            <span className="fw-bold">Generate AI Prompt</span>
                        </button>
                    </div>
                )}
            </div>
            
            {/* Strategy Selector */}
            {!isLocked && (
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

            {/* Workspace */}
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragStart={(e) => setActiveId(e.active.id)} onDragEnd={(e) => !isLocked && handleDragEnd(e)}>
                <SortableContext items={sortedParagraphs.map(p => p.id)} strategy={verticalListSortingStrategy} disabled={!isReorderMode || isLocked}>
                    
                    <SectionDivider index={0} onInsert={handleInsertParagraph} disabled={isLocked} />

                    <div className="d-flex flex-column">
                        {sortedParagraphs.map((para, index) => (
                            <React.Fragment key={para.id}>
                                <div className="mb-0">
                                    <ParagraphStudio
                                        paragraph={para}
                                        jobFeatures={data.job.features}
                                        fullCV={data.cv}
                                        ideaMap={ideaMap}
                                        pairMap={pairMap}
                                        isSubmitting={isSubmitting}
                                        isReorderMode={isReorderMode}
                                        onUpdate={handleGlobalUpdate}
                                        onAddArgument={(pid, classif) => handleAddArgument(pid, classif)} 
                                        onDeleteIdea={(idea, p) => handleDeleteIdea(idea, p)}
                                        onRevertIdea={(id) => handleRevertIdea(id)}
                                        onShowPreview={setPreviewItem}
                                        readOnly={isLocked} 
                                    />
                                </div>
                                <SectionDivider index={index + 1} onInsert={handleInsertParagraph} disabled={isLocked} />
                            </React.Fragment>
                        ))}
                    </div>
                </SortableContext>
                <DragOverlay>
                    {activeId ? <div className="bg-white p-3 rounded-4 shadow-lg border border-primary-subtle"><GripVertical className="d-inline me-2"/> Moving...</div> : null}
                </DragOverlay>
            </DndContext>

            <PromptModal isOpen={isPromptModalOpen} jsonString={clPromptJson} onClose={() => setIsPromptModalOpen(false)} />
            {previewItem && <CVItemPreviewModal isOpen={!!previewItem} onClose={() => setPreviewItem(null)} itemToPreview={previewItem} />}

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