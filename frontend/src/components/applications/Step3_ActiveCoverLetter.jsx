// frontend/src/components/applications/Step3_ActiveCoverLetter.jsx
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
    generateCoverLetterPrompt,
    createCoverLetter,
    fetchCoverLetterDetails,
    addCoverLetterIdea,
    updateCoverLetterIdea,
    deleteCoverLetterIdea,
    addCoverLetterParagraph,
    updateCoverLetterParagraph,
    autofillCoverLetter
} from '../../api/applicationClient.js';

import PromptModal from './PromptModal.jsx';
import CVItemPreviewModal from './CVItemPreviewModal.jsx';
import ParagraphStudio from './ParagraphStudio.jsx';

import {
    Wand2, FileText, ArrowRight, ArrowLeft, Loader2, BrainCircuit,
    Sparkles, Plus, Layout, GripVertical
} from 'lucide-react';

import {
    DndContext, closestCenter, KeyboardSensor, PointerSensor,
    useSensor, useSensors, DragOverlay
} from '@dnd-kit/core';
import {
    arrayMove, SortableContext, sortableKeyboardCoordinates,
    verticalListSortingStrategy
} from '@dnd-kit/sortable';

// --- HOVER DIVIDER (Invisible Touch Targets) ---
const SectionDivider = ({ index, onInsert }) => {
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

const Step3_ActiveCoverLetter = ({
    application,
    mapping,
    onPrev,
    onNext,
    job,
    onCoverLetterCreated,
    fullCV
}) => {
    const [coverLetter, setCoverLetter] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState(null);
    const [strategy, setStrategy] = useState('standard');
    const [isReorderMode, setIsReorderMode] = useState(false);

    // Modal States
    const [clPromptJson, setClPromptJson] = useState('');
    const [isPromptModalOpen, setIsPromptModalOpen] = useState(false);
    const [previewItem, setPreviewItem] = useState(null);
    const [activeId, setActiveId] = useState(null);

    // --- Data Maps ---
    const cvLookups = useMemo(() => {
        const map = new Map();
        if (!fullCV) return map;
        const register = (arr, type, nameFn) => (arr || []).forEach(item => map.set(item.id, { ...item, _displayName: nameFn(item), _type: type }));
        register(fullCV.experiences, 'experiences', i => `${i.title} @ ${i.company}`);
        register(fullCV.projects, 'projects', i => i.title);
        register(fullCV.education, 'education', i => `${i.degree} @ ${i.institution}`);
        register(fullCV.hobbies, 'hobbies', i => i.name);
        register(fullCV.skills, 'skills', i => i.name);
        register(fullCV.achievements, 'achievements', i => i.text.substring(0, 50) + "...");
        return map;
    }, [fullCV]);

    const { ideaMap, pairMap } = useMemo(() => {
        const iMap = new Map(coverLetter?.ideas.map(i => [i.id, i]));
        const pMap = new Map(mapping?.pairs.map(p => {
            const cvItem = cvLookups.get(p.context_item_id);
            return [p.id, {
                ...p,
                context_item_text: cvItem?._displayName || p.context_item_text || "Unknown Evidence",
                context_item_type: cvItem?._type || p.context_item_type || "misc"
            }];
        }));
        return { ideaMap: iMap, pairMap: pMap };
    }, [coverLetter?.ideas, mapping?.pairs, cvLookups]);

    // --- Handlers ---
    const loadCoverLetter = useCallback(async () => {
        setIsLoading(true);
        try {
            let clData;
            if (application.cover_letter_id) {
                clData = (await fetchCoverLetterDetails(application.cover_letter_id)).data;
            } else {
                if (!application.job_id || !application.base_cv_id || !application.mapping_id) return;
                const res = await createCoverLetter(application.job_id, application.base_cv_id, application.mapping_id);
                clData = res.data;
                if (onCoverLetterCreated) onCoverLetterCreated(clData.id);
            }
            
            if (clData.paragraphs.length === 0) {
                const filled = await autofillCoverLetter(clData.id, 'standard');
                setCoverLetter(filled.data);
            } else {
                setCoverLetter(clData);
            }
        } catch (err) { setError("Failed to load data."); } 
        finally { setIsLoading(false); }
    }, [application, onCoverLetterCreated]);

    useEffect(() => { loadCoverLetter(); }, [loadCoverLetter]);

    const handleInsertParagraph = async (index) => {
        setIsSubmitting(true);
        try {
            const res = await addCoverLetterParagraph(coverLetter.id, [], "Untitled Section", "", "user");
            const newPara = res.data;
            setCoverLetter(prev => {
                const oldParas = [...prev.paragraphs];
                oldParas.splice(index, 0, newPara);
                const reordered = oldParas.map((p, idx) => ({ ...p, order: idx }));
                reordered.forEach(p => {
                    if (p.order !== (prev.paragraphs.find(old => old.id === p.id)?.order)) {
                        updateCoverLetterParagraph(prev.id, p.id, { order: p.order });
                    }
                });
                return { ...prev, paragraphs: reordered };
            });
        } catch (err) { alert("Failed to add section"); }
        finally { setIsSubmitting(false); }
    };

    const handleGlobalUpdate = async (type, id, updates) => {
        if (type === 'idea') {
            setCoverLetter(prev => ({ ...prev, ideas: prev.ideas.map(i => i.id === id ? { ...i, ...updates } : i) }));
            await updateCoverLetterIdea(coverLetter.id, id, updates);
        } else if (type === 'paragraph') {
             setCoverLetter(prev => ({ ...prev, paragraphs: prev.paragraphs.map(p => p.id === id ? { ...p, ...updates } : p) }));
            await updateCoverLetterParagraph(coverLetter.id, id, updates);
        }
    };

    const sensors = useSensors(
        useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
        useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
    );

    const handleDragEnd = (event) => {
        const { active, over } = event;
        setActiveId(null);
        if (active.id !== over.id) {
            setCoverLetter((items) => {
                const oldIndex = items.paragraphs.findIndex(p => p.id === active.id);
                const newIndex = items.paragraphs.findIndex(p => p.id === over.id);
                const newOrder = arrayMove(items.paragraphs, oldIndex, newIndex);
                newOrder.forEach((p, idx) => updateCoverLetterParagraph(items.id, p.id, { order: idx }));
                return { ...items, paragraphs: newOrder };
            });
        }
    };

    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin text-primary" /></div>;
    if (!coverLetter) return <div className="alert alert-warning">Initializing...</div>;

    const sortedParagraphs = [...(coverLetter?.paragraphs || [])].sort((a, b) => (a.order || 0) - (b.order || 0));

    return (
        <div className="container-xl py-4" style={{ maxWidth: '1100px' }}>
            
            {/* --- RESPONSIVE HEADER --- */}
            <div className="d-flex flex-wrap justify-content-between align-items-end mb-4 position-sticky top-0 bg-white pt-3 pb-3 border-bottom z-3" style={{backdropFilter: 'blur(12px)', background: 'rgba(255,255,255,0.85)'}}>
                <div className="mb-2 mb-lg-0">
                    <div className="d-flex align-items-center gap-2 text-primary mb-1">
                        <BrainCircuit size={20} />
                        <span className="fw-bold small text-uppercase tracking-wide">AI Studio</span>
                    </div>
                    <h2 className="fw-bold text-dark mb-0 tracking-tight" style={{letterSpacing: '-0.03em'}}>Cover Letter Strategy</h2>
                </div>
                <div className="d-flex gap-2 flex-wrap align-items-center">
                     <button 
                        className={`btn btn-sm d-flex align-items-center gap-2 transition-all rounded-pill px-3 ${isReorderMode ? 'btn-dark' : 'btn-light text-muted'}`}
                        onClick={() => setIsReorderMode(!isReorderMode)}
                    >
                        <Layout size={16} /> <span className="fw-medium">{isReorderMode ? 'Done Organizing' : 'Organize'}</span>
                    </button>
                    <div className="vr h-50 my-auto opacity-25 d-none d-sm-block"></div>
                    <button className="btn btn-sm btn-outline-dark rounded-pill d-flex align-items-center gap-2 px-3" onClick={() => {/* Copy Logic */}}>
                        <FileText size={16} /> <span className="fw-medium">Copy</span>
                    </button>
                    <button className="btn btn-sm btn-primary rounded-pill d-flex align-items-center gap-2 px-3 shadow-sm hover-lift" onClick={async () => {
                        setIsSubmitting(true);
                        const res = await generateCoverLetterPrompt(mapping.id);
                        setClPromptJson(JSON.stringify(res.data, null, 2));
                        setIsPromptModalOpen(true);
                        setIsSubmitting(false);
                    }} disabled={isSubmitting}>
                        {isSubmitting ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>} 
                        <span className="fw-bold">Generate</span>
                    </button>
                </div>
            </div>

            {/* --- STRATEGY SELECTOR --- */}
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
                                await autofillCoverLetter(coverLetter.id, e.target.value, 'reset').then(res => setCoverLetter(res.data));
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

            {/* --- WORKSPACE --- */}
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragStart={(e) => setActiveId(e.active.id)} onDragEnd={handleDragEnd}>
                <SortableContext items={sortedParagraphs.map(p => p.id)} strategy={verticalListSortingStrategy} disabled={!isReorderMode}>
                    
                    <SectionDivider index={0} onInsert={handleInsertParagraph} />

                    <div className="d-flex flex-column">
                        {sortedParagraphs.map((para, index) => (
                            <React.Fragment key={para.id}>
                                <div className="mb-0">
                                    <ParagraphStudio
                                        paragraph={para}
                                        jobFeatures={job?.features || []}
                                        fullCV={fullCV}
                                        ideaMap={ideaMap}
                                        pairMap={pairMap}
                                        isSubmitting={isSubmitting}
                                        isReorderMode={isReorderMode}
                                        onUpdate={handleGlobalUpdate}
                                        onAddArgument={(pid, classif) => addCoverLetterIdea(coverLetter.id, "New Argument", [], "", classif).then(loadCoverLetter)} 
                                        onDeleteIdea={async (idea, p) => {
                                            if(!window.confirm("Delete?")) return;
                                            await updateCoverLetterParagraph(coverLetter.id, p.id, { idea_ids: p.idea_ids.filter(id => id !== idea.id)});
                                            await loadCoverLetter();
                                        }}
                                        onRevertIdea={async (id) => { await updateCoverLetterIdea(coverLetter.id, id, { owner: 'autofill' }); loadCoverLetter(); }}
                                        onShowPreview={setPreviewItem}
                                    />
                                </div>
                                <SectionDivider index={index + 1} onInsert={handleInsertParagraph} />
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

            {/* STYLES */}
            <style>{`
                .section-divider-zone:hover .divider-line { opacity: 1 !important; }
                .section-divider-zone:hover .divider-btn { opacity: 1 !important; transform: translate(-50%, -50%) scale(1) !important; }
                /* Always show on touch devices when interacting */
                .section-divider-zone:active .divider-line { opacity: 1 !important; }
                .hover-lift:hover { transform: translateY(-1px); }
            `}</style>
        </div>
    );
};

export default Step3_ActiveCoverLetter;