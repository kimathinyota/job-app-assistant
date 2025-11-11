// frontend/src/components/applications/Step3_BuildCoverLetter.jsx
import React, { useState, useEffect, useMemo, forwardRef } from 'react';
import {
    generateCoverLetterPrompt,
    createCoverLetter,
    fetchCoverLetterDetails,
    addCoverLetterIdea,
    updateCoverLetterIdea,
    deleteCoverLetterIdea,
    addCoverLetterParagraph,
    updateCoverLetterParagraph,
    deleteCoverLetterParagraph
} from '../../api/applicationClient.js';
import PromptModal from './PromptModal.jsx';
import {
    DndContext,
    DragOverlay,
    closestCorners,
    KeyboardSensor,
    PointerSensor,
    useSensor,
    useSensors,
    useDroppable
} from '@dnd-kit/core';
import {
    SortableContext,
    arrayMove,
    sortableKeyboardCoordinates,
    useSortable,
    verticalListSortingStrategy
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

// --- STYLES ---
const DND_STYLES = `
/* (Styles are unchanged from your file) */
.cl-builder-container {
    display: flex;
    flex-direction: row;
    gap: 1.5rem;
    width: 100%;
    overflow-x: auto;
    padding-bottom: 1rem;
}
.cl-column {
    flex: 1 1 33.33%;
    min-width: 320px;
    height: 75vh;
    display: flex;
    flex-direction: column;
}
.cl-column-header {
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--bs-primary);
}
.cl-column-body {
    flex-grow: 1;
    overflow-y: auto;
    background-color: var(--bs-light);
    border-radius: 0.375rem;
    padding: 0.75rem;
}
.cl-column-body.over {
    background-color: var(--bs-primary-bg-subtle);
    outline: 2px dashed var(--bs-primary);
}
.pair-chip {
    cursor: grab;
    touch-action: none;
    background-color: var(--bs-white);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    padding: 0.5rem 0.75rem;
    box-shadow: var(--bs-box-shadow-sm);
    margin-bottom: 0.5rem;
    position: relative;
}
.pair-chip-icon {
    position: absolute;
    top: 0.5rem;
    right: 0.75rem;
    font-size: 1.25rem;
    color: var(--bs-secondary);
}
.pair-chip.dragging {
    opacity: 0.9;
    box-shadow: var(--bs-box-shadow-lg);
}
.idea-card {
    background-color: var(--bs-white);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    margin-bottom: 1rem;
    touch-action: none;
}
.idea-card-header {
    /* cursor: grab; <-- REMOVED from header */
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--bs-border-color-translucent);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.idea-card-body {
    padding: 1rem;
    min-height: 80px;
    background-color: var(--bs-white);
}
.idea-card-body.over {
    background-color: var(--bs-success-bg-subtle);
}
.idea-card.dragging {
    opacity: 0.9;
    box-shadow: var(--bs-box-shadow-lg);
}
.paragraph-card {
    background-color: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
    margin-bottom: 1rem;
    touch-action: none;
}
.paragraph-card-header {
    /* cursor: grab; <-- REMOVED from header */
    padding: 0.75rem 1rem;
    border-bottom: 1px solid var(--bs-border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: var(--bs-white);
}
.paragraph-card-body {
    padding: 1rem;
    min-height: 100px;
}
.paragraph-card-body.over {
    background-color: var(--bs-primary-bg-subtle);
}
.modal-backdrop.show {
    opacity: 0.5;
}
.modal.show {
    display: block;
}
`;

// --- Helper Functions (Unchanged) ---
const getItemIcon = (itemType) => {
    switch (itemType) {
        case 'experiences': return 'bi bi-briefcase-fill';
        case 'projects': return 'bi bi-tools';
        case 'education': return 'bi bi-mortarboard-fill';
        case 'hobbies': return 'bi bi-controller';
        default: return 'bi bi-file-earmark-text-fill';
    }
};

// --- Draggable Components ---

// Col 1: Pair Chip (NOW WITH onRemove BUTTON)
const PairChip = forwardRef(({ pair, onRemove, ...props }, ref) => (
    <div ref={ref} {...props} className="pair-chip">
        {/* --- NEW: Close button --- */}
        {onRemove && (
            <button
                type="button"
                className="btn-close btn-sm"
                onClick={(e) => {
                    e.stopPropagation(); // Prevents drag-and-drop
                    onRemove();
                }}
                title="Remove pair"
                style={{ 
                    position: 'absolute', 
                    top: '5px', 
                    right: '5px', 
                    zIndex: 2 
                }}
            ></button>
        )}
        {/* --- MODIFIED: Icon position adjusts if close button exists --- */}
        <i 
            className={`${getItemIcon(pair.context_item_type)} pair-chip-icon`} 
            style={{ right: onRemove ? '30px' : '0.75rem' }}
        ></i>
        {/* --- MODIFIED: Text padding to avoid close button --- */}
        <strong className="d-block small pe-4">{pair.feature_text}</strong>
        <span className="text-muted small pe-3">
            {pair.context_item_text}
        </span>
        {pair.annotation && (
            <em className="d-block small text-primary mt-1 border-top pt-1 pe-3">
                Note: {pair.annotation}
            </em>
        )}
    </div>
));


// Col 2: Idea Card (DELETE BUTTON FIXED)
const IdeaCard = ({ idea, pairsInIdea, onDelete, onUpdateAnnotation, onRemovePair }) => {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging
    } = useSortable({ 
        id: `pool-idea-${idea.id}`,
        data: { 
            type: 'idea', 
            idea 
        } 
    });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
    };
    
    const [annotation, setAnnotation] = useState(idea.annotation || '');
    const [isEditing, setIsEditing] = useState(false);

    const handleAnnotationBlur = () => {
        setIsEditing(false);
        if (annotation !== (idea.annotation || '')) {
            onUpdateAnnotation(idea.id, annotation);
        }
    };
    
    useEffect(() => {
        setAnnotation(idea.annotation || '');
    }, [idea.annotation]);

    return (
        <div ref={setNodeRef} style={style} className={`idea-card ${isDragging ? 'dragging' : ''}`}>
            {/* --- FIX: Listeners moved to h6 --- */}
            <div className="idea-card-header">
                <h6 
                    className="h6 mb-0" 
                    {...attributes} 
                    {...listeners} 
                    style={{ cursor: 'grab', flexGrow: 1 }}
                >
                    {idea.title}
                </h6>
                <button
                    className="btn-close btn-sm"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDelete(idea.id);
                    }}
                    title="Delete Idea"
                />
            </div>
            <SortableContext
                id={`idea-card-body-${idea.id}`}
                items={pairsInIdea.map(p => `idea-${idea.id}-pair-${p.id}`)}
                strategy={verticalListSortingStrategy}
            >
                <div className={`idea-card-body`}>
                    {isEditing ? (
                        <textarea
                            className="form-control form-control-sm mb-2"
                            value={annotation}
                            onChange={(e) => setAnnotation(e.target.value)}
                            onBlur={handleAnnotationBlur}
                            autoFocus
                            rows={3}
                        />
                    ) : (
                        <p 
                            className="small text-muted fst-italic"
                            onClick={() => setIsEditing(true)}
                            style={{minHeight: '1.5rem', cursor: 'pointer', whiteSpace: 'pre-wrap'}}
                        >
                            {annotation || "Click to add general notes..."}
                        </p>
                    )}

                    {pairsInIdea.map(pair => (
                        <SortablePairChip 
                            key={pair.id} 
                            pair={pair} 
                            ideaId={idea.id}
                            onRemove={onRemovePair} // <-- NEW: Pass handler down
                        />
                    ))}
                    {pairsInIdea.length === 0 && !annotation && (
                         <p className="small text-muted text-center mb-0">
                            Drag proof here
                        </p>
                    )}
                </div>
            </SortableContext>
        </div>
    );
};


// Col 3: Paragraph Card (DELETE BUTTON FIXED)
const ParagraphCard = ({ paragraph, ideasInParagraph, onDelete }) => {
    const {
        attributes,
        listeners,
        setNodeRef: setSortableNodeRef,
        transform,
        transition,
    } = useSortable({ 
        id: `para-${paragraph.id}`,
        data: { 
            type: 'paragraph', 
            paragraph 
        } 
    });

    const {
        setNodeRef: setDroppableNodeRef,
        isOver
    } = useDroppable({
        id: `para-card-body-${paragraph.id}`,
        data: {
            type: 'paragraph-body',
            paragraph
        }
    });

    const style = { transform: CSS.Transform.toString(transform), transition };

    return (
        <div ref={setSortableNodeRef} style={style} className="paragraph-card">
            {/* --- FIX: Listeners moved to h6 --- */}
            <div className="paragraph-card-header">
                <h6 
                    className="h6 mb-0" 
                    {...attributes} 
                    {...listeners} 
                    style={{ cursor: 'grab', flexGrow: 1 }}
                >
                    {paragraph.purpose}
                </h6>
                <button
                    className="btn-close btn-sm"
                    onClick={(e) => {
                        e.stopPropagation();
                        onDelete(paragraph.id);
                    }}
                    title="Delete Paragraph"
                />
            </div>
            
            <SortableContext
                id={`para-card-body-${paragraph.id}`}
                items={ideasInParagraph.map(i => `para-${paragraph.id}-idea-${i.id}`)}
                strategy={verticalListSortingStrategy}
            >
                <div ref={setDroppableNodeRef} className={`paragraph-card-body ${isOver ? 'over' : ''}`}>
                    {ideasInParagraph.length === 0 && (
                        <p className="small text-muted fst-italic mb-0">
                            Drag arguments here
                        </p>
                    )}
                    {ideasInParagraph.map(idea => (
                        <SortableIdeaCard
                            key={idea.id}
                            idea={idea}
                            paragraphId={paragraph.id}
                        />
                    ))}
                </div>
            </SortableContext>
        </div>
    );
};

// --- Sortable Wrappers ---

// Sortable Pair (NOW ACCEPTS onRemove)
const SortablePairChip = ({ pair, ideaId, onRemove }) => {
    const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging
    } = useSortable({ 
        id: ideaId ? `idea-${ideaId}-pair-${pair.id}` : `pool-pair-${pair.id}`,
        data: { 
            type: 'pair', 
            pair,
            sourceIdeaId: ideaId
        } 
    });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        opacity: isDragging ? 0.5 : 1
    };

    return (
        <PairChip
            ref={setNodeRef}
            style={style}
            pair={pair}
            // --- NEW: Pass onRemove handler to PairChip ---
            onRemove={onRemove ? () => onRemove(ideaId, pair.id) : null}
            {...attributes}
            {...listeners}
        />
    );
};

// Sortable Idea (Unchanged)
const SortableIdeaCard = ({ idea, paragraphId }) => {
     const {
        attributes,
        listeners,
        setNodeRef,
        transform,
        transition,
        isDragging
    } = useSortable({ 
        id: `para-${paragraphId}-idea-${idea.id}`, 
        data: { 
            type: 'idea', 
            idea,
            sourceParagraphId: paragraphId
        } 
    });

    const style = {
        transform: CSS.Transform.toString(transform),
        transition,
        opacity: isDragging ? 0.5 : 1
    };
    
    return (
        <div ref={setNodeRef} style={style} {...attributes} {...listeners} className="idea-card">
            <div className="idea-card-header">
                <h6 className="h6 mb-0 small">{idea.title}</h6>
            </div>
             {idea.annotation && (
                <div className="idea-card-body py-2">
                    <p className="small text-muted fst-italic mb-0" style={{whiteSpace: 'pre-wrap'}}>
                        {idea.annotation.substring(0, 50)}
                        {idea.annotation.length > 50 ? "..." : ""}
                    </p>
                </div>
            )}
        </div>
    );
};

// Droppable Column Body (Unchanged)
const DroppableColumnBody = ({ id, items, children }) => {
    const { setNodeRef, isOver } = useDroppable({ id: id });

    return (
        <SortableContext
            id={id}
            items={items}
            strategy={verticalListSortingStrategy}
        >
            <div ref={setNodeRef} className={`cl-column-body ${isOver ? 'over' : ''}`}>
                {children}
            </div>
        </SortableContext>
    );
};

// --- Main Step 3 Component ---
const Step3_BuildCoverLetter = ({ 
    application, 
    mapping, 
    onPrev, 
    onNext, 
    job, 
    onCoverLetterCreated 
}) => {
    const [coverLetter, setCoverLetter] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeDragId, setActiveDragId] = useState(null);
    const [pairFilterId, setPairFilterId] = useState('all');
    const [newParaPurpose, setNewParaPurpose] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [clPromptJson, setClPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);
    
    const [showResearchModal, setShowResearchModal] = useState(false);
    const [companyResearch, setCompanyResearch] = useState("");

    // --- Data Loading & Scaffolding (Unchanged) ---
    const loadCoverLetter = async () => {
        if (!coverLetter) setIsLoading(true); 
        setError(null);
        try {
            let clData;
            if (application.cover_letter_id) {
                clData = (await fetchCoverLetterDetails(application.cover_letter_id)).data;
            } else {
                const res = await createCoverLetter(application.job_id, application.base_cv_id, application.mapping_id);
                clData = res.data;
                
                if (onCoverLetterCreated) {
                    await onCoverLetterCreated(clData.id);
                }
                
                await addCoverLetterParagraph(clData.id, [], "Introduction");
                await addCoverLetterParagraph(clData.id, [], "Body 1 (Why Me)");
                await addCoverLetterParagraph(clData.id, [], "Body 2 (Why You)");
                await addCoverLetterParagraph(clData.id, [], "Conclusion");
                
                clData = (await fetchCoverLetterDetails(clData.id)).data;
            }
            setCoverLetter(clData);
        } catch (err) {
            setError("Failed to load or create cover letter data.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const reloadCoverLetter = async () => {
        if (!coverLetter?.id) {
             await loadCoverLetter();
             return;
        }
        try {
             setIsSubmitting(true);
             const clData = (await fetchCoverLetterDetails(coverLetter.id)).data;
             setCoverLetter(clData);
        } catch (err) {
             console.error("Failed to reload CL:", err);
             setError("Failed to refresh data.");
        } finally {
            setIsSubmitting(false);
        }
    }

    useEffect(() => {
        loadCoverLetter();
    }, [application.id, application.cover_letter_id]);

    // --- Memos for Data Manipulation (Unchanged) ---
    const {
        pairMap,
        ideaMap,
        filteredPairs,
        availableIdeas,
        paragraphs,
        itemsById,
    } = useMemo(() => {
        const pMap = new Map((mapping?.pairs || []).map(p => [p.id, p]));
        const iMap = new Map(coverLetter ? coverLetter.ideas.map(i => [i.id, i]) : []);
        
        const fPairs = (mapping?.pairs || []).filter(p =>
            pairFilterId === 'all' || p.feature_id === pairFilterId
        );

        const pairedIdeaIds = new Set();
        (coverLetter?.paragraphs || []).forEach(para => {
            para.idea_ids.forEach(id => pairedIdeaIds.add(id));
        });
        const avIdeas = (coverLetter?.ideas || []).filter(i => !pairedIdeaIds.has(i.id));
        
        const paras = (coverLetter?.paragraphs || []).sort((a, b) => a.order - b.order);

        const allItems = {};
        (mapping?.pairs || []).forEach(p => {
             allItems[`pool-pair-${p.id}`] = { ...p, type: 'pair' };
             (coverLetter?.ideas || []).forEach(i => {
                if (i.mapping_pair_ids.includes(p.id)) {
                    allItems[`idea-${i.id}-pair-${p.id}`] = { ...p, type: 'pair' };
                }
             });
        });
        (coverLetter?.ideas || []).forEach(i => {
            allItems[`pool-idea-${i.id}`] = { ...i, type: 'idea' };
            (coverLetter?.paragraphs || []).forEach(p => {
                if (p.idea_ids.includes(i.id)) {
                    allItems[`para-${p.id}-idea-${i.id}`] = { ...i, type: 'idea' };
                }
            });
        });

        return {
            pairMap: pMap,
            ideaMap: iMap,
            filteredPairs: fPairs,
            availableIdeas: avIdeas,
            paragraphs: paras,
            itemsById: allItems
        };
    }, [mapping?.pairs, coverLetter, job?.features, pairFilterId]);

    // --- Form Handlers ---
    
    // --- 'Smart Suggest' Handler (REPLACED) ---
    const handleSmartSuggest = () => {
        setShowResearchModal(true);
    };

    // --- NEW: Handler for the modal submission ---
    const handleSubmitSmartArguments = async () => {
        setIsSubmitting(true);
        setShowResearchModal(false);

        try {
            const whyMePara = paragraphs.find(p => p.purpose === 'Body 1 (Why Me)');
            const whyYouPara = paragraphs.find(p => p.purpose === 'Body 2 (Why You)');
            
            const ideaCreationPromises = [];

            // 1. Create Company Research Idea
            if (companyResearch.trim()) {
                const title = "Company Mission & Values";
                const annotation = `Agrees with mission statement because... ${companyResearch}`;
                
                ideaCreationPromises.push(
                    addCoverLetterIdea(coverLetter.id, title, [], annotation)
                        .then(res => ({
                            newIdea: res.data,
                            intendedParagraph: whyYouPara
                        }))
                );
            }

            // 2. Create Requirement-based Ideas
            const pairsByFeature = mapping.pairs.reduce((acc, pair) => {
                if (!pair.feature_id) return acc;
                let alreadyUsed = false;
                for (const idea of coverLetter.ideas) {
                    if (idea.mapping_pair_ids.includes(pair.id)) {
                        alreadyUsed = true;
                        break;
                    }
                }
                if (alreadyUsed) return acc;

                if (!acc[pair.feature_id]) {
                    acc[pair.feature_id] = {
                        title: `Regarding: ${pair.feature_text}`,
                        pairIds: []
                    };
                }
                acc[pair.feature_id].pairIds.push(pair.id);
                return acc;
            }, {});

            Object.values(pairsByFeature).forEach(group => {
                if (group.pairIds.length > 0) { // <-- BUG FIX IS HERE
                    ideaCreationPromises.push(
                        addCoverLetterIdea(coverLetter.id, group.title, group.pairIds, null)
                            .then(res => ({
                                newIdea: res.data,
                                intendedParagraph: whyMePara
                            }))
                    );
                }
            });

            if (ideaCreationPromises.length === 0 && !companyResearch.trim()) {
                alert("No new arguments to suggest. All mapped pairs are already in use.");
                setIsSubmitting(false);
                return;
            }

            const creationResults = await Promise.all(ideaCreationPromises);

            const updatesByPara = new Map();
            creationResults.forEach(({ newIdea, intendedParagraph }) => {
                if (intendedParagraph) {
                    if (!updatesByPara.has(intendedParagraph.id)) {
                        updatesByPara.set(intendedParagraph.id, {
                            para: intendedParagraph,
                            newIdeaIds: [...intendedParagraph.idea_ids]
                        });
                    }
                    updatesByPara.get(intendedParagraph.id).newIdeaIds.push(newIdea.id);
                }
            });

            const updatePromises = [];
            updatesByPara.forEach(({ para, newIdeaIds }) => {
                updatePromises.push(
                    updateCoverLetterParagraph(coverLetter.id, para.id, { idea_ids: newIdeaIds })
                );
            });

            await Promise.all(updatePromises);

        } catch (err) {
            alert("Failed to suggest arguments.");
            console.error(err);
        } finally {
            await reloadCoverLetter();
            setCompanyResearch("");
            setIsSubmitting(false);
        }
    };
    
    // --- NEW: Handler to remove a pair from an idea ---
    const handleRemovePairFromIdea = async (ideaId, pairId) => {
        const idea = ideaMap.get(ideaId);
        if (!idea) return;
        
        setIsSubmitting(true);
        try {
            const newPairIds = idea.mapping_pair_ids.filter(id => id !== pairId);
            await updateCoverLetterIdea(coverLetter.id, ideaId, { mapping_pair_ids: newPairIds });
            await reloadCoverLetter();
        } catch (err) {
            alert("Failed to remove pair.");
            await reloadCoverLetter();
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleUpdateIdeaAnnotation = async (ideaId, newAnnotation) => {
         try {
            await updateCoverLetterIdea(coverLetter.id, ideaId, { annotation: newAnnotation });
            setCoverLetter(prev => ({
                ...prev,
                ideas: prev.ideas.map(i => 
                    i.id === ideaId ? { ...i, annotation: newAnnotation } : i
                )
            }));
        } catch (err) { 
            alert("Failed to update annotation."); 
            await reloadCoverLetter();
        }
    };

    const handleDeleteIdea = async (ideaId) => {
        if (!window.confirm("Delete this idea?")) return;
        setIsSubmitting(true);
        try {
            await deleteCoverLetterIdea(coverLetter.id, ideaId);
            await reloadCoverLetter();
        } catch (err) { alert("Failed to delete idea."); }
        finally { setIsSubmitting(false); }
    };

    const handleCreateParagraph = async (e) => {
        e.preventDefault();
        if (!newParaPurpose.trim()) return;
        setIsSubmitting(true);
        try {
            await addCoverLetterParagraph(coverLetter.id, [], newParaPurpose);
            await reloadCoverLetter();
            setNewParaPurpose('');
        } catch (err) { alert("Failed to create paragraph."); }
        finally { setIsSubmitting(false); }
    };

    const handleDeleteParagraph = async (paraId) => {
        if (!window.confirm("Delete this paragraph?")) return;
        setIsSubmitting(true);
        try {
            await deleteCoverLetterParagraph(coverLetter.id, paraId);
            await reloadCoverLetter();
        } catch (err) { alert("Failed to delete paragraph."); }
        finally { setIsSubmitting(false); }
    };

    // --- Drag-and-Drop Handlers (Unchanged) ---
    const sensors = useSensors(
        useSensor(PointerSensor),
        useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
    );

    const handleDragStart = (event) => setActiveDragId(event.active.id);
    const handleDragCancel = () => setActiveDragId(null);

    const handleDragEnd = async (event) => {
        const { active, over } = event;
        setActiveDragId(null);
        if (!over) return;
        
        const activeData = active.data.current;
        const overId = over.id;

        if (active.id === over.id) return;
        
        const overContainerId = (over.data.current?.sortable?.containerId) 
            ? over.data.current.sortable.containerId 
            : overId;

        setIsSubmitting(true);
        
        try {
            if (activeData?.type === 'pair') {
                const pairId = activeData.pair.id;
                const sourceIdeaId = activeData.sourceIdeaId;
                
                if (overContainerId.startsWith('idea-card-body-')) {
                    const targetIdeaId = overContainerId.split('idea-card-body-')[1];
                    
                    if (sourceIdeaId && sourceIdeaId !== targetIdeaId) {
                        const sourceIdea = ideaMap.get(sourceIdeaId);
                        const newPairIds = sourceIdea.mapping_pair_ids.filter(id => id !== pairId);
                        await updateCoverLetterIdea(coverLetter.id, sourceIdeaId, { mapping_pair_ids: newPairIds });
                    }
                    
                    const targetIdea = ideaMap.get(targetIdeaId);
                    if (targetIdea && !targetIdea.mapping_pair_ids.includes(pairId)) {
                        const newPairIds = [...targetIdea.mapping_pair_ids, pairId];
                        await updateCoverLetterIdea(coverLetter.id, targetIdeaId, { mapping_pair_ids: newPairIds });
                    }
                }
                else if (overContainerId === 'pair-pool' && sourceIdeaId) {
                    const sourceIdea = ideaMap.get(sourceIdeaId);
                    const newPairIds = sourceIdea.mapping_pair_ids.filter(id => id !== pairId);
                    await updateCoverLetterIdea(coverLetter.id, sourceIdeaId, { mapping_pair_ids: newPairIds });
                }
            }
            
            if (activeData?.type === 'idea') {
                const ideaId = activeData.idea.id;
                const sourceParagraphId = activeData.sourceParagraphId;
                
                if (overContainerId.startsWith('para-card-body-')) {
                    const targetParagraphId = overContainerId.split('para-card-body-')[1];
                    
                    if (sourceParagraphId && sourceParagraphId !== targetParagraphId) {
                        const sourcePara = paragraphs.find(p => p.id === sourceParagraphId);
                        const newIdeaIds = sourcePara.idea_ids.filter(id => id !== ideaId);
                        await updateCoverLetterParagraph(coverLetter.id, sourceParagraphId, { idea_ids: newIdeaIds });
                    }
                    
                    const targetPara = paragraphs.find(p => p.id === targetParagraphId);
                    if (targetPara && !targetPara.idea_ids.includes(ideaId)) {
                        const newIdeaIds = [...targetPara.idea_ids, ideaId];
                        await updateCoverLetterParagraph(coverLetter.id, targetParagraphId, { idea_ids: newIdeaIds });
                    }
                }
                else if (overContainerId === 'idea-pool' && sourceParagraphId) {
                    const sourcePara = paragraphs.find(p => p.id === sourceParagraphId);
                    const newIdeaIds = sourcePara.idea_ids.filter(id => id !== ideaId);
                    await updateCoverLetterParagraph(coverLetter.id, sourceParagraphId, { idea_ids: newIdeaIds });
                }
            }

            await reloadCoverLetter();

        } catch (err) {
            alert("Error during drag operation.");
            console.error(err);
            await reloadCoverLetter();
        } finally {
            setIsSubmitting(false);
        }
    };
    
    // --- Generate Prompt (Unchanged) ---
    const handleGeneratePrompt = async () => {
        if (coverLetter.paragraphs.length === 0) {
            alert("Please create at least one paragraph for the outline.");
            return;
        }
        setIsLoadingPrompt(true);
        try {
            const res = await generateCoverLetterPrompt(mapping.id);
            setClPromptJson(JSON.stringify(res.data, null, 2));
            setIsModalOpen(true);
        } catch (err) { alert("Failed to generate prompt."); }
        finally { setIsLoadingPrompt(false); }
    };
    
    // --- Render ---
    if (isLoading && !coverLetter) return <div>Loading Cover Letter builder...</div>;
    if (error) return <div className="alert alert-danger">{error}</div>;
    if (!coverLetter || !mapping || !job) return <div className="alert alert-info">Initializing...</div>;

    const activeDragItem = activeDragId ? itemsById[activeDragId] : null;

    return (
        <DndContext
            sensors={sensors}
            collisionDetection={closestCorners}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
            onDragCancel={handleDragCancel}
        >
            <style>{DND_STYLES}</style>
            <div>
                <h4 className="h5">Step 3: Build Your Cover Letter</h4>
                <p className="text-muted">
                    Build your outline: Drag <strong className="text-success">Proof</strong> to <strong className="text-primary">Arguments</strong>, then drag <strong className="text-primary">Arguments</strong> to your <strong className="text-dark">Outline</strong>.
                </p>

                {isSubmitting && <div className="spinner-border spinner-border-sm text-primary" role="status"><span className="visually-hidden">Loading...</span></div>}

                <div className="cl-builder-container">
                    {/* --- Column 1: Proof (Pairs) (Unchanged) --- */}
                    <div className="cl-column">
                        <div className="cl-column-header">
                            <h6 className="h5 mb-0 text-success"><i className="bi bi-check-circle-fill me-2"></i> 1. Proof (Evidence)</h6>
                            <small className="text-muted">Your mapped CV items. Drag them to an Argument.</small>
                        </div>
                        <div className="mb-2">
                            <label htmlFor="pair-filter" className="form-label fw-medium small">
                                Filter by Job Requirement:
                            </label>
                            <select
                                id="pair-filter"
                                className="form-select form-select-sm"
                                value={pairFilterId}
                                onChange={(e) => setPairFilterId(e.target.value)}
                            >
                                <option value="all">Show All Mapped Pairs ({mapping.pairs.length})</option>
                                {(job?.features || []).map(f => (
                                    <option key={f.id} value={f.id}>
                                        {f.description.substring(0, 75)}...
                                    </option>
                                ))}
                            </select>
                        </div>
                        <DroppableColumnBody
                            id="pair-pool"
                            items={filteredPairs.map(p => `pool-pair-${p.id}`)}
                        >
                            {filteredPairs.map(pair => (
                                <SortablePairChip key={pair.id} pair={pair} ideaId={null} />
                            ))}
                        </DroppableColumnBody>
                    </div>

                    {/* --- Column 2: Arguments (Ideas) (MODIFIED) --- */}
                    <div className="cl-column">
                         <div className="cl-column-header">
                            <h6 className="h5 mb-0 text-primary"><i className="bi bi-lightbulb-fill me-2"></i> 2. Arguments (Idea Bank)</h6>
                            <small className="text-muted">Your talking points. Drag them to the Outline.</small>
                        </div>
                        {/* --- MODIFIED: Removed 'Why You' button --- */}
                        <div className="d-grid gap-2 mb-3">
                            <button 
                                className="btn btn-primary btn-sm"
                                onClick={handleSmartSuggest}
                                disabled={isSubmitting}
                            >
                                <i className="bi bi-magic me-2"></i>
                                Smart Suggest Arguments
                            </button>
                        </div>
                        
                        <DroppableColumnBody
                            id="idea-pool"
                            items={availableIdeas.map(i => `pool-idea-${i.id}`)}
                        >
                            {availableIdeas.length === 0 && (
                                <p className="small text-muted fst-italic">Suggest arguments or add a 'Why You' argument to start.</p>
                            )}
                            {availableIdeas.map(idea => {
                                const pairsInIdea = idea.mapping_pair_ids
                                    .map(id => pairMap.get(id))
                                    .filter(Boolean);
                                return (
                                    <IdeaCard
                                        key={idea.id}
                                        idea={idea}
                                        pairsInIdea={pairsInIdea}
                                        onDelete={handleDeleteIdea}
                                        onUpdateAnnotation={handleUpdateIdeaAnnotation}
                                        onRemovePair={handleRemovePairFromIdea} // <-- NEW: Pass handler
                                    />
                                );
                            })}
                        </DroppableColumnBody>
                    </div>
                    
                    {/* --- Column 3: Outline (Paragraphs) (Unchanged) --- */}
                    <div className="cl-column">
                         <div className="cl-column-header">
                            <h6 className="h5 mb-0"><i className="bi bi-card-list me-2"></i> 3. Outline (Paragraphs)</h6>
                            <small className="text-muted">Your final cover letter structure.</small>
                        </div>
                         <form onSubmit={handleCreateParagraph} className="d-flex gap-2 mb-3">
                            <input
                                type="text"
                                className="form-control form-control-sm"
                                value={newParaPurpose}
                                onChange={(e) => setNewParaPurpose(e.target.value)}
                                placeholder="Add Custom Paragraph"
                            />
                            <button type="submit" className="btn btn-secondary btn-sm" disabled={!newParaPurpose.trim() || isSubmitting}>
                                +
                            </button>
                        </form>
                        <DroppableColumnBody
                            id="paragraph-outline"
                            items={paragraphs.map(p => `para-${p.id}`)}
                        >
                            {paragraphs.map(para => {
                                const ideasInPara = para.idea_ids
                                    .map(id => ideaMap.get(id))
                                    .filter(Boolean);
                                return (
                                    <ParagraphCard
                                        key={para.id}
                                        paragraph={para}
                                        ideasInParagraph={ideasInPara}
                                        onDelete={handleDeleteParagraph}
                                    />
                                );
                            })}
                        </DroppableColumnBody>
                    </div>
                </div>

                {/* --- Drag Overlay (Unchanged) --- */}
                <DragOverlay>
                    {activeDragItem ? (
                        activeDragItem.type === 'pair' ? (
                            <PairChip pair={activeDragItem} className="pair-chip dragging" />
                        ) : (
                            <IdeaCard 
                                idea={activeDragItem} 
                                pairsInIdea={[]}
                                onDelete={()=>{}} 
                                onUpdateAnnotation={()=>{}}
                            />
                        )
                    ) : null}
                </DragOverlay>

                {/* --- Navigation (Unchanged) --- */}
                <button
                    className="btn btn-info mt-3"
                    onClick={handleGeneratePrompt}
                    disabled={isLoadingPrompt || coverLetter.paragraphs.length === 0}
                >
                    {isLoadingPrompt ? "Generating..." : "Generate Cover Letter Prompt"}
                </button>

                <PromptModal
                    isOpen={isModalOpen}
                    jsonString={clPromptJson}
                    onClose={() => setIsModalOpen(false)}
                />

                <div className="d-flex justify-content-between mt-4">
                    <button className="btn btn-secondary" onClick={onPrev}>
                        &lt; Back: Generate CV
                    </button>
                    <button className="btn btn-primary" onClick={onNext}>
                        Next: Submit & Track &gt;
                    </button>
                </div>
            </div>

            {/* --- NEW: Company Research Modal --- */}
            {showResearchModal && <div className="modal-backdrop fade show"></div>}
            <div 
                className={`modal fade ${showResearchModal ? 'show' : ''}`} 
                style={{ display: showResearchModal ? 'block' : 'none' }}
                tabIndex="-1"
            >
                <div className="modal-dialog modal-dialog-centered">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h5 className="modal-title">Company Research (Optional)</h5>
                            <button 
                                type="button" 
                                className="btn-close" 
                                onClick={() => setShowResearchModal(false)}
                            ></button>
                        </div>
                        <div className="modal-body">
                            <p>
                                Add any research about the company (e.g., mission statement, ethics, recent news) 
                                to create a "Why this company" argument.
                            </p>
                            <div className="mb-3">
                                <label htmlFor="company-research-textarea" className="form-label">
                                    How do you align with their mission?
                                </label>
                                <textarea
                                    id="company-research-textarea"
                                    className="form-control"
                                    rows="4"
                                    placeholder="e.g., 'Their focus on renewable energy aligns with my passion for sustainable technology...'"
                                    value={companyResearch}
                                    onChange={(e) => setCompanyResearch(e.target.value)}
                                ></textarea>
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button 
                                type="button" 
                                className="btn btn-secondary" 
                                onClick={() => {
                                    setCompanyResearch("");
                                    handleSubmitSmartArguments();
                                }}
                                disabled={isSubmitting}
                            >
                                Skip
                            </button>
                            <button 
                                type="button" 
                                className="btn btn-primary" 
                                onClick={handleSubmitSmartArguments}
                                disabled={isSubmitting}
                            >
                                {isSubmitting ? (
                                    <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                                ) : (
                                    'Create Arguments'
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
            {/* --- END: Company Research Modal --- */}

        </DndContext>
    );
};

export default Step3_BuildCoverLetter;