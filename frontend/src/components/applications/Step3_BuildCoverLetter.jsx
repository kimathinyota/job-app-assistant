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
import CL_SuggestionModal from './CL_SuggestionModal.jsx';
import CL_EvidenceGroup from './CL_EvidenceGroup.jsx';
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
    position: relative; /* --- ADDED FOR ABSOLUTE BUTTON --- */
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

// --- Helper Functions ---
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

// Col 1: Pair Chip (Exported)
export const PairChip = forwardRef(({ pair, onRemove, ...props }, ref) => (
    <div ref={ref} {...props} className="pair-chip">
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
        <i 
            className={`${getItemIcon(pair.context_item_type)} pair-chip-icon`} 
            style={{ right: onRemove ? '30px' : '0.75rem' }}
        ></i>
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


// Col 2: Idea Card (Refactored to use CL_EvidenceGroup AND new button location)
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

    // --- NEW: Group pairs by evidence item ---
    const groupedPairs = useMemo(() => {
        const groups = new Map();
        for (const pair of pairsInIdea) {
            const key = pair.context_item_id;
            const title = pair.context_item_text || "Unknown Evidence";
            if (!groups.has(key)) {
                groups.set(key, { title, pairs: [] });
            }
            groups.get(key).pairs.push(pair);
        }
        return Array.from(groups.values());
    }, [pairsInIdea]);

    return (
        <div ref={setNodeRef} style={style} className={`idea-card ${isDragging ? 'dragging' : ''}`}>
            {/* --- NEW DELETE BUTTON (Col 2) --- */}
            <button
                type="button"
                className="btn-close btn-sm"
                onClick={(e) => {
                    e.stopPropagation();
                    onDelete(idea.id);
                }}
                title="Delete Idea"
                style={{
                    position: 'absolute',
                    top: '0.5rem',
                    right: '0.75rem',
                    zIndex: 2,
                }}
            />
            {/* --- END NEW BUTTON --- */}
            
            <div className="idea-card-header">
                <h6 
                    className="h6 mb-0" 
                    {...attributes} 
                    {...listeners} 
                    style={{ 
                        cursor: 'grab', 
                        flexGrow: 1,
                        paddingRight: '1.5rem' // --- ADDED PADDING ---
                    }}
                >
                    {idea.title}
                </h6>
                {/* --- Button was removed from here --- */}
            </div>
            
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

                {groupedPairs.map(group => (
                    <CL_EvidenceGroup
                        key={group.title}
                        cvItemText={group.title}
                        pairs={group.pairs}
                        ideaId={idea.id}
                        onRemovePair={onRemovePair} // Pass handler down
                    />
                ))}
                
                {pairsInIdea.length === 0 && !annotation && (
                     <p className="small text-muted text-center mb-0">
                        Drag proof here
                    </p>
                )}
            </div>
        </div>
    );
};


// Col 3: Paragraph Card (MODIFIED to pass down onUnlinkIdea)
const ParagraphCard = ({ paragraph, ideasInParagraph, onDelete, onUnlinkIdea }) => {
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
                            onUnlink={() => onUnlinkIdea(paragraph.id, idea.id)} // --- PASS HANDLER ---
                        />
                    ))}
                </div>
            </SortableContext>
        </div>
    );
};

// --- Sortable Wrappers ---

// Sortable Pair (Exported)
export const SortablePairChip = ({ pair, ideaId, onRemove }) => {
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
            onRemove={onRemove ? () => onRemove(ideaId, pair.id) : null}
            {...attributes}
            {...listeners}
        />
    );
};

// Sortable Idea (MODIFIED to include onUnlink in new location)
const SortableIdeaCard = ({ idea, paragraphId, onUnlink }) => {
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
        <div ref={setNodeRef} style={style} {...attributes} className="idea-card">
            {/* --- NEW UNLINK BUTTON (Col 3) --- */}
            <button
                type="button"
                className="btn-close btn-sm"
                onClick={(e) => {
                    e.stopPropagation();
                    onUnlink();
                }}
                title="Unlink from paragraph"
                style={{
                    position: 'absolute',
                    top: '0.5rem',
                    right: '0.75rem',
                    zIndex: 2,
                }}
            />
            {/* --- END NEW BUTTON --- */}
            
            <div className="idea-card-header">
                {/* Drag Handle */}
                <h6 
                    className="h6 mb-0 small" 
                    {...listeners} 
                    style={{ 
                        cursor: 'grab', 
                        flexGrow: 1,
                        paddingRight: '1.5rem' // --- ADDED PADDING ---
                    }}
                >
                    {idea.title}
                </h6>
                {/* --- Button was removed from here --- */}
            </div>
             {idea.annotation && (
                <div className="idea-card-body py-2">
                    <p className="small text-muted fst-italic mb-0" style={{whiteSpace: 'pre-wrap'}}>
                        {idea.annotation.substring(0, 50)}{idea.annotation.length > 50 ? "..." : ""}
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
    const [isPromptModalOpen, setIsPromptModalOpen] = useState(false);
    const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);
    
    // --- NEW: State for the Suggestion Modal ---
    const [isSuggestionModalOpen, setIsSuggestionModalOpen] = useState(false);

    // --- Data Loading & Scaffolding (Unchanged, but added reload) ---
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
             // Show subtle loading
             const clData = (await fetchCoverLetterDetails(coverLetter.id)).data;
             setCoverLetter(clData);
        } catch (err) {
             console.error("Failed to reload CL:", err);
             setError("Failed to refresh data.");
        }
    }

    useEffect(() => {
        loadCoverLetter();
    }, [application.id, application.cover_letter_id]);

    // --- Memos for Data Manipulation (Refactored for Grouping) ---
    const {
        pairMap,
        ideaMap,
        // --- NEW: Grouped pairs for Col 1 ---
        groupedFilteredPairs, 
        // --- END NEW ---
        availableIdeas,
        paragraphs,
        itemsById,
    } = useMemo(() => {
        const pMap = new Map((mapping?.pairs || []).map(p => [p.id, p]));
        const iMap = new Map(coverLetter ? coverLetter.ideas.map(i => [i.id, i]) : []);
        
        // 1. Get pairs matching the filter
        const fPairs = (mapping?.pairs || []).filter(p =>
            pairFilterId === 'all' || p.feature_id === pairFilterId
        );

        // --- NEW: Group the filtered pairs ---
        const gPairs = new Map();
        for (const pair of fPairs) {
            const key = pair.context_item_id;
            const title = pair.context_item_text || "Unknown Evidence";
            if (!gPairs.has(key)) {
                gPairs.set(key, { title, pairs: [] });
            }
            gPairs.get(key).pairs.push(pair);
        }
        // --- END NEW ---

        const pairedIdeaIds = new Set();
        (coverLetter?.paragraphs || []).forEach(para => {
            para.idea_ids.forEach(id => pairedIdeaIds.add(id));
        });
        const avIdeas = (coverLetter?.ideas || []).filter(i => !pairedIdeaIds.has(i.id));
        
        const paras = (coverLetter?.paragraphs || []).sort((a, b) => a.order - b.order);

        // This lookup map is still needed for the DragOverlay
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
            groupedFilteredPairs: Array.from(gPairs.values()), // Convert map to array for rendering
            availableIdeas: avIdeas,
            paragraphs: paras,
            itemsById: allItems
        };
    }, [mapping?.pairs, coverLetter, job?.features, pairFilterId]);

    // --- Form Handlers ---
    
    // --- NEW: Handler for the suggestion modal ---
    const handleSuggestionsAccepted = async () => {
        // This is called by the modal when an action is successful
        await reloadCoverLetter();
        // The modal will stay open and re-calculate its suggestions
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

    // --- NEW HANDLER FOR UNLINKING ---
    const handleUnlinkIdea = async (paragraphId, ideaId) => {
        if (isSubmitting) return;
        setIsSubmitting(true);
        try {
            const sourcePara = paragraphs.find(p => p.id === paragraphId);
            if (!sourcePara) throw new Error("Paragraph not found");

            const newIdeaIds = sourcePara.idea_ids.filter(id => id !== ideaId);
            
            await updateCoverLetterParagraph(
                coverLetter.id, 
                paragraphId, 
                { idea_ids: newIdeaIds }
            );
            
            // Reload all data
            await reloadCoverLetter();
        } catch (err) {
            console.error("Failed to unlink idea:", err);
            alert("Failed to unlink idea.");
            await reloadCoverLetter(); // Reload even on failure to reset state
        } finally {
            setIsSubmitting(false);
        }
    };
    // --- END NEW HANDLER ---

    // --- (Unchanged Handlers) ---
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
            // --- Dragging a PAIR ---
            if (activeData?.type === 'pair') {
                const pairId = activeData.pair.id;
                const sourceIdeaId = activeData.sourceIdeaId;
                
                // Dropping PAIR into an IDEA
                if (overContainerId.startsWith('idea-card-body-')) {
                    const targetIdeaId = overContainerId.split('idea-card-body-')[1];
                    
                    // 1. Remove from source idea (if it was in one)
                    if (sourceIdeaId && sourceIdeaId !== targetIdeaId) {
                        const sourceIdea = ideaMap.get(sourceIdeaId);
                        const newPairIds = sourceIdea.mapping_pair_ids.filter(id => id !== pairId);
                        await updateCoverLetterIdea(coverLetter.id, sourceIdeaId, { mapping_pair_ids: newPairIds });
                    }
                    
                    // 2. Add to target idea
                    const targetIdea = ideaMap.get(targetIdeaId);
                    if (targetIdea && !targetIdea.mapping_pair_ids.includes(pairId)) {
                        const newPairIds = [...targetIdea.mapping_pair_ids, pairId];
                        await updateCoverLetterIdea(coverLetter.id, targetIdeaId, { mapping_pair_ids: newPairIds });
                    }
                }
                // Dropping PAIR back into the POOL
                else if (overContainerId === 'pair-pool' && sourceIdeaId) {
                    // Remove from source idea
                    const sourceIdea = ideaMap.get(sourceIdeaId);
                    const newPairIds = sourceIdea.mapping_pair_ids.filter(id => id !== pairId);
                    await updateCoverLetterIdea(coverLetter.id, sourceIdeaId, { mapping_pair_ids: newPairIds });
                }
            }
            
            // --- Dragging an IDEA ---
            if (activeData?.type === 'idea') {
                const ideaId = activeData.idea.id;
                const sourceParagraphId = activeData.sourceParagraphId;
                
                // Dropping IDEA into a PARAGRAPH
                if (overContainerId.startsWith('para-card-body-')) {
                    const targetParagraphId = overContainerId.split('para-card-body-')[1];
                    
                    // 1. Remove from source paragraph (if it was in one)
                    if (sourceParagraphId && sourceParagraphId !== targetParagraphId) {
                        const sourcePara = paragraphs.find(p => p.id === sourceParagraphId);
                        const newIdeaIds = sourcePara.idea_ids.filter(id => id !== ideaId);
                        await updateCoverLetterParagraph(coverLetter.id, sourceParagraphId, { idea_ids: newIdeaIds });
                    }
                    
                    // 2. Add to target paragraph
                    const targetPara = paragraphs.find(p => p.id === targetParagraphId);
                    if (targetPara && !targetPara.idea_ids.includes(ideaId)) {
                        const newIdeaIds = [...targetPara.idea_ids, ideaId];
                        await updateCoverLetterParagraph(coverLetter.id, targetParagraphId, { idea_ids: newIdeaIds });
                    }
                }
                // Dropping IDEA back into the POOL
                else if (overContainerId === 'idea-pool' && sourceParagraphId) {
                    // Remove from source paragraph
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
            setIsPromptModalOpen(true);
        } catch (err) { alert("Failed to generate prompt."); }
        finally { setIsLoadingPrompt(false); }
    };
    
    // --- Render ---
    if (isLoading && !coverLetter) return <div>Loading Cover Letter builder...</div>;
    if (error) return <div className="alert alert-danger">{error}</div>;
    if (!coverLetter || !mapping || !job) return <div className="alert alert-info">Initializing...</div>;

    const activeDragItem = activeDragId ? itemsById[activeDragId] : null;
    
    // We need a flat list of all sortable pair IDs for the Column 1 Droppable
    const col1SortableIds = groupedFilteredPairs.flatMap(
        group => group.pairs.map(p => `pool-pair-${p.id}`)
    );

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
                    {/* --- Column 1: Proof (Pairs) (Refactored) --- */}
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
                            items={col1SortableIds} // Pass flat list of IDs
                        >
                            {/* Render the new grouped component */}
                            {groupedFilteredPairs.map(group => (
                                <CL_EvidenceGroup
                                    key={group.title}
                                    cvItemText={group.title}
                                    pairs={group.pairs}
                                    ideaId={null} // Not in an idea
                                    onRemovePair={null} // Not removable
                                />
                            ))}
                        </DroppableColumnBody>
                    </div>

                    {/* --- Column 2: Arguments (Ideas) (Refactored) --- */}
                    <div className="cl-column">
                         <div className="cl-column-header">
                            <h6 className="h5 mb-0 text-primary"><i className="bi bi-lightbulb-fill me-2"></i> 2. Arguments (Idea Bank)</h6>
                            <small className="text-muted">Your talking points. Drag them to the Outline.</small>
                        </div>
                        <div className="d-grid gap-2 mb-3">
                            {/* --- MODIFIED: Button now opens new modal --- */}
                            <button 
                                className="btn btn-primary btn-sm"
                                onClick={() => setIsSuggestionModalOpen(true)}
                                disabled={isSubmitting}
                            >
                                <i className="bi bi-magic me-2"></i>
                                Open CL Assistant
                            </button>
                        </div>
                        
                        <DroppableColumnBody
                            id="idea-pool"
                            items={availableIdeas.map(i => `pool-idea-${i.id}`)}
                        >
                            {availableIdeas.length === 0 && (
                                <p className="small text-muted fst-italic">No ideas in the bank. Use the Assistant or drag ideas from the outline.</p>

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
                                        onRemovePair={handleRemovePairFromIdea}
                                    />
                                );
                            })}
                        </DroppableColumnBody>
                    </div>
                    
                    {/* --- Column 3: Outline (Paragraphs) (MODIFIED) --- */}
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
                                        onUnlinkIdea={handleUnlinkIdea} // --- PASS HANDLER DOWN ---
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
                            // Pass all required props to IdeaCard
                            <IdeaCard 
                                idea={activeDragItem} 
                                pairsInIdea={[]} // Not needed for drag overlay
                                onDelete={()=>{}} 
                                onUpdateAnnotation={()=>{}}
                                onRemovePair={()=>{}}
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
                    isOpen={isPromptModalOpen}
                    jsonString={clPromptJson}
                    onClose={() => setIsPromptModalOpen(false)}
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

            {/* --- NEW: Render the Suggestion Modal --- */}
            {isSuggestionModalOpen && (
                <CL_SuggestionModal
                    isOpen={isSuggestionModalOpen}
                    onClose={() => setIsSuggestionModalOpen(false)}
                    coverLetter={coverLetter}
                    mapping={mapping}
                    job={job}
                    onSuggestionsAccepted={handleSuggestionsAccepted}
                />
            )}
        </DndContext>
    );
};

export default Step3_BuildCoverLetter;