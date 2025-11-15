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
import IntelligentTextArea from './IntelligentTextArea.jsx';
import CVItemPreviewModal from './CVItemPreviewModal.jsx';
import IntelligentTextAreaModal from './IntelligentTextAreaModal.jsx';
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
import { 
    Wand2, FileText, GripVertical, Trash2, Plus, ArrowRight, ArrowLeft, 
    Briefcase, GraduationCap, FolderGit2, Heart, Lightbulb, Layers, 
    LayoutList, CheckCircle2, AlertCircle, Loader2, Sparkles, BrainCircuit,
    Maximize2, Minimize2, X, MessageSquare, Filter, Search, // <-- ADDED SEARCH
    PanelLeftClose, PanelRightClose, PanelLeftOpen, PanelRightOpen
} from 'lucide-react';

// --- MODERNIZED STYLES (Fixed layout + Scroll) ---
const DND_STYLES = `
.step3-layout {
    display: flex;
    flex-direction: column;
    /* This height applies to ALL screen sizes, fixing the indefinite scroll */
    height: calc(100vh - 140px);
    min-height: 600px;
}

.cl-builder-container {
    flex: 1;
    display: flex;
    flex-direction: row;
    gap: 1rem;
    width: 100%;
    min-height: 0; 
    padding-bottom: 0.5rem;
}

.cl-column {
    display: flex;
    flex-direction: column;
    background-color: #f8fafc;
    border-radius: 1rem;
    padding: 1rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    transition: flex 0.3s ease, width 0.3s ease;
    height: 100%; 
    overflow: hidden; 
}

.cl-column-header {
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center; 
    flex-wrap: nowrap;
    white-space: nowrap;
    flex-shrink: 0; 
}

.cl-column-body {
    flex-grow: 1;
    overflow-y: auto; 
    padding-right: 4px;
}

/* Custom Scrollbar */
.cl-column-body::-webkit-scrollbar { width: 6px; }
.cl-column-body::-webkit-scrollbar-track { background: transparent; }
.cl-column-body::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 10px; }
.cl-column-body::-webkit-scrollbar-thumb:hover { background-color: #94a3b8; }

.cl-column-body.over {
    background-color: rgba(59, 130, 246, 0.05);
    border-radius: 0.5rem;
    outline: 2px dashed #3b82f6;
    outline-offset: -4px;
}

/* Collapsed Column Styles */
.cl-column-collapsed {
    background-color: #e2e8f0;
    cursor: pointer;
    padding: 1rem 0;
    align-items: center;
    border: 1px dashed #cbd5e1;
    width: 50px !important;
    min-width: 50px !important;
    flex: 0 0 50px !important;
}
.cl-column-collapsed:hover {
    background-color: #cbd5e1;
}
.vertical-text {
    writing-mode: vertical-rl;
    text-orientation: mixed;
    transform: rotate(180deg);
    white-space: nowrap;
    font-weight: bold;
    color: #64748b;
    margin-top: 2rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.8rem;
}

/* Cards */
.pair-chip {
    cursor: grab;
    touch-action: none;
    background-color: white;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #10b981;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    position: relative;
    transition: all 0.2s ease;
}
.pair-chip:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
}
.pair-chip.dragging {
    opacity: 0.9;
    transform: rotate(2deg);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    z-index: 999;
    cursor: grabbing;
}

.idea-card {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #8b5cf6;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    touch-action: none;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
}
.idea-card-header {
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #f1f5f9;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    background-color: white;
    border-top-right-radius: 0.75rem;
}
.idea-card-body { padding: 1rem; min-height: 60px; }
.idea-card-body.over { background-color: #f0fdf4; }
.idea-card.dragging { opacity: 0.95; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); z-index: 999; cursor: grabbing; }

.paragraph-card {
    background-color: white;
    border: 1px solid #e2e8f0;
    border-top: 4px solid #3b82f6;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    touch-action: none;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.paragraph-card-header {
    padding: 1rem;
    border-bottom: 1px solid #f1f5f9;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #f8fafc;
    border-top-left-radius: 0.75rem;
    border-top-right-radius: 0.75rem;
}
.paragraph-card-body { padding: 1rem; min-height: 100px; background-color: #f8fafc; border-bottom-left-radius: 0.75rem; border-bottom-right-radius: 0.75rem; }
.paragraph-card-body.over { background-color: #eff6ff; }

.hover-opacity-100:hover { opacity: 1 !important; }
.group:hover .group-hover-opacity-100 { opacity: 1 !important; }

/* ====================================================================
   --- (THIS IS THE MODIFIED MOBILE FIX) ---
====================================================================
*/
@media (max-width: 991.98px) {
    /*
     * REMOVED height: auto from .step3-layout.
     * It will now use the desktop height: calc(100vh - 140px),
     * which constrains the page and fixes the footer issue.
    */

    .cl-builder-container {
        /* Allow horizontal scrolling */
        overflow-x: auto;
        /* Stop the container itself from scrolling vertically */
        overflow-y: hidden; 
        padding-bottom: 1rem;
        /* Ensure the container doesn't shrink its children */
        flex-shrink: 0;
    }

    .cl-column {
        /* * This is the magic. 
         * Give each column a fixed, usable width.
         * 340px is a good width for most phones.
        */
        width: 340px; 
        min-width: 340px;
        /* Stop the column from shrinking */
        flex-shrink: 0;
        /* This 100% height now refers to the fixed-height cl-builder-container */
        height: 100%;
    }

    /* Your existing collapsed style will work perfectly with this */
    .cl-column-collapsed {
        width: 50px !important;
        min-width: 50px !important;
    }
}
`;

// ... (GetItemIcon Helper - Unchanged) ...
const GetItemIcon = ({ type, size=16, className }) => {
    switch (type) {
        case 'experiences': return <Briefcase size={size} className={className} />;
        case 'projects': return <FolderGit2 size={size} className={className} />;
        case 'education': return <GraduationCap size={size} className={className} />;
        case 'hobbies': return <Heart size={size} className={className} />;
        default: return <FileText size={size} className={className} />;
    }
};

// --- DRAGGABLE COMPONENTS ---
// ... (PairChip component - Unchanged) ...
export const PairChip = forwardRef(({ pair, onRemove, ...props }, ref) => (
    <div ref={ref} {...props} className="pair-chip group">
        {onRemove && (
            <button
                type="button"
                className="btn btn-link p-0 text-muted position-absolute top-0 end-0 mt-1 me-1 opacity-0 group-hover-opacity-100 transition-all hover-text-danger"
                onPointerDown={(e) => {
                    e.stopPropagation(); 
                }}
                onClick={(e) => { 
                    e.stopPropagation(); 
                    onRemove(); 
                }}
                title="Remove pair"
            >
                <X size={14} />
            </button>
        )}
        <div className="d-flex align-items-start gap-2">
            <div className="mt-1 text-success opacity-75 flex-shrink-0">
                <GetItemIcon type={pair.context_item_type} />
            </div>
            <div className="flex-grow-1 pe-2" style={{minWidth: 0}}>
                <p className="mb-1 small fw-bold text-dark" style={{lineHeight: '1.3', wordBreak: 'break-word', overflowWrap: 'anywhere'}}>
                    {pair.feature_text}
                </p>
                <p className="mb-0 text-muted small" style={{wordBreak: 'break-word', overflowWrap: 'anywhere'}}>
                    {pair.context_item_text}
                </p>
                {pair.annotation && (
                    <div className="d-flex align-items-start gap-1 mt-2 pt-2 border-top border-light text-primary">
                        <MessageSquare size={10} className="mt-1 flex-shrink-0"/>
                        <span className="small fst-italic text-wrap" style={{fontSize: '0.75rem', lineHeight: '1.3'}}>
                            {pair.annotation}
                        </span>
                    </div>
                )}
            </div>
        </div>
    </div>
));
// ... (IdeaCard component - Unchanged) ...
const IdeaCard = ({ 
    idea, pairsInIdea, onDelete, onUpdateAnnotation, onRemovePair, 
    fullCV, onShowPreview, onMaximize 
}) => {
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ 
        id: `pool-idea-${idea.id}`, data: { type: 'idea', idea } 
    });

    const style = { transform: CSS.Transform.toString(transform), transition };
    const { setNodeRef: setDroppableNodeRef, isOver } = useDroppable({
        id: `idea-card-body-${idea.id}`, data: { type: 'idea-body', idea: idea }
    });
    
    const groupedPairs = useMemo(() => {
        const groups = new Map();
        for (const pair of pairsInIdea) {
            const key = pair.context_item_id;
            const title = pair.context_item_text || "Unknown Evidence";
            if (!groups.has(key)) groups.set(key, { title, pairs: [] });
            groups.get(key).pairs.push(pair);
        }
        return Array.from(groups.values());
    }, [pairsInIdea]);

    return (
        <div ref={setNodeRef} style={style} className={`idea-card ${isDragging ? 'dragging' : ''}`}>
            <div className="idea-card-header">
                <div className="d-flex align-items-start gap-2 flex-grow-1">
                    <GripVertical size={16} className="text-muted cursor-grab mt-1 flex-shrink-0" {...attributes} {...listeners}/>
                    <h6 className="fw-bold text-dark mb-0 small" style={{lineHeight: '1.4', wordBreak: 'break-word'}}>
                        {idea.title}
                    </h6>
                </div>
                <button className="btn btn-link p-0 text-muted hover-text-danger ms-2 flex-shrink-0" onClick={(e) => { e.stopPropagation(); onDelete(idea.id); }}>
                    <Trash2 size={14}/>
                </button>
            </div>
            
            <div ref={setDroppableNodeRef} className={`idea-card-body ${isOver ? 'over' : ''}`}>
                <IntelligentTextArea
                    initialValue={idea.annotation || ''}
                    onSave={(newAnnotation) => onUpdateAnnotation(idea.id, newAnnotation)}
                    cv={fullCV}
                    onShowPreview={onShowPreview}
                    onMaximize={onMaximize}
                />
                <div className="mt-3 d-flex flex-column gap-2">
                    {groupedPairs.map(group => (
                        <CL_EvidenceGroup
                            key={group.title}
                            cvItemText={group.title}
                            pairs={group.pairs}
                            ideaId={idea.id}
                            onRemovePair={onRemovePair} 
                        />
                    ))}
                </div>
                
                {pairsInIdea.length === 0 && !idea.annotation && (
                     <div className="text-center py-3 border-2 border-dashed rounded bg-light">
                        <p className="small text-muted mb-0">Drag proof here</p>
                    </div>
                )}
            </div>
        </div>
    );
};
// ... (ParagraphCard component - Unchanged) ...
const ParagraphCard = ({ paragraph, ideasInParagraph, onDelete, onUnlinkIdea }) => {
    const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ 
        id: `para-${paragraph.id}`, data: { type: 'paragraph', paragraph } 
    });
    const { setNodeRef: setDroppableNodeRef, isOver } = useDroppable({
        id: `para-card-body-${paragraph.id}`, data: { type: 'paragraph-body', paragraph }
    });
    const style = { transform: CSS.Transform.toString(transform), transition };

    return (
        <div ref={setNodeRef} style={style} className="paragraph-card">
            <div className="paragraph-card-header">
                <div className="d-flex align-items-center gap-2 flex-grow-1">
                    <GripVertical size={16} className="text-muted cursor-grab" {...attributes} {...listeners}/>
                    <h6 className="fw-bold text-dark mb-0 small text-uppercase tracking-wide">{paragraph.purpose}</h6>
                </div>
                <button className="btn btn-link p-0 text-muted hover-text-danger" onClick={(e) => { e.stopPropagation(); onDelete(paragraph.id); }}>
                    <Trash2 size={14}/>
                </button>
            </div>
            <SortableContext
                id={`para-card-body-${paragraph.id}`}
                items={ideasInParagraph.map(i => `para-${paragraph.id}-idea-${i.id}`)}
                strategy={verticalListSortingStrategy}
            >
                <div ref={setDroppableNodeRef} className={`paragraph-card-body ${isOver ? 'over' : ''}`}>
                    {ideasInParagraph.length === 0 && (
                        <div className="text-center py-4 border-2 border-dashed rounded bg-white bg-opacity-50">
                            <ArrowLeft size={16} className="text-muted mb-1"/>
                            <p className="small text-muted fst-italic mb-0">Drag arguments here</p>
                        </div>
                    )}
                    {ideasInParagraph.map(idea => (
                        <SortableIdeaCard
                            key={idea.id}
                            idea={idea}
                            paragraphId={paragraph.id}
                            onUnlink={() => onUnlinkIdea(paragraph.id, idea.id)}
                        />
                    ))}
                </div>
            </SortableContext>
        </div>
    );
};

// ... (Sortable Wrappers: SortablePairChip, SortableIdeaCard, DroppableColumnBody - UNCHANGED) ...
export const SortablePairChip = ({ pair, ideaId, onRemove }) => {
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ 
        id: ideaId ? `idea-${ideaId}-pair-${pair.id}` : `pool-pair-${pair.id}`,
        data: { type: 'pair', pair, sourceIdeaId: ideaId } 
    });
    const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1 };
    return ( <PairChip ref={setNodeRef} style={style} pair={pair} onRemove={onRemove} {...attributes} {...listeners} /> );
};

const SortableIdeaCard = ({ idea, paragraphId, onUnlink }) => {
     const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ 
        id: `para-${paragraphId}-idea-${idea.id}`, 
        data: { type: 'idea', idea, sourceParagraphId: paragraphId } 
    });
    const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1 };
    return (
        <div ref={setNodeRef} style={style} {...attributes} className="idea-card">
            <div className="idea-card-header py-2">
                <div className="d-flex align-items-start gap-2 flex-grow-1" style={{minWidth:0}}>
                    <GripVertical size={14} className="text-muted mt-1 flex-shrink-0" {...listeners}/>
                    <h6 className="fw-bold text-dark mb-0 small" style={{lineHeight: '1.4', wordBreak: 'break-word', overflowWrap: 'anywhere'}}>{idea.title}</h6>
                </div>
                <button className="btn btn-link p-0 text-muted hover-text-danger ms-2 flex-shrink-0" onClick={(e) => { e.stopPropagation(); onUnlink(); }}>
                    <X size={14}/>
                </button>
            </div>
             {idea.annotation && (
                <div className="px-3 py-2 bg-light border-top">
                    <p className="small text-muted fst-italic mb-0 text-wrap">
                        {idea.annotation.substring(0, 100)}{idea.annotation.length > 100 ? "..." : ""}
                    </p>
                </div>
            )}
        </div>
    );
};

const DroppableColumnBody = ({ id, items, children }) => {
    const { setNodeRef, isOver } = useDroppable({ id: id });
    return (
        <SortableContext id={id} items={items} strategy={verticalListSortingStrategy}>
            <div ref={setNodeRef} className={`cl-column-body ${isOver ? 'over' : ''}`}>
                {children}
            </div>
        </SortableContext>
    );
};

// --- Main Component ---
const Step3_BuildCoverLetter = ({ 
    application, mapping, onPrev, onNext, job, onCoverLetterCreated, fullCV 
}) => {
    const [coverLetter, setCoverLetter] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const [activeDragId, setActiveDragId] = useState(null);
    const [pairFilterId, setPairFilterId] = useState('all');
    const [pairSearchTerm, setPairSearchTerm] = useState(''); // <-- NEW STATE
    
    const [newIdeaTitle, setNewIdeaTitle] = useState('');
    const [newParaPurpose, setNewParaPurpose] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [clPromptJson, setClPromptJson] = useState('');
    const [isPromptModalOpen, setIsPromptModalOpen] = useState(false);
    const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);
    const [isSuggestionModalOpen, setIsSuggestionModalOpen] = useState(false);

    const [previewItem, setPreviewItem] = useState(null);
    const [previewItemType, setPreviewItemType] = useState(null);
    const [maximizedIdea, setMaximizedIdea] = useState(null);

    const [isLeftColOpen, setIsLeftColOpen] = useState(true);
    const [isRightColOpen, setIsRightColOpen] = useState(true);

    // ... (getColumnStyle helper - Unchanged) ...
    const getColumnStyle = (colType) => {
        if (colType === 'left' && !isLeftColOpen) return {};
        if (colType === 'right' && !isRightColOpen) return {};
        return { flex: 1 };
    };

    // ... (cvLookups - Unchanged) ...
    const cvLookups = useMemo(() => {
        const map = new Map();
        if (!fullCV) return map;
        const add = (arr, type, nameFn) => (arr || []).forEach(item => 
            map.set(item.id, { name: nameFn(item), type })
        );
        add(fullCV.experiences, 'experiences', i => `${i.title} @ ${i.company}`);
        add(fullCV.projects, 'projects', i => i.title);
        add(fullCV.education, 'education', i => `${i.degree} @ ${i.institution}`);
        add(fullCV.hobbies, 'hobbies', i => i.name);
        return map;
    }, [fullCV]);

    // ... (Data Loading functions - Unchanged) ...
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
                if (onCoverLetterCreated) { await onCoverLetterCreated(clData.id); }
                await addCoverLetterParagraph(clData.id, [], "Introduction");
                await addCoverLetterParagraph(clData.id, [], "Body 1 (Why Me)");
                await addCoverLetterParagraph(clData.id, [], "Body 2 (Why You)");
                await addCoverLetterParagraph(clData.id, [], "Conclusion");
                clData = (await fetchCoverLetterDetails(clData.id)).data;
            }
            setCoverLetter(clData);
        } catch (err) {
            setError("Failed to load or create cover letter data.");
        } finally {
            setIsLoading(false);
        }
    };

    const reloadCoverLetter = async () => {
        if (!coverLetter?.id) { await loadCoverLetter(); return; }
        try {
             const clData = (await fetchCoverLetterDetails(coverLetter.id)).data;
             setCoverLetter(clData);
        } catch (err) { setError("Failed to refresh data."); }
    }

    useEffect(() => { loadCoverLetter(); }, [application.id, application.cover_letter_id]);
    
    // --- (MODIFIED) Memos for Data ---
    const { pairMap, ideaMap, groupedFilteredPairs, availableIdeas, paragraphs, itemsById } = useMemo(() => {
        const pMap = new Map((mapping?.pairs || []).map(p => {
            const lookup = cvLookups.get(p.context_item_id);
            const realText = lookup?.name || p.context_item_text || "Unknown Evidence";
            const realType = lookup?.type || p.context_item_type || "other";
            return [p.id, { ...p, context_item_text: realText, context_item_type: realType }];
        }));

        const iMap = new Map(coverLetter ? coverLetter.ideas.map(i => [i.id, i]) : []);
        
        // --- (MODIFIED) Pair filtering logic ---
        const lowerSearchTerm = pairSearchTerm.toLowerCase();
        const fPairs = Array.from(pMap.values()).filter(p => {
            // 1. Filter by dropdown
            const filterMatch = pairFilterId === 'all' || p.feature_id === pairFilterId;
            if (!filterMatch) return false;

            // 2. Filter by search term
            const searchMatch = !lowerSearchTerm || 
                p.feature_text.toLowerCase().includes(lowerSearchTerm) ||
                p.context_item_text.toLowerCase().includes(lowerSearchTerm) ||
                (p.annotation && p.annotation.toLowerCase().includes(lowerSearchTerm));
            
            return searchMatch;
        });
        // --- (End of modification) ---

        const gPairs = new Map();
        for (const pair of fPairs) {
            const key = pair.context_item_id;
            const title = pair.context_item_text;
            if (!gPairs.has(key)) gPairs.set(key, { title, pairs: [] });
            gPairs.get(key).pairs.push(pair);
        }

        const pairedIdeaIds = new Set();
        (coverLetter?.paragraphs || []).forEach(para => para.idea_ids.forEach(id => pairedIdeaIds.add(id)));
        const avIdeas = (coverLetter?.ideas || []).filter(i => !pairedIdeaIds.has(i.id));
        const paras = (coverLetter?.paragraphs || []).sort((a, b) => a.order - b.order);

        const allItems = {};
        Array.from(pMap.values()).forEach(p => {
             allItems[`pool-pair-${p.id}`] = { ...p, type: 'pair' };
             (coverLetter?.ideas || []).forEach(i => {
                if (i.mapping_pair_ids.includes(p.id)) allItems[`idea-${i.id}-pair-${p.id}`] = { ...p, type: 'pair' };
             });
        });
        (coverLetter?.ideas || []).forEach(i => {
            allItems[`pool-idea-${i.id}`] = { ...i, type: 'idea' };
            (coverLetter?.paragraphs || []).forEach(p => {
                if (p.idea_ids.includes(i.id)) allItems[`para-${p.id}-idea-${i.id}`] = { ...i, type: 'idea' };
            });
        });

        return { pairMap: pMap, ideaMap: iMap, groupedFilteredPairs: Array.from(gPairs.values()), availableIdeas: avIdeas, paragraphs: paras, itemsById: allItems };
    }, [mapping?.pairs, coverLetter, job?.features, pairFilterId, cvLookups, pairSearchTerm]); // <-- ADDED pairSearchTerm

    // ... (All handlers are UNCHANGED) ...
    const handleSuggestionsAccepted = async () => await reloadCoverLetter();
    const handleRemovePairFromIdea = async (ideaId, pairId) => {
        const idea = ideaMap.get(ideaId); if (!idea) return;
        setIsSubmitting(true);
        console.log("Removing pair", pairId, "from idea", ideaId);
        try {
            const newPairIds = idea.mapping_pair_ids.filter(id => id !== pairId);
            await updateCoverLetterIdea(coverLetter.id, ideaId, { mapping_pair_ids: newPairIds });
            await reloadCoverLetter();
        } catch (err) { alert("Failed to remove pair."); await reloadCoverLetter(); } 
        finally { setIsSubmitting(false); }
    };
    const handleUnlinkIdea = async (paragraphId, ideaId) => {
        if (isSubmitting) return; setIsSubmitting(true);
        try {
            const sourcePara = paragraphs.find(p => p.id === paragraphId);
            const newIdeaIds = sourcePara.idea_ids.filter(id => id !== ideaId);
            await updateCoverLetterParagraph(coverLetter.id, paragraphId, { idea_ids: newIdeaIds });
            await reloadCoverLetter();
        } catch (err) { alert("Failed to unlink idea."); await reloadCoverLetter(); } 
        finally { setIsSubmitting(false); }
    };
    const handleUpdateIdeaAnnotation = async (ideaId, newAnnotation) => {
         try {
            setCoverLetter(prev => ({
                ...prev, ideas: prev.ideas.map(i => i.id === ideaId ? { ...i, annotation: newAnnotation } : i)
            }));
            if (maximizedIdea && maximizedIdea.id === ideaId) setMaximizedIdea(prev => ({ ...prev, annotation: newAnnotation }));
            await updateCoverLetterIdea(coverLetter.id, ideaId, { annotation: newAnnotation });
        } catch (err) { alert("Failed to update annotation."); await reloadCoverLetter(); }
    };
    const handleDeleteIdea = async (ideaId) => {
        if (!window.confirm("Delete this idea?")) return;
        setIsSubmitting(true);
        try { await deleteCoverLetterIdea(coverLetter.id, ideaId); await reloadCoverLetter(); } 
        catch (err) { alert("Failed to delete idea."); } finally { setIsSubmitting(false); }
    };
    const handleCreateCustomIdea = async (e) => {
        e.preventDefault(); if (!newIdeaTitle.trim()) return;
        setIsSubmitting(true);
        try { await addCoverLetterIdea(coverLetter.id, newIdeaTitle, []); await reloadCoverLetter(); setNewIdeaTitle(''); } 
        catch (err) { alert("Failed to create argument."); } finally { setIsSubmitting(false); }
    };
    const handleCreateParagraph = async (e) => {
        e.preventDefault(); if (!newParaPurpose.trim()) return;
        setIsSubmitting(true);
        try { await addCoverLetterParagraph(coverLetter.id, [], newParaPurpose); await reloadCoverLetter(); setNewParaPurpose(''); } 
        catch (err) { alert("Failed to create paragraph."); } finally { setIsSubmitting(false); }
    };
    const handleDeleteParagraph = async (paraId) => {
        if (!window.confirm("Delete this paragraph?")) return;
        setIsSubmitting(true);
        try { await deleteCoverLetterParagraph(coverLetter.id, paraId); await reloadCoverLetter(); } 
        catch (err) { alert("Failed to delete paragraph."); } finally { setIsSubmitting(false); }
    };
    const handleShowPreview = (item, type) => { setPreviewItem(item); setPreviewItemType(type); };

    // ... (DnD Logic - Unchanged) ...
    const sensors = useSensors(useSensor(PointerSensor), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));
    const handleDragStart = (event) => setActiveDragId(event.active.id);
    const handleDragCancel = () => setActiveDragId(null);
    const handleDragEnd = async (event) => {
        const { active, over } = event; setActiveDragId(null); if (!over) return;
        const activeData = active.data.current; const overId = over.id;
        if (active.id === over.id) return;
        
        let overContainerId = overId;
        if (over.data.current?.sortable?.containerId) {
            overContainerId = over.data.current.sortable.containerId;
        } else if (over.data.current?.type === 'idea-body') {
            overContainerId = `idea-card-body-${over.data.current.idea.id}`;
        } else if (over.data.current?.type === 'paragraph-body') {
            overContainerId = `para-card-body-${over.data.current.paragraph.id}`;
        }

        setIsSubmitting(true);
        try {
            if (activeData?.type === 'pair') {
                const pairId = activeData.pair.id;
                const sourceIdeaId = activeData.sourceIdeaId;
                let targetIdeaId = null;
                const overData = over.data.current;
                
                if (overData?.type === 'idea-body') targetIdeaId = overData.idea.id;
                else if (overData?.type === 'pair' && overData.sourceIdeaId) targetIdeaId = overData.sourceIdeaId;
                else if (overContainerId.startsWith('idea-card-body-')) targetIdeaId = overContainerId.split('idea-card-body-')[1];

                if (targetIdeaId) {
                    if (sourceIdeaId && sourceIdeaId !== targetIdeaId) {
                        const sourceIdea = ideaMap.get(sourceIdeaId);
                        if (sourceIdea) {
                            const newPairIds = sourceIdea.mapping_pair_ids.filter(id => id !== pairId);
                            await updateCoverLetterIdea(coverLetter.id, sourceIdeaId, { mapping_pair_ids: newPairIds });
                        }
                    }
                    const targetIdea = ideaMap.get(targetIdeaId);
                    if (targetIdea && !targetIdea.mapping_pair_ids.includes(pairId)) {
                        const newPairIds = [...targetIdea.mapping_pair_ids, pairId];
                        await updateCoverLetterIdea(coverLetter.id, targetIdeaId, { mapping_pair_ids: newPairIds });
                    }
                } else if (overContainerId === 'pair-pool' && sourceIdeaId) {
                    const sourceIdea = ideaMap.get(sourceIdeaId);
                     if (sourceIdea) {
                        const newPairIds = sourceIdea.mapping_pair_ids.filter(id => id !== pairId);
                        await updateCoverLetterIdea(coverLetter.id, sourceIdeaId, { mapping_pair_ids: newPairIds });
                    }
                }
            }
            if (activeData?.type === 'idea') {
                const ideaId = activeData.idea.id;
                const sourceParagraphId = activeData.sourceParagraphId;

                let targetParagraphId = null;
                if (overContainerId.startsWith('para-card-body-')) {
                    targetParagraphId = overContainerId.split('para-card-body-')[1];
                }

                if (targetParagraphId) {
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
        } catch (err) { alert("Error during drag operation."); await reloadCoverLetter(); } 
        finally { setIsSubmitting(false); }
    };

    const handleGeneratePrompt = async () => {
        if (coverLetter.paragraphs.length === 0) { alert("Please create at least one paragraph for the outline."); return; }
        setIsLoadingPrompt(true);
        try {
            const res = await generateCoverLetterPrompt(mapping.id);
            setClPromptJson(JSON.stringify(res.data, null, 2));
            setIsPromptModalOpen(true);
        } catch (err) { alert("Failed to generate prompt."); }
        finally { setIsLoadingPrompt(false); }
    };
    
    // ... (Loading/Error states - Unchanged) ...
    if (isLoading && !coverLetter) return <div className="text-center p-5"><div className="spinner-border text-primary"/></div>;
    if (error) return <div className="alert alert-danger">{error}</div>;
    if (!coverLetter || !mapping || !job) return <div className="alert alert-info">Initializing...</div>;

    const activeDragItem = activeDragId ? itemsById[activeDragId] : null;
    const col1SortableIds = groupedFilteredPairs.flatMap(group => group.pairs.map(p => `pool-pair-${p.id}`));

    const handleMaximizeCenter = () => {
        if (isLeftColOpen || isRightColOpen) {
            setIsLeftColOpen(false);
            setIsRightColOpen(false);
        } else {
            setIsLeftColOpen(true);
            setIsRightColOpen(true);
        }
    };

    // --- RENDER ---
    return (
        <DndContext sensors={sensors} collisionDetection={closestCorners} onDragStart={handleDragStart} onDragEnd={handleDragEnd} onDragCancel={handleDragCancel}>
            {/* The style tag now includes the height fix for mobile */}
            <style>{DND_STYLES}</style>
            
            <div className="step3-layout">
                {/* Header Section (Fixed) - Unchanged */}
                <div className="flex-shrink-0 mb-3">
                    <div className="d-flex justify-content-between align-items-center mb-4">
                        <div>
                            <h4 className="fw-bold text-dark mb-1 d-flex align-items-center gap-2">
                                <BrainCircuit size={24} className="text-primary"/> Cover Letter Builder
                            </h4>
                            <p className="text-muted small mb-0">Drag items between columns to structure your letter.</p>
                        </div>
                        <button className="btn btn-primary shadow-sm d-flex align-items-center gap-2" onClick={handleGeneratePrompt} disabled={isLoadingPrompt || coverLetter.paragraphs.length === 0}>
                            {isLoadingPrompt ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>} Generate Prompt
                        </button>
                    </div>

                    {isSubmitting && <div className="alert alert-light border small mb-3 py-2 d-flex align-items-center gap-2"><Loader2 size={14} className="animate-spin"/> Saving changes...</div>}
                </div>

                {/* Main Content (Scrollable Columns) */}
                <div className="cl-builder-container">
                    
                    {/* --- (MODIFIED) COL 1: EVIDENCE --- */}
                    <div className={`cl-column ${!isLeftColOpen ? 'cl-column-collapsed' : ''}`} style={getColumnStyle('left')}>
                        {isLeftColOpen ? (
                            <>
                                <div className="cl-column-header">
                                    <div className="d-flex align-items-center gap-2 overflow-hidden">
                                        <h6 className="fw-bold text-success mb-0 d-flex align-items-center gap-2 text-nowrap"><CheckCircle2 size={18}/> Evidence Pool</h6>
                                        <span className="badge bg-light text-muted border text-nowrap">{groupedFilteredPairs.length}</span>
                                    </div>
                                    <button className="btn btn-link p-0 text-muted ms-2" onClick={() => setIsLeftColOpen(false)}><PanelLeftClose size={18}/></button>
                                </div>
                                
                                {/* --- NEW SEARCH BAR --- */}
                                <div className="mb-2 position-relative">
                                    <Search size={14} className="position-absolute top-50 start-0 translate-middle-y ms-3 text-muted"/>
                                    <input
                                        type="text"
                                        className="form-control form-control-sm bg-white border shadow-sm ps-5"
                                        placeholder="Search evidence..."
                                        value={pairSearchTerm}
                                        onChange={(e) => setPairSearchTerm(e.target.value)}
                                    />
                                </div>
                                
                                {/* --- EXISTING FILTER --- */}
                                <div className="mb-3 position-relative">
                                    <Filter size={14} className="position-absolute top-50 start-0 translate-middle-y ms-3 text-muted"/>
                                    <select className="form-select form-select-sm bg-white border shadow-sm ps-5" value={pairFilterId} onChange={(e) => setPairFilterId(e.target.value)}>
                                        <option value="all">All Mapped Evidence</option>
                                        {(job?.features || []).map(f => <option key={f.id} value={f.id}>{f.description.substring(0, 60)}...</option>)}
                                    </select>
                                </div>
                                
                                <DroppableColumnBody id="pair-pool" items={col1SortableIds} >
                                    {groupedFilteredPairs.length === 0 && (
                                        <div className="text-center py-5 text-muted small border border-dashed rounded">
                                            No evidence found.
                                        </div>
                                    )}
                                    {groupedFilteredPairs.map(group => (
                                        <CL_EvidenceGroup key={group.title} cvItemText={group.title} pairs={group.pairs} ideaId={null} onRemovePair={null}/>
                                    ))}
                                </DroppableColumnBody>
                            </>
                        ) : (
                            <div className="d-flex flex-column align-items-center h-100 pt-4" onClick={() => setIsLeftColOpen(true)}>
                                <PanelLeftOpen size={20} className="text-muted mb-3"/>
                                <span className="vertical-text">EVIDENCE</span>
                            </div>
                        )}
                    </div>

                    {/* COL 2: ARGUMENTS (Unchanged) */}
                    <div className="cl-column shadow-sm" style={getColumnStyle('center')}>
                        {/* ... (All original code for Col 2) ... */}
                        <div className="cl-column-header">
                            <div className="d-flex align-items-center gap-2 overflow-hidden">
                                <h6 className="fw-bold text-primary mb-0 d-flex align-items-center gap-2 text-nowrap"><Lightbulb size={18}/> Arguments</h6>
                                <span className="badge bg-light text-muted border text-nowrap">{availableIdeas.length}</span>
                            </div>
                            <button className="btn btn-sm btn-light border shadow-sm text-muted d-flex align-items-center gap-1" onClick={handleMaximizeCenter}>
                                {isLeftColOpen || isRightColOpen ? <Maximize2 size={14}/> : <Minimize2 size={14}/>}
                                {isLeftColOpen || isRightColOpen ? "Focus" : "Reset"}
                            </button>
                        </div>
                        <div className="mb-3 d-flex gap-2">
                            <button className="btn btn-white border shadow-sm flex-grow-1 d-flex align-items-center justify-content-center gap-2 text-primary btn-sm" onClick={() => setIsSuggestionModalOpen(true)} disabled={isSubmitting}>
                                <Sparkles size={14}/> AI Ideas
                            </button>
                        </div>
                        <form onSubmit={handleCreateCustomIdea} className="d-flex gap-2 mb-3">
                            <input type="text" className="form-control form-control-sm border shadow-sm" value={newIdeaTitle} onChange={(e) => setNewIdeaTitle(e.target.value)} placeholder="New argument topic..." />
                            <button type="submit" className="btn btn-dark btn-sm shadow-sm" disabled={!newIdeaTitle.trim() || isSubmitting}><Plus size={16}/></button>
                        </form>
                        <DroppableColumnBody id="idea-pool" items={availableIdeas.map(i => `pool-idea-${i.id}`)}>
                            {availableIdeas.length === 0 && <div className="text-center py-5 text-muted small border border-dashed rounded">No unused arguments.</div>}
                            {availableIdeas.map(idea => {
                                const pairsInIdea = idea.mapping_pair_ids.map(id => pairMap.get(id)).filter(Boolean);
                                return <IdeaCard key={idea.id} idea={idea} pairsInIdea={pairsInIdea} onDelete={handleDeleteIdea} onUpdateAnnotation={handleUpdateIdeaAnnotation} onRemovePair={handleRemovePairFromIdea} fullCV={fullCV} onShowPreview={handleShowPreview} onMaximize={() => setMaximizedIdea(idea)} />;
                            })}
                        </DroppableColumnBody>
                    </div>
                    
                    {/* COL 3: OUTLINE (Unchanged) */}
                    <div className={`cl-column ${!isRightColOpen ? 'cl-column-collapsed' : ''}`} style={getColumnStyle('right')}>
                        {/* ... (All original code for Col 3) ... */}
                        {isRightColOpen ? (
                            <>
                                <div className="cl-column-header">
                                    <div className="d-flex align-items-center gap-2 overflow-hidden">
                                        <h6 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2 text-nowrap"><LayoutList size={18}/> Outline</h6>
                                    </div>
                                    <button className="btn btn-link p-0 text-muted ms-2" onClick={() => setIsRightColOpen(false)}><PanelRightClose size={18}/></button>
                                </div>
                                <form onSubmit={handleCreateParagraph} className="d-flex gap-2 mb-3">
                                    <input type="text" className="form-control form-control-sm border shadow-sm" value={newParaPurpose} onChange={(e) => setNewParaPurpose(e.target.value)} placeholder="New paragraph..." />
                                    <button type="submit" className="btn btn-dark btn-sm shadow-sm" disabled={!newParaPurpose.trim() || isSubmitting}><Plus size={16}/></button>
                                </form>
                                <DroppableColumnBody id="paragraph-outline" items={paragraphs.map(p => `para-${p.id}`)}>
                                    {paragraphs.map(para => {
                                        const ideasInPara = para.idea_ids.map(id => ideaMap.get(id)).filter(Boolean);
                                        return <ParagraphCard key={para.id} paragraph={para} ideasInParagraph={ideasInPara} onDelete={handleDeleteParagraph} onUnlinkIdea={handleUnlinkIdea} />;
                                    })}
                                </DroppableColumnBody>
                            </>
                        ) : (
                            <div className="d-flex flex-column align-items-center h-100 pt-4" onClick={() => setIsRightColOpen(true)}>
                                <PanelRightOpen size={20} className="text-muted mb-3"/>
                                <span className="vertical-text">OUTLINE</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* DragOverlay (Unchanged) */}
                <DragOverlay>
                    {activeDragItem ? (
                        activeDragItem.type === 'pair' ? <PairChip pair={activeDragItem} className="pair-chip dragging" /> 
                        : <div className="idea-card dragging p-3"><h6 className="mb-0 fw-bold">{activeDragItem.title}</h6></div>
                    ) : null}
                </DragOverlay>

                {/* PromptModal (Unchanged) */}
                <PromptModal isOpen={isPromptModalOpen} jsonString={clPromptJson} onClose={() => setIsPromptModalOpen(false)} />

                {/* Footer (Unchanged) */}
                <div className="d-flex justify-content-between flex-shrink-0 border-top pt-3 mt-3">
                    <button className="btn btn-outline-secondary d-flex align-items-center gap-2 px-4" onClick={onPrev}><ArrowLeft size={16}/> Back</button>
                    <button className="btn btn-outline-primary d-flex align-items-center gap-2 px-4" onClick={onNext}>Next <ArrowRight size={16}/></button>
                </div>
            </div>

            {/* Other Modals (Unchanged) */}
            {isSuggestionModalOpen && <CL_SuggestionModal isOpen={isSuggestionModalOpen} onClose={() => setIsSuggestionModalOpen(false)} coverLetter={coverLetter} mapping={mapping} job={job} onSuggestionsAccepted={handleSuggestionsAccepted} />}
            
            {previewItem && <CVItemPreviewModal isOpen={previewItem !== null} onClose={() => { setPreviewItem(null); setPreviewItemType(null); }} itemToPreview={{ item: previewItem, type: previewItemType }} allSkills={fullCV?.skills} allAchievements={fullCV?.achievements} allExperiences={fullCV?.experiences} allEducation={fullCV?.education} allHobbies={fullCV?.hobbies} />}
            
            {maximizedIdea && <IntelligentTextAreaModal isOpen={maximizedIdea !== null} onClose={() => setMaximizedIdea(null)} initialValue={maximizedIdea.annotation || ''} title={maximizedIdea.title} onSave={(newAnnotation) => { handleUpdateIdeaAnnotation(maximizedIdea.id, newAnnotation); setMaximizedIdea(null); }} cv={fullCV} onShowPreview={handleShowPreview} />}
        </DndContext>
    );
};

export default Step3_BuildCoverLetter;