// frontend/src/components/applications/ParagraphStudio.jsx
import React, { useState, useMemo } from 'react';
import { 
    Sparkles, Pencil, Plus, GripVertical, ChevronDown, ChevronUp, AlignLeft, Lightbulb,
    Trash2 
} from 'lucide-react';
import ArgumentCard from './ArgumentCard.jsx';
import ProseEditor from './ProseEditor.jsx';
import { SortableContext, verticalListSortingStrategy, useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

// --- SUB-COMPONENTS ---

const SortableArgumentList = ({ ideas, ...props }) => (
    <SortableContext items={ideas.map(i => i.id)} strategy={verticalListSortingStrategy}>
        <div className="d-flex flex-column gap-3">
            {ideas.map(idea => <SortableArgumentItem key={idea.id} idea={idea} {...props} />)}
        </div>
    </SortableContext>
);

const SortableArgumentItem = ({ idea, ...props }) => {
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: idea.id });
    const style = { transform: CSS.Transform.toString(transform), transition, opacity: isDragging ? 0.5 : 1 };
    return (
        <div ref={setNodeRef} style={style}>
            <ArgumentCard idea={idea} dragHandleProps={{ ...attributes, ...listeners }} {...props} />
        </div>
    );
};

const ParagraphStudio = ({
    paragraph,
    jobFeatures,
    fullCV,
    ideaMap,
    pairMap,
    isSubmitting,
    isReorderMode, 
    onUpdate,
    onAddArgument,
    onDeleteIdea,
    onRevertIdea,
    onShowPreview,
    onDeleteParagraph
}) => {
    const [view, setView] = useState('strategy'); 
    const [isCollapsed, setIsCollapsed] = useState(false);
    
    const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: paragraph.id, disabled: !isReorderMode });
    
    const style = { 
        transform: CSS.Transform.toString(transform), 
        transition: transition || 'box-shadow 0.2s ease, transform 0.2s ease', 
        opacity: isDragging ? 0.4 : 1,
        zIndex: isDragging ? 999 : 1
    };
    
    const ideas = useMemo(() => paragraph.idea_ids.map(id => ideaMap.get(id)).filter(Boolean), [paragraph.idea_ids, ideaMap]);

    // Local state for smooth title editing
    const [title, setTitle] = useState(paragraph.purpose);
    const handleTitleBlur = () => {
        if (title !== paragraph.purpose) onUpdate('paragraph', paragraph.id, { purpose: title, owner: 'user' });
    };

    return (
        <div 
            ref={setNodeRef} 
            style={style} 
            className={`paragraph-studio-card position-relative bg-white transition-all ${isReorderMode ? 'reorder-mode' : ''}`}
        >
            {/* --- HEADER --- */}
            <div 
                className={`d-flex align-items-center p-3 ${isCollapsed ? '' : 'border-bottom'}`} 
                style={{minHeight: '72px', cursor: isReorderMode ? 'grab' : 'default'}}
            >
                {/* Drag Handle (Show only in Reorder Mode) */}
                {isReorderMode && (
                    <div className="me-3 text-muted cursor-grab flex-shrink-0" {...attributes} {...listeners} title="Drag to reorder section">
                        <GripVertical size={20} />
                    </div>
                )}

                {/* Toggle Expand (Fixed Centering) */}
                <button 
                    className="btn btn-icon btn-light rounded-circle me-3 transition-transform d-flex align-items-center justify-content-center" 
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    style={{width: 36, height: 36, transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)'}}
                    title={isCollapsed ? "Expand Section" : "Collapse Section"}
                >
                    <ChevronDown size={20} className="text-dark" />
                </button>

                {/* Title & Meta */}
                <div className="flex-grow-1 d-flex flex-column justify-content-center">
                    <div className="d-flex align-items-center gap-2">
                        <input 
                            type="text"
                            className="form-control border-0 bg-transparent p-0 fw-bold text-dark shadow-none heading-font"
                            style={{fontSize: '1.15rem', letterSpacing: '-0.02em'}}
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            onBlur={handleTitleBlur}
                        />
                        {paragraph.owner === 'user' && (
                            <span className="badge bg-success-subtle text-success border border-success-subtle rounded-pill px-2 py-1" style={{fontSize: '0.65em', letterSpacing: '0.05em'}}>
                                CUSTOM
                            </span>
                        )}
                    </div>
                    
                    {/* Collapsed Summary State */}
                    {isCollapsed && (
                        <div className="d-flex align-items-center gap-3 mt-1 animate-fade-in">
                            <span className="d-flex align-items-center gap-1 text-muted tiny fw-medium">
                                <Lightbulb size={12} /> {ideas.length} Arguments
                            </span>
                            <span className="d-flex align-items-center gap-1 text-muted tiny fw-medium">
                                <AlignLeft size={12} /> {paragraph.draft_text ? 'Drafted' : 'Empty'}
                            </span>
                        </div>
                    )}
                </div>

                {/* Action Buttons */}
                <div className="d-flex align-items-center flex-shrink-0">
                    {/* View Toggle (Hidden when collapsed) */}
                    {!isCollapsed && !isReorderMode && (
                        <div className="d-flex bg-light rounded-pill p-1 ms-3">
                            <button 
                                onClick={() => setView('strategy')}
                                className={`btn btn-sm rounded-pill d-flex align-items-center gap-2 px-3 py-1 fw-semibold transition-all ${view === 'strategy' ? 'bg-white shadow-sm text-primary' : 'text-muted hover-text-dark'}`}
                                style={{fontSize: '0.85rem'}}
                            >
                                <Sparkles size={14} /> <span className="d-none d-sm-inline">Strategy</span>
                            </button>
                            <button 
                                onClick={() => setView('write')}
                                className={`btn btn-sm rounded-pill d-flex align-items-center gap-2 px-3 py-1 fw-semibold transition-all ${view === 'write' ? 'bg-white shadow-sm text-primary' : 'text-muted hover-text-dark'}`}
                                style={{fontSize: '0.85rem'}}
                            >
                                <Pencil size={14} /> <span className="d-none d-sm-inline">Write</span>
                            </button>
                        </div>
                    )}

                    {/* Delete Button (Show only when not reordering and delete function is passed) */}
                    {!isReorderMode && onDeleteParagraph && (
                        <button 
                            className="btn btn-icon btn-light text-danger rounded-circle ms-3 d-flex align-items-center justify-content-center"
                            onClick={() => onDeleteParagraph(paragraph.id)}
                            disabled={isSubmitting}
                            title="Delete Paragraph Section"
                            style={{width: 36, height: 36}}
                        >
                            <Trash2 size={20} />
                        </button>
                    )}
                </div>
            </div>

            {/* --- EXPANDED CONTENT --- */}
            {!isCollapsed && (
                <div className="p-4 bg-body animate-slide-down">
                    {view === 'strategy' ? (
                        <div>
                            {ideas.length === 0 ? (
                                <div className="empty-state-box text-center py-5 rounded-4 mb-3 border border-dashed">
                                    <div className="mb-3 text-muted opacity-50"><Sparkles size={32} /></div>
                                    <p className="text-muted fw-medium mb-0">No strategy defined yet.</p>
                                    <p className="text-muted small opacity-75">Add an argument to start planning this section.</p>
                                </div>
                            ) : (
                                <SortableArgumentList 
                                    ideas={ideas}
                                    paragraph={paragraph}
                                    pairMap={pairMap}
                                    fullCV={fullCV}
                                    jobFeatures={jobFeatures}
                                    onUpdate={(id, data) => onUpdate('idea', id, data)}
                                    onDelete={onDeleteIdea}
                                    onRevert={onRevertIdea}
                                    onShowPreview={onShowPreview}
                                />
                            )}

                            <button
                                className="btn btn-dashed w-100 mt-3 py-3 d-flex align-items-center justify-content-center gap-2 rounded-3 transition-all"
                                onClick={() => onAddArgument(paragraph.id, "unclassified")}
                                disabled={isSubmitting}
                            >
                                <div className="icon-circle bg-primary-subtle text-primary"><Plus size={16} /></div>
                                <span className="fw-semibold text-primary small">Add Argument Slot</span>
                            </button>
                        </div>
                    ) : (
                        <ProseEditor
                            paragraph={paragraph}
                            ideas={ideas}
                            pairMap={pairMap}
                            fullCV={fullCV}
                            jobFeatures={jobFeatures}
                            onSave={(id, data) => onUpdate('paragraph', id, data)}
                            onShowPreview={onShowPreview}
                        />
                    )}
                </div>
            )}
            
            <style>{`
                .paragraph-studio-card {
                    border-radius: 16px;
                    border: 1px solid rgba(0,0,0,0.06);
                    box-shadow: 0 4px 20px -4px rgba(0,0,0,0.05);
                    overflow: hidden;
                }
                .paragraph-studio-card:hover {
                    box-shadow: 0 10px 30px -5px rgba(0,0,0,0.08);
                    transform: translateY(-1px);
                }
                .reorder-mode {
                    border: 2px dashed var(--bs-primary);
                    box-shadow: none !important;
                    transform: none !important;
                }
                .btn-dashed {
                    border: 2px dashed rgba(var(--bs-primary-rgb), 0.2);
                    background: rgba(var(--bs-primary-rgb), 0.02);
                }
                .btn-dashed:hover {
                    background: rgba(var(--bs-primary-rgb), 0.06);
                    border-color: var(--bs-primary);
                }
                .icon-circle {
                    width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
                }
                .animate-fade-in { animation: fadeIn 0.3s ease-out; }
                .animate-slide-down { animation: slideDown 0.2s ease-out; }
                @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
                @keyframes slideDown { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
                .heading-font { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
                .tiny { font-size: 0.75rem; }
            `}</style>
        </div>
    );
};

export default ParagraphStudio;