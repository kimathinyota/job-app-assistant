// frontend/src/components/applications/ParagraphStudio.jsx
import React, { useState, useMemo } from 'react';
import { 
    Sparkles, Pencil, Plus, GripVertical, ChevronDown, AlignLeft, Lightbulb,
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
            {/* --- MODERN HEADER --- */}
            <div 
                className={`d-flex flex-column flex-md-row align-items-stretch align-items-md-center p-3 gap-3 ${isCollapsed ? '' : 'border-bottom'}`} 
                style={{cursor: isReorderMode ? 'grab' : 'default'}}
            >
                {/* TOP ROW: Identity (Title, Badge, Delete) */}
                <div className="d-flex align-items-center flex-grow-1 min-w-0">
                    
                    {/* Drag Handle (Desktop & Mobile) */}
                    {isReorderMode && (
                        <div className="me-2 text-muted cursor-grab flex-shrink-0" {...attributes} {...listeners}>
                            <GripVertical size={20} />
                        </div>
                    )}

                    {/* Expand Toggle (Increased Size) */}
                    <button 
                        className="btn btn-icon btn-light rounded-circle me-2 flex-shrink-0 d-flex align-items-center justify-content-center" 
                        onClick={() => setIsCollapsed(!isCollapsed)}
                        style={{width: 42, height: 42, transform: isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)', transition: 'transform 0.2s'}}
                    >
                        <ChevronDown size={24} className="text-dark" />
                    </button>

                    {/* Title Input Area */}
                    <div className="flex-grow-1 min-w-0 d-flex flex-column justify-content-center me-2">
                        <div className="d-flex align-items-center">
                            <input 
                                type="text"
                                className="form-control border-0 bg-transparent p-0 fw-bold text-dark shadow-none heading-font"
                                style={{fontSize: '1.1rem', letterSpacing: '-0.02em'}}
                                value={title}
                                onChange={(e) => setTitle(e.target.value)}
                                onBlur={handleTitleBlur}
                            />
                            {paragraph.owner === 'user' && (
                                <span className="badge bg-success-subtle text-success border border-success-subtle rounded-pill ms-2 flex-shrink-0 px-2" style={{fontSize: '0.6rem'}}>
                                    CUSTOM
                                </span>
                            )}
                        </div>
                        {/* Collapsed Meta */}
                        {isCollapsed && (
                            <div className="d-flex align-items-center gap-2 mt-1 opacity-75">
                                <span className="tiny d-flex align-items-center gap-1"><Lightbulb size={10}/> {ideas.length}</span>
                                <span className="tiny d-flex align-items-center gap-1"><AlignLeft size={10}/> {paragraph.draft_text ? 'Drafted' : 'Empty'}</span>
                            </div>
                        )}
                    </div>

                    {/* Delete Button (Larger Size) */}
                    {!isReorderMode && onDeleteParagraph && (
                        <button 
                            className="btn btn-icon btn-light text-danger rounded-circle flex-shrink-0 d-flex align-items-center justify-content-center hover-bg-danger-subtle"
                            onClick={() => onDeleteParagraph(paragraph.id)}
                            disabled={isSubmitting}
                            title="Delete Section"
                            style={{width: 42, height: 42}}
                        >
                            <Trash2 size={20} />
                        </button>
                    )}
                </div>

                {/* BOTTOM ROW (Mobile) / RIGHT SIDE (Desktop): Segmented Control */}
                {!isCollapsed && !isReorderMode && (
                    <div className="d-flex flex-shrink-0 w-100 w-md-auto">
                        <div className="bg-light rounded-3 p-1 d-flex w-100">
                            <button 
                                onClick={() => setView('strategy')}
                                className={`btn btn-sm flex-grow-1 flex-md-grow-0 rounded-3 d-flex align-items-center justify-content-center gap-2 px-3 py-1 fw-bold transition-all ${view === 'strategy' ? 'bg-white shadow-sm text-primary' : 'text-muted hover-text-dark'}`}
                                style={{minWidth: '100px'}}
                            >
                                <Sparkles size={14} /> Strategy
                            </button>
                            <button 
                                onClick={() => setView('write')}
                                className={`btn btn-sm flex-grow-1 flex-md-grow-0 rounded-3 d-flex align-items-center justify-content-center gap-2 px-3 py-1 fw-bold transition-all ${view === 'write' ? 'bg-white shadow-sm text-primary' : 'text-muted hover-text-dark'}`}
                                style={{minWidth: '100px'}}
                            >
                                <Pencil size={14} /> Write
                            </button>
                        </div>
                    </div>
                )}
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
                .hover-bg-danger-subtle:hover { background-color: var(--bs-danger-bg-subtle) !important; color: var(--bs-danger) !important; }
                
                /* Breakpoint utility for w-md-auto */
                @media (min-width: 768px) {
                    .w-md-auto { width: auto !important; }
                    .flex-md-grow-0 { flex-grow: 0 !important; }
                }
            `}</style>
        </div>
    );
};

export default ParagraphStudio;