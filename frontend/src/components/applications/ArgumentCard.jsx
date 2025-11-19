// frontend/src/components/applications/ArgumentCard.jsx
import React, { useState, useMemo, useRef } from 'react';
import { User, Trash2, ChevronDown, ChevronUp, GripVertical, RefreshCcw, Plus, Link as LinkIcon, Target, FileText, Quote, BrainCircuit } from 'lucide-react';
import AnnotationEditorImport from './AnnotationEditor.jsx';

// --- WRAPPER ---
const AnnotationEditorWrapper = React.forwardRef((props, ref) => <AnnotationEditorImport {...props} ref={ref} />);

// --- 1. MODERN MAPPING TICKET (with Notes Footer) ---
const ModernMappingCard = ({ pair, onRemove }) => {
    return (
        <div className="mapping-ticket d-flex flex-column border rounded-3 overflow-hidden bg-white mb-2 group transition-all hover-shadow">
            <div className="d-flex">
                {/* Left Decoration */}
                <div className="bg-success-subtle border-end border-success-subtle d-flex align-items-center justify-content-center" style={{width: '30px'}}>
                    <LinkIcon size={14} className="text-success opacity-75" />
                </div>
                
                {/* Content */}
                <div className="flex-grow-1 p-2 d-flex flex-column gap-2" style={{minWidth: 0}}>
                    {/* Requirement */}
                    <div>
                        <div className="d-flex align-items-center gap-2 mb-1">
                            <Target size={12} className="text-primary opacity-75" />
                            <span className="text-uppercase text-muted fw-bold tracking-wide" style={{fontSize: '0.65rem'}}>Requirement</span>
                        </div>
                        <div className="text-dark fw-medium small lh-sm text-wrap">
                            {pair.feature_text}
                        </div>
                    </div>

                    {/* Evidence */}
                    <div className="border-top border-light pt-2">
                        <div className="d-flex align-items-center gap-2 mb-1">
                            <FileText size={12} className="text-success opacity-75" />
                            <span className="text-uppercase text-muted fw-bold tracking-wide" style={{fontSize: '0.65rem'}}>Evidence</span>
                        </div>
                        <div className="text-dark small lh-sm text-wrap">
                            {pair.context_item_text}
                        </div>
                    </div>
                </div>

                {/* Action (Hover only) */}
                <div className="border-start border-light d-flex flex-column">
                    <button 
                        className="btn btn-link p-0 h-100 d-flex align-items-center justify-content-center text-muted hover-text-danger opacity-0 group-hover-opacity-100 transition-all" 
                        style={{width: '30px'}}
                        onClick={(e) => { e.stopPropagation(); onRemove(); }}
                        title="Remove Link"
                    >
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>

            {/* --- NOTE FOOTER (NON-OBTRUSIVE) --- */}
            {pair.annotation && (
                <div className="bg-light border-top px-3 py-2 d-flex align-items-start gap-2">
                    <Quote size={12} className="text-muted mt-1 flex-shrink-0 opacity-50" />
                    <span className="small text-muted fst-italic lh-sm" style={{fontSize: '0.8rem'}}>
                        {pair.annotation}
                    </span>
                </div>
            )}
        </div>
    );
};

// --- 2. MODERN ENTITY PILL ---
const ModernEntityPill = ({ item, onRemove }) => (
    <div className="d-inline-flex align-items-center gap-2 ps-2 pe-1 py-1 bg-white border rounded-pill shadow-sm me-2 mb-2 transition-all hover-border-primary">
        <div className="d-flex align-items-center gap-1 text-primary">
            <BrainCircuit size={12} />
        </div>
        <div className="d-flex flex-column" style={{lineHeight: 1}}>
            <span className="fw-bold text-dark" style={{fontSize: '0.75rem'}}>{item.title || item.name}</span>
            <span className="text-muted" style={{fontSize: '0.6rem'}}>{item._type}</span>
        </div>
        <button 
            className="btn btn-icon btn-sm rounded-circle text-muted hover-text-danger ms-1"
            style={{width: 20, height: 20}}
            onClick={(e) => { e.stopPropagation(); onRemove(); }}
        >
            <Trash2 size={12} />
        </button>
    </div>
);

const ArgumentCard = ({
    idea,
    paragraph,
    pairMap,
    fullCV,
    onUpdate,
    onDelete,
    onRevert,
    dragHandleProps
}) => {
    const isGapChip = idea.title.startsWith('❓');
    const isUserOwned = idea.owner === 'user';
    const [isExpanded, setIsExpanded] = useState(isUserOwned || isGapChip);
    const editorRef = useRef(null);
    const addButtonRef = useRef(null); // Ref for the add button

    const attachedPairs = useMemo(() => 
        (idea.mapping_pair_ids || []).map(id => pairMap.get(id)).filter(Boolean)
    , [idea.mapping_pair_ids, pairMap]);

    const attachedEntities = useMemo(() => {
        if (!idea.related_entity_ids || !fullCV) return [];
        const result = [];
        const find = (list, type) => list?.forEach(i => {
            if (idea.related_entity_ids.includes(i.id)) result.push({...i, _type: type});
        });
        find(fullCV.experiences, 'Experience');
        find(fullCV.projects, 'Project');
        find(fullCV.skills, 'Skill');
        find(fullCV.education, 'Education');
        return result;
    }, [idea.related_entity_ids, fullCV]);

    const mappingSuggestions = useMemo(() => {
        return Array.from(pairMap.values()).map(p => ({
            id: p.id,
            label: `Requirement: ${p.feature_text.substring(0,40)}...`,
            subtitle: `Evidence: ${p.context_item_text}`,
            _type: 'evidence' 
        }));
    }, [pairMap]);

    const handleMention = (item, type) => {
        if (type === 'evidence') {
            if (idea.mapping_pair_ids?.includes(item.id)) return;
            onUpdate(idea.id, { mapping_pair_ids: [...(idea.mapping_pair_ids || []), item.id], owner: 'user' });
        } else {
            if (idea.related_entity_ids?.includes(item.id)) return;
            onUpdate(idea.id, { related_entity_ids: [...(idea.related_entity_ids || []), item.id], owner: 'user' });
        }
    };

    const handleAddClick = () => {
        if (!isExpanded) setIsExpanded(true);
        // Call openMenu with the button's rect to enable smart positioning
        if (editorRef.current && addButtonRef.current) {
            const rect = addButtonRef.current.getBoundingClientRect();
            editorRef.current.openMenu(rect);
        }
    };

    const borderClass = isGapChip ? 'border-warning' : isUserOwned ? 'border-success' : 'border-light-subtle';
    const accentColor = isGapChip ? '#ffc107' : isUserOwned ? '#198754' : '#cbd5e1';

    return (
        <div className={`card shadow-sm mb-0 transition-all overflow-visible`} style={{border: '1px solid #e2e8f0', borderLeft: `4px solid ${accentColor}`}}>
            {/* HEADER */}
            <div className="card-header border-0 d-flex align-items-center p-2 bg-transparent gap-2">
                <div className="text-muted cursor-grab px-1" {...dragHandleProps}><GripVertical size={14} /></div>
                
                <div className="flex-grow-1 d-flex align-items-center gap-2 overflow-hidden">
                    {isUserOwned && <User size={14} className="text-success" />}
                    <input 
                        type="text" defaultValue={idea.title} 
                        onBlur={(e) => { if(e.target.value !== idea.title) onUpdate(idea.id, { title: e.target.value, owner: 'user' }); }}
                        className="form-control form-control-sm border-0 bg-transparent fw-bold text-dark shadow-none p-0"
                    />
                </div>

                {!isExpanded && (attachedPairs.length + attachedEntities.length) > 0 && (
                    <span className="badge bg-secondary-subtle text-secondary border border-secondary-subtle fw-normal rounded-pill">
                        {attachedPairs.length + attachedEntities.length} Links
                    </span>
                )}

                <div className="d-flex align-items-center gap-1 opacity-50 hover-opacity-100 transition-opacity">
                    {isUserOwned && <button className="btn btn-icon btn-sm text-muted hover-text-primary" onClick={() => onRevert(idea.id)}><RefreshCcw size={14}/></button>}
                    <button className="btn btn-icon btn-sm text-muted hover-text-danger" onClick={() => onDelete(idea, paragraph)}><Trash2 size={14}/></button>
                    <button className="btn btn-icon btn-sm text-muted" onClick={() => setIsExpanded(!isExpanded)}>{isExpanded ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}</button>
                </div>
            </div>

            {isExpanded && (
                <div className="card-body pt-0 pb-3 ps-3 pe-3">
                    
                    {/* LINKED EVIDENCE SECTION */}
                    <div className="mb-3">
                        <div className="d-flex align-items-center justify-content-between mb-2">
                            <span className="tiny fw-bold text-uppercase text-muted tracking-wider" style={{fontSize:'0.65rem'}}>Linked Context</span>
                            
                            {/* THE NEW ADD BUTTON */}
                            <button 
                                ref={addButtonRef}
                                className="btn btn-sm btn-light border shadow-sm text-primary d-flex align-items-center gap-2 px-2 py-1"
                                style={{fontSize:'0.75rem'}}
                                onClick={handleAddClick}
                            >
                                <Plus size={12}/> Link Item
                            </button>
                        </div>

                        {/* Mappings List */}
                        <div className="d-flex flex-column gap-0">
                            {attachedPairs.map(pair => (
                                <ModernMappingCard 
                                    key={pair.id} 
                                    pair={pair} 
                                    onRemove={() => onUpdate(idea.id, { mapping_pair_ids: idea.mapping_pair_ids.filter(pid => pid !== pair.id), owner: 'user' })} 
                                />
                            ))}
                        </div>

                        {/* Loose Entities List */}
                        <div className="mt-1">
                            {attachedEntities.map(ent => (
                                <ModernEntityPill 
                                    key={ent.id} 
                                    item={ent} 
                                    onRemove={() => onUpdate(idea.id, { related_entity_ids: idea.related_entity_ids.filter(eid => eid !== ent.id), owner: 'user' })} 
                                />
                            ))}
                        </div>

                        {/* Empty State */}
                        {attachedPairs.length === 0 && attachedEntities.length === 0 && (
                            <div className="p-3 border rounded-3 bg-light-subtle text-center border-dashed">
                                <span className="text-muted small opacity-75">
                                    No evidence linked yet. Use the <b>Link Item</b> button or type <b>@</b> below.
                                </span>
                            </div>
                        )}
                    </div>

                    {/* NOTES SECTION */}
                    <div className="position-relative">
                         <div className="absolute-label text-muted">
                            <Quote size={12} /> Notes
                         </div>
                         <AnnotationEditorWrapper 
                            ref={editorRef}
                            initialValue={idea.annotation || ""} 
                            onSave={(val) => onUpdate(idea.id, { annotation: val, owner: 'user' })} 
                            fullCV={fullCV}
                            extraSuggestions={mappingSuggestions}
                            onMention={handleMention}
                        />
                    </div>
                </div>
            )}
            
            <style>{`
                .mapping-ticket { border: 1px solid #e2e8f0; }
                .hover-shadow:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-color: #cbd5e1; }
                .group:hover .group-hover-opacity-100 { opacity: 1 !important; }
                .hover-border-primary:hover { border-color: var(--bs-primary) !important; }
                .absolute-label {
                    position: absolute; top: -18px; left: 4px; font-size: 0.65rem; 
                    font-weight: 700; text-transform: uppercase; display: flex; align-items: center; gap: 4px;
                }
                .border-dashed { border-style: dashed !important; }
            `}</style>
        </div>
    );
};

export default ArgumentCard;