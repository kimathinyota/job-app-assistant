// frontend/src/components/applications/ArgumentCard.jsx
import React, { useState, useMemo, useRef } from 'react';
import { User, Trash2, ChevronDown, ChevronUp, ChevronRight, GripVertical, RefreshCcw, Plus, Link as LinkIcon, Target, FileText, Quote, BrainCircuit, Award, Briefcase, GraduationCap, Cpu, Heart, FolderGit2 } from 'lucide-react';
import AnnotationEditorImport from './AnnotationEditor.jsx';

// --- WRAPPER ---
const AnnotationEditorWrapper = React.forwardRef((props, ref) => <AnnotationEditorImport {...props} ref={ref} />);

// --- ICON HELPER ---
const getEntityIcon = (type) => {
    switch (type) {
        case 'Achievement': return Award;
        case 'Experience': return Briefcase;
        case 'Education': return GraduationCap;
        case 'Project': return FolderGit2;
        case 'Hobby': return Heart;
        case 'Skill': return Cpu;
        default: return BrainCircuit;
    }
};

// --- 1. MODERN MAPPING TICKET (Requirement + Evidence) ---
const ModernMappingCard = ({ pair, onRemove }) => {
    return (
        <div className="mapping-ticket d-flex flex-column border rounded-3 overflow-hidden bg-white mb-2 group transition-all hover-shadow">
            <div className="d-flex">
                {/* Left Decoration */}
                <div className="bg-success-subtle border-end border-success-subtle d-flex align-items-center justify-content-center" style={{width: '32px', flexShrink: 0}}>
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
                <div className="border-start border-light d-flex align-items-center justify-content-center bg-light-subtle" style={{width: '32px', flexShrink: 0}}>
                    <button 
                        className="btn btn-link p-0 w-100 h-100 d-flex align-items-center justify-content-center text-muted hover-text-danger opacity-50 group-hover-opacity-100 transition-all" 
                        onClick={(e) => { e.stopPropagation(); onRemove(); }}
                        title="Remove Link"
                    >
                        <Trash2 size={14} />
                    </button>
                </div>
            </div>

            {/* --- NOTE FOOTER --- */}
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

// --- 2. MODERN ENTITY ROW (For "Heavy" items: Achievements, Experience, Projects) ---
const ModernEntityRow = ({ item, onRemove }) => {
    const Icon = getEntityIcon(item._type);

    // Determine main text and subtext based on type
    const title = item.title || item.degree || item.name || item.text || "Untitled";
    const subtitle = item._type;

    return (
        <div className="d-flex border rounded-3 overflow-hidden bg-white mb-2 group transition-all hover-shadow">
             {/* Left Icon Area */}
             <div className="bg-light border-end d-flex align-items-center justify-content-center text-primary" style={{width: '32px', flexShrink: 0}}>
                <Icon size={14} />
            </div>

            {/* Content Area (Text Wrap Enabled) */}
            <div className="flex-grow-1 p-2 d-flex flex-column justify-content-center" style={{minWidth: 0}}>
                <div className="fw-bold text-dark small lh-sm text-wrap mb-1">{title}</div>
                <div className="text-muted smaller lh-sm text-wrap text-truncate-2-lines">
                    {subtitle}
                </div>
            </div>

             {/* Delete Action (Vertically Centered) */}
             <div className="border-start d-flex align-items-center justify-content-center bg-light-subtle" style={{width: '32px', flexShrink: 0}}>
                <button 
                    className="btn btn-link p-0 w-100 h-100 d-flex align-items-center justify-content-center text-muted hover-text-danger opacity-50 group-hover-opacity-100 transition-all"
                    onClick={(e) => { e.stopPropagation(); onRemove(); }}
                >
                    <Trash2 size={14} />
                </button>
            </div>
        </div>
    );
};

// --- 3. MODERN ENTITY PILL (For "Light" items: Skills, Hobbies) ---
const ModernEntityPill = ({ item, onRemove }) => {
    const Icon = getEntityIcon(item._type);
    return (
        <div className="d-inline-flex align-items-center bg-white border rounded-pill shadow-sm me-2 mb-2 transition-all hover-border-primary" style={{maxWidth: '100%'}}>
            {/* Icon */}
            <div className="ps-2 pe-1 py-1 d-flex align-items-center text-primary">
                <Icon size={12} />
            </div>
            
            {/* Text */}
            <div className="d-flex flex-column py-1" style={{lineHeight: 1, minWidth: 0}}>
                <span className="fw-bold text-dark text-truncate px-1" style={{fontSize: '0.75rem', maxWidth: '180px'}}>
                    {item.title || item.name}
                </span>
            </div>

            {/* Divider & Trash */}
            <div className="border-start ms-1 h-100 py-1"></div>
            <button 
                className="btn btn-link p-0 d-flex align-items-center justify-content-center text-muted hover-text-danger px-2 rounded-e-pill"
                style={{height: '100%'}}
                onClick={(e) => { e.stopPropagation(); onRemove(); }}
            >
                <Trash2 size={12} />
            </button>
        </div>
    );
};

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
    
    // State for Card Expansion
    const [isExpanded, setIsExpanded] = useState(isUserOwned || isGapChip);
    
    // State for Context Section Visibility
    const [showContext, setShowContext] = useState(true);

    const editorRef = useRef(null);
    const addButtonRef = useRef(null);

    const attachedPairs = useMemo(() => 
        (idea.mapping_pair_ids || []).map(id => pairMap.get(id)).filter(Boolean)
    , [idea.mapping_pair_ids, pairMap]);

    // --- SPLIT ENTITIES ---
    const { blockEntities, pillEntities } = useMemo(() => {
        const blocks = [];
        const pills = [];
        
        if (idea.related_entity_ids && fullCV) {
            const process = (list, type, isPill) => list?.forEach(i => {
                if (idea.related_entity_ids.includes(i.id)) {
                    const enriched = {...i, _type: type};
                    isPill ? pills.push(enriched) : blocks.push(enriched);
                }
            });

            process(fullCV.experiences, 'Experience', false);
            process(fullCV.projects, 'Project', false);
            process(fullCV.education, 'Education', false);
            process(fullCV.achievements, 'Achievement', false); 
            process(fullCV.skills, 'Skill', true);              
            process(fullCV.hobbies, 'Hobby', false);             
        }
        return { blockEntities: blocks, pillEntities: pills };
    }, [idea.related_entity_ids, fullCV]);

    const totalLinkedCount = attachedPairs.length + blockEntities.length + pillEntities.length;

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
        if (!showContext) setShowContext(true);
    };

    // --- HANDLER: ADD BUTTON CLICK ---
    const handleAddClick = () => {
        // 1. Expand Card if closed
        if (!isExpanded) setIsExpanded(true);
        
        // 2. Expand Context area if closed
        if (!showContext) setShowContext(true);
        
        // 3. Trigger the Editor menu (use Timeout to allow render if it was hidden)
        setTimeout(() => {
            if (editorRef.current) {
                // Calls the method exposed in IntelligentTextArea
                editorRef.current.openMenu(); 
            }
        }, 100);
    };

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

                {!isExpanded && totalLinkedCount > 0 && (
                    <span className="badge bg-secondary-subtle text-secondary border border-secondary-subtle fw-normal rounded-pill">
                        {totalLinkedCount} Links
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
                    
                    {/* LINKED EVIDENCE SECTION (COLLAPSIBLE) */}
                    <div className="mb-3">
                        <div className="d-flex align-items-center justify-content-between mb-2">
                            {/* Toggle Header */}
                            <button 
                                className="btn btn-sm btn-link p-0 text-decoration-none d-flex align-items-center gap-1 text-muted hover-text-dark"
                                onClick={() => setShowContext(!showContext)}
                                title={showContext ? "Collapse Context" : "Expand Context"}
                            >
                                {showContext ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                <span className="tiny fw-bold text-uppercase tracking-wider" style={{fontSize:'0.65rem'}}>
                                    Linked Context {totalLinkedCount > 0 && !showContext && `(${totalLinkedCount})`}
                                </span>
                            </button>
                            
                            {/* ADD BUTTON */}
                            <button 
                                ref={addButtonRef}
                                className="btn btn-sm btn-light border shadow-sm text-primary d-flex align-items-center gap-2 px-2 py-1"
                                style={{fontSize:'0.75rem'}}
                                onClick={handleAddClick}
                            >
                                <Plus size={12}/> Link Item
                            </button>
                        </div>

                        {/* Collapsible Content Area */}
                        {showContext && (
                            <div className="animate-fade-in">
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

                                {/* Block Entities (Achievements, Exp, etc) */}
                                <div className="d-flex flex-column gap-0">
                                    {blockEntities.map(ent => (
                                        <ModernEntityRow
                                            key={ent.id}
                                            item={ent}
                                            onRemove={() => onUpdate(idea.id, { related_entity_ids: idea.related_entity_ids.filter(eid => eid !== ent.id), owner: 'user' })}
                                        />
                                    ))}
                                </div>

                                {/* Pill Entities (Skills, Hobbies) */}
                                <div className="mt-1">
                                    {pillEntities.map(ent => (
                                        <ModernEntityPill 
                                            key={ent.id} 
                                            item={ent} 
                                            onRemove={() => onUpdate(idea.id, { related_entity_ids: idea.related_entity_ids.filter(eid => eid !== ent.id), owner: 'user' })} 
                                        />
                                    ))}
                                </div>

                                {/* Empty State */}
                                {totalLinkedCount === 0 && (
                                    <div className="p-3 border rounded-3 bg-light-subtle text-center border-dashed">
                                        <span className="text-muted small opacity-75">
                                            No evidence linked yet. Use the <b>Link Item</b> button or type <b>@</b> below.
                                        </span>
                                    </div>
                                )}
                            </div>
                        )}
                    </div>

                    {/* NOTES SECTION (Always Visible when Expanded) */}
                    <div className="position-relative mt-3">
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
                .text-truncate-2-lines {
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }
                .smaller { font-size: 0.75rem; }
                @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
                .animate-fade-in { animation: fadeIn 0.2s ease-out; }
            `}</style>
        </div>
    );
};

export default ArgumentCard;