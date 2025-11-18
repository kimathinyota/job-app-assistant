// frontend/src/components/applications/ArgumentCard.jsx
// (Keep imports the same)
import React, { useState } from 'react';
import { User, HelpCircle, Trash2, ChevronDown, ChevronUp, MessageSquare, GripVertical, RefreshCcw } from 'lucide-react';
import PairChip from './PairChip.jsx';
import AnnotationEditor from './AnnotationEditor.jsx';

const ArgumentCard = ({
    idea,
    paragraph,
    pairMap, // Now fully hydrated with text
    onUpdate,
    onDelete,
    onRevert,
    dragHandleProps
}) => {
    const isGapChip = idea.title.startsWith('❓');
    const isUserOwned = idea.owner === 'user';
    const [isExpanded, setIsExpanded] = useState(isUserOwned || isGapChip);

    // The pairMap now contains real text thanks to Step3_ActiveCoverLetter's useMemo
    const evidencePairs = idea.mapping_pair_ids.map(id => pairMap.get(id)).filter(Boolean);

    const borderClass = isGapChip ? 'border-warning-subtle' : isUserOwned ? 'border-success-subtle' : 'border-light';
    
    return (
        <div className={`card shadow-sm border-0 mb-0 transition-all ${borderClass}`} style={{borderLeft: isUserOwned ? '4px solid #198754' : isGapChip ? '4px solid #ffc107' : '1px solid #e2e8f0'}}>
            <div className="card-header border-0 d-flex align-items-center p-2 bg-transparent gap-2">
                <div className="text-muted cursor-grab px-1" {...dragHandleProps}><GripVertical size={14} /></div>
                
                <div className="flex-grow-1 d-flex align-items-center gap-2 overflow-hidden">
                    {isUserOwned && <User size={16} className="text-success" />}
                    <input 
                        type="text" 
                        defaultValue={idea.title} 
                        onBlur={(e) => {
                            if(e.target.value !== idea.title) onUpdate(idea.id, { title: e.target.value, owner: 'user' });
                        }}
                        className="form-control form-control-sm border-0 bg-transparent fw-bold text-dark shadow-none p-0"
                    />
                </div>

                <div className="d-flex align-items-center gap-1">
                    {isUserOwned && <button className="btn btn-sm btn-link text-muted hover-text-primary" onClick={() => onRevert(idea.id)}><RefreshCcw size={14}/></button>}
                    <button className="btn btn-sm btn-link text-muted hover-text-danger" onClick={() => onDelete(idea, paragraph)}><Trash2 size={14}/></button>
                    <button className="btn btn-sm btn-link text-muted" onClick={() => setIsExpanded(!isExpanded)}>{isExpanded ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}</button>
                </div>
            </div>

            {isExpanded && (
                <div className="card-body pt-0 pb-3 ps-4 pe-3">
                    {evidencePairs.length > 0 && (
                        <div className="mb-3 ps-2 border-start border-2 border-light">
                            {evidencePairs.map(pair => (
                                <PairChip key={pair.id} pair={pair} onRemove={isUserOwned ? () => onUpdate(idea.id, { mapping_pair_ids: idea.mapping_pair_ids.filter(pid => pid !== pair.id), owner: 'user' }) : null} />
                            ))}
                        </div>
                    )}
                    <div className="bg-light rounded-3 p-2 mt-2">
                        <div className="d-flex align-items-center gap-2 mb-1 text-muted small px-1"><MessageSquare size={12} /> <span className="fw-bold text-uppercase" style={{fontSize: '0.7rem'}}>Strategy Notes</span></div>
                        <AnnotationEditor initialValue={idea.annotation || ""} onSave={(val) => onUpdate(idea.id, { annotation: val, owner: 'user' })} />
                    </div>
                </div>
            )}
        </div>
    );
};

export default ArgumentCard;