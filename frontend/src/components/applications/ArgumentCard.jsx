// frontend/src/components/applications/ArgumentCard.jsx
import React, { useState } from 'react';
import { User, HelpCircle, Trash2, ChevronDown, ChevronUp, MessageSquare } from 'lucide-react';
import PairChip from './PairChip.jsx'; // The evidence chip
import AnnotationEditor from './AnnotationEditor.jsx'; // The inline notes editor

const ArgumentCard = ({
    idea,
    paragraph,
    pairMap,
    fullCV,
    jobFeatures,
    onUpdateIdea,
    onDeleteIdea,
    onShowPreview
}) => {
    const isGapChip = idea.title.startsWith('❓');
    const isUserOwned = idea.owner === 'user';
    const [isExpanded, setIsExpanded] = useState(isUserOwned || isGapChip); // Auto-expand user/gap chips

    const evidencePairs = idea.mapping_pair_ids
        .map(id => pairMap.get(id))
        .filter(Boolean);

    const cardClass = isGapChip 
        ? 'border-warning-subtle' 
        : isUserOwned
        ? 'border-success-subtle'
        : 'border-light';

    const headerClass = isGapChip 
        ? 'bg-warning-subtle text-warning-emphasis' 
        : isUserOwned
        ? 'bg-success-subtle text-success-emphasis'
        : 'bg-light';
        
    const handleTitleChange = (e) => {
        // Simple inline title edit on blur
        const newTitle = e.target.value.trim();
        if (newTitle && newTitle !== idea.title) {
            onUpdateIdea(idea.id, { title: newTitle });
        }
    };
    
    const handleAnnotationSave = (newAnnotation) => {
        if (newAnnotation !== idea.annotation) {
            onUpdateIdea(idea.id, { annotation: newAnnotation });
        }
    };

    const handleRemovePair = (pairId) => {
        if (!window.confirm("Remove this piece of evidence from the argument?")) return;
        const newPairIds = idea.mapping_pair_ids.filter(id => id !== pairId);
        onUpdateIdea(idea.id, { mapping_pair_ids: newPairIds });
    };

    return (
        <div className={`card ${cardClass}`} style={{borderWidth: '2px'}}>
            <div className={`card-header d-flex justify-content-between align-items-center p-2 ${headerClass}`}>
                <div className="d-flex align-items-center gap-2 flex-grow-1">
                    <button 
                        className="btn btn-sm btn-link text-decoration-none" 
                        onClick={() => setIsExpanded(!isExpanded)}
                        title={isExpanded ? "Collapse" : "Expand"}
                    >
                        {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
                    </button>
                    
                    <span className="flex-shrink-0">
                        {isGapChip && <HelpCircle size={18} />}
                        {isUserOwned && !isGapChip && <User size={18} />}
                    </span>
                    
                    <input
                        type="text"
                        defaultValue={idea.title}
                        onBlur={handleTitleChange}
                        className="form-control form-control-sm border-0 bg-transparent fw-bold"
                        style={{boxShadow: 'none', color: 'inherit'}}
                        disabled={!isUserOwned && !isGapChip} // Can only edit titles of user/gap chips
                    />
                </div>
                
                {isUserOwned && (
                    <button 
                        title="Delete Argument"
                        className="btn btn-sm btn-link text-danger"
                        onClick={() => onDeleteIdea(idea, paragraph)}
                    >
                        <Trash2 size={16} />
                    </button>
                )}
            </div>

            {isExpanded && (
                <div className="card-body p-3">
                    {evidencePairs.length > 0 && (
                        <>
                            <h6 className="small text-muted fw-bold text-uppercase">Evidence</h6>
                            <div className="d-flex flex-column gap-2 mb-3">
                                {evidencePairs.map(pair => (
                                    <PairChip 
                                        key={pair.id}
                                        pair={pair}
                                        onRemove={isUserOwned ? () => handleRemovePair(pair.id) : null} // Only user-owned can remove pairs
                                    />
                                ))}
                            </div>
                        </>
                    )}
                    
                    <div>
                        <h6 className="small text-muted fw-bold text-uppercase d-flex align-items-center gap-1">
                            <MessageSquare size={14} /> 
                            Notes for AI (or you)
                        </h6>
                        <AnnotationEditor
                            initialValue={idea.annotation || ""}
                            onSave={handleAnnotationSave}
                            fullCV={fullCV}
                            jobFeatures={jobFeatures}
                            onShowPreview={onShowPreview}
                        />
                    </div>
                </div>
            )}
        </div>
    );
};

export default ArgumentCard;