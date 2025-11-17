// frontend/src/components/applications/PairChip.jsx
import React, { forwardRef } from 'react';
import { 
    Briefcase, GraduationCap, FolderGit2, Heart, FileText, 
    X, MessageSquare 
} from 'lucide-react';

const GetItemIcon = ({ type, size=16, className }) => {
    switch (type) {
        case 'experiences': return <Briefcase size={size} className={className} />;
        case 'projects': return <FolderGit2 size={size} className={className} />;
        case 'education': return <GraduationCap size={size} className={className} />;
        case 'hobbies': return <Heart size={size} className={className} />;
        default: return <FileText size={size} className={className} />;
    }
};

// This component is re-used from your original file to display evidence
export const PairChip = forwardRef(({ pair, onRemove, ...props }, ref) => (
    <div ref={ref} {...props} className="pair-chip group" style={{
        backgroundColor: 'white',
        border: '1px solid #e2e8f0',
        borderLeft: '3px solid #10b981',
        borderRadius: '0.5rem',
        padding: '0.75rem',
        marginBottom: '0.75rem',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
        position: 'relative',
        transition: 'all 0.2s ease',
    }}>
        {onRemove && (
            <button
                type="button"
                className="btn btn-link p-0 text-muted position-absolute"
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => { 
                    e.stopPropagation(); 
                    onRemove(); 
                }}
                title="Remove pair"
                style={{
                    top: '0.25rem',
                    right: '0.25rem',
                    opacity: 0,
                    transition: 'opacity 0.2s ease'
                }}
                onMouseEnter={e => e.currentTarget.style.opacity = 1}
                onMouseLeave={e => e.currentTarget.style.opacity = 0}
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
                    <div className="d-flex align-items-start gap-1 mt-2 pt-2 border-top" style={{borderColor: '#f1f5f9'}}>
                        <MessageSquare size={10} className="mt-1 flex-shrink-0 text-primary"/>
                        <span className="small fst-italic text-wrap" style={{fontSize: '0.75rem', lineHeight: '1.3', color: '#0d6efd'}}>
                            {pair.annotation}
                        </span>
                    </div>
                )}
            </div>
        </div>
        {/* Simple inline style for hover effect on parent */}
        <style>{`
            .pair-chip:hover {
                border-color: #cbd5e1;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            .pair-chip:hover > button {
                opacity: 0.7 !important;
            }
            .pair-chip:hover > button:hover {
                opacity: 1 !important;
                color: #dc3545 !important;
            }
        `}</style>
    </div>
));

export default PairChip;