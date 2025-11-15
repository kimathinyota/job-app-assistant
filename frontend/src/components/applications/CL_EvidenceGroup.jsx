// frontend/src/components/applications/CL_EvidenceGroup.jsx
import React, { useMemo } from 'react';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { SortablePairChip } from './Step3_BuildCoverLetter.jsx';

/**
 * A component that visually groups mapping pairs under their
 * CV evidence source (e.g., "From: Senior Engineer @ TechCorp").
 */
const CL_EvidenceGroup = ({
    cvItemText,
    pairs,
    ideaId, // If provided, pairs are inside an Idea card
    onRemovePair // Pass-through for removing pairs from ideas
}) => {
    
    // Generate unique IDs for sortable context
    const sortableIds = useMemo(() => {
        return pairs.map(p => 
            ideaId 
                ? `idea-${ideaId}-pair-${p.id}` 
                : `pool-pair-${p.id}`
        );
    }, [pairs, ideaId]);

    if (!pairs || pairs.length === 0) {
        return null;
    }

    return (
        // STYLING: Changed to a light card for better UI consistency
        <div className="card card-body bg-light bg-opacity-50 border-0 p-2 mb-2">
            <small 
                className="fw-bold text-muted mb-2 px-1"
                title={cvItemText}
            >
                {cvItemText}
            </small>
            <SortableContext
                id={`group-${ideaId || 'pool'}-${pairs[0].context_item_id}`}
                items={sortableIds}
                strategy={verticalListSortingStrategy}
            >
                {pairs.map(pair => (
                    <SortablePairChip
                        key={pair.id}
                        pair={pair}
                        ideaId={ideaId}
                        
                        // --- THIS IS THE FIX ---
                        // The 'x' button's onClick calls `onRemove()` with no arguments.
                        // We must wrap `onRemovePair` in a new function that
                        // provides the `ideaId` and `pair.id` it needs.
                        onRemove={ideaId ? () => onRemovePair(ideaId, pair.id) : null}
                    />
                ))}
            </SortableContext>
        </div>
    );
};

export default CL_EvidenceGroup;