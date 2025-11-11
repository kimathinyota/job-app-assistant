// frontend/src/components/applications/CL_EvidenceGroup.jsx
import React, { useMemo } from 'react';
import { SortableContext, verticalListSortingStrategy } from '@dnd-kit/sortable';
import { SortablePairChip } from './Step3_BuildCoverLetter.jsx'; // Will be exported from Step3

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
        <div className="mb-3">
            <strong 
                className="d-block small text-muted border-bottom pb-1 mb-2"
                title={cvItemText}
            >
                From: {cvItemText.length > 50 ? cvItemText.substring(0, 50) + '...' : cvItemText}
            </strong>
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
                        // Only pass onRemove if it's for an idea
                        onRemove={ideaId ? () => onRemovePair(ideaId, pair.id) : null}
                    />
                ))}
            </SortableContext>
        </div>
    );
};

export default CL_EvidenceGroup;