// frontend/src/components/applications/Step1_Mapping.jsx
import React, { useState, useMemo } from 'react';
import { addMappingPair, deleteMappingPair } from '../../api/applicationClient';

const Step1_Mapping = ({ job, cv, mapping, onMappingChanged, onNext }) => {
    const [selectedReqId, setSelectedReqId] = useState(null);
    
    // --- START CHANGES ---
    // State is now generic to hold any context item
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    // --- END CHANGES ---

    const [isSubmitting, setIsSubmitting] = useState(false);

    // --- START CHANGES ---
    // 1. Create a combined list of ALL CV evidence
    // We use useMemo to prevent this from recalculating on every render
    const cvEvidenceList = useMemo(() => [
        ...cv.experiences.map(item => ({
            id: item.id,
            type: 'experiences',
            text: `${item.title} @ ${item.company}`
        })),
        ...cv.projects.map(item => ({
            id: item.id,
            type: 'projects',
            text: `${item.title} (Project)`
        })),
        ...cv.education.map(item => ({
            id: item.id,
            type: 'education',
            text: `${item.degree} @ ${item.institution}`
        })),
        ...cv.hobbies.map(item => ({
            id: item.id,
            type: 'hobbies',
            text: `${item.name} (Hobby)`
        })),
    ], [cv.experiences, cv.projects, cv.education, cv.hobbies]); // Dependencies

    

    // 2. Create maps for easy text lookup for BOTH sides
    const reqTextMap = useMemo(() => 
        new Map(job.features.map(f => [f.id, f.description])), 
        [job.features]
    );
    
    const contextItemTextMap = useMemo(() => 
        new Map(cvEvidenceList.map(item => [item.id, item.text])),
        [cvEvidenceList]
    );
    // --- END CHANGES ---
    
    // 3. Update the click handler for the CV list
    const handleSelectContextItem = (item) => {
        setSelectedContextId(item.id);
        setSelectedContextType(item.type);
    };

    const handleCreatePair = async () => {
        // 4. Update check to use new state
        if (!selectedReqId || !selectedContextId || !selectedContextType) return;
        setIsSubmitting(true);
        try {
            // 5. Call the updated API client function
            // (This assumes applicationClient.js was updated as per our previous conversation)
            await addMappingPair(mapping.id, selectedReqId, selectedContextId, selectedContextType);
            await onMappingChanged(); // Tell parent to refetch
            
            // 6. Reset all selection state
            setSelectedReqId(null);
            setSelectedContextId(null);
            setSelectedContextType(null);
        } catch (err) {
            alert("Failed to create pair.");
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDeletePair = async (pairId) => {
        // This assumes your API client and backend route for deletion exist
        try {
            await deleteMappingPair(mapping.id, pairId);
            await onMappingChanged(); // Tell parent to refetch
        } catch (err) {
             alert("Failed to delete pair. This function might not be fully implemented in the backend.");
             console.error(err);
        }
    };


    return (
        <div>
            <h4 className="h5">Step 1: Map CV to Job Requirements</h4>
            <p className="text-muted">
                Click a requirement, then click the CV item that proves it.
            </p>
            <div className="row" style={{ minHeight: '400px' }}>
                {/* Panel 1: Job Requirements */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Job Requirements</h6>
                    <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {job.features.map(req => (
                            <button
                                key={req.id}
                                type="button"
                                className={`list-group-item list-group-item-action ${selectedReqId === req.id ? 'active' : ''}`}
                                onClick={() => setSelectedReqId(req.id)}
                            >
                                {req.description}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Panel 2: Your CV Evidence (MODIFIED) */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                    <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {/* --- START CHANGES --- */}
                        {/* 7. Map over the new combined cvEvidenceList */}
                        {cvEvidenceList.map(item => (
                            <button
                                key={item.id}
                                type="button"
                                className={`list-group-item list-group-item-action ${selectedContextId === item.id ? 'active' : ''}`}
                                onClick={() => handleSelectContextItem(item)}
                            >
                                {item.text}
                            </button>
                        ))}
                        {/* --- END CHANGES --- */}
                    </div>
                </div>

                {/* Panel 3: Mapped Pairs (MODIFIED) */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">✅ Mapped Pairs</h6>
                    <button 
                        className="btn btn-success w-100 mb-3"
                        // 8. Update disabled check
                        disabled={!selectedReqId || !selectedContextId || isSubmitting}
                        onClick={handleCreatePair}
                    >
                        {isSubmitting ? 'Pairing...' : 'Create Pair'}
                    </button>
                    
                    <div style={{ maxHeight: '340px', overflowY: 'auto' }}>
                        {/* 9. Update rendering to use new model fields */}
                        {mapping.pairs.map(pair => (
                            <div key={pair.id} className="card card-body p-3 mb-2 shadow-sm">
                                <p className="small mb-1">
                                    <strong>Req:</strong> {reqTextMap.get(pair.feature_id) || pair.feature_text}
                                </p>
                                <p className="small mb-2">
                                    <strong>Maps to:</strong> 
                                    {/* Use the new map and new ID field from the model */}
                                    {contextItemTextMap.get(pair.context_item_id) || pair.context_item_text}
                                </p>
                                <button 
                                    className="btn btn-danger btn-sm"
                                    onClick={() => handleDeletePair(pair.id)}
                                >
                                    Ungroup
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
            
            <div className="text-end mt-4">
                <button className="btn btn-primary" onClick={onNext}>
                    Next: Review CV &gt;
                </button>
            </div>
        </div>
    );
};

export default Step1_Mapping;