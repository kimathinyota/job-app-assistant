// frontend/src/components/applications/Step1_Mapping.jsx
import React, { useState } from 'react';
import { addMappingPair, deleteMappingPair } from '../../api/applicationClient';

const Step1_Mapping = ({ job, cv, mapping, onMappingChanged, onNext }) => {
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedCvId, setSelectedCvId] = useState(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Create maps for easy text lookup
    const reqTextMap = new Map(job.features.map(f => [f.id, f.description]));
    const expTextMap = new Map(cv.experiences.map(e => [e.id, `${e.title} @ ${e.company}`]));
    
    // Add projects to the same map
    cv.projects.forEach(p => {
        expTextMap.set(p.id, `${p.title} (Project)`);
    });

    const handleCreatePair = async () => {
        if (!selectedReqId || !selectedCvId) return;
        setIsSubmitting(true);
        try {
            // Note: Your backend route only accepts an experienceId, not a projectId.
            // This is a limitation in `backend/routes/mapping.py`
            // We'll assume for now that the user only clicks experiences.
            await addMappingPair(mapping.id, selectedReqId, selectedCvId);
            await onMappingChanged(); // Tell parent to refetch
            setSelectedReqId(null);
            setSelectedCvId(null);
        } catch (err) {
            alert("Failed to create pair.");
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDeletePair = async (pairId) => {
        // This assumes you add the `delete_mapping_pair` route
        // and add it to the `applicationClient.js`
        // try {
        //     await deleteMappingPair(mapping.id, pairId);
        //     await onMappingChanged(); // Tell parent to refetch
        // } catch (err) {
        //     alert("Failed to delete pair.");
        // }
        alert("Delete functionality not yet implemented in backend/routes/mapping.py");
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

                {/* Panel 2: Your CV Evidence */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                    <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {cv.experiences.map(exp => (
                            <button
                                key={exp.id}
                                type="button"
                                className={`list-group-item list-group-item-action ${selectedCvId === exp.id ? 'active' : ''}`}
                                onClick={() => setSelectedCvId(exp.id)}
                            >
                                {exp.title} @ {exp.company}
                            </button>
                        ))}
                        {/* You would also add projects here if backend route supported it */}
                    </div>
                </div>

                {/* Panel 3: Mapped Pairs */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">✅ Mapped Pairs</h6>
                    <button 
                        className="btn btn-success w-100 mb-3"
                        disabled={!selectedReqId || !selectedCvId || isSubmitting}
                        onClick={handleCreatePair}
                    >
                        {isSubmitting ? 'Pairing...' : 'Create Pair'}
                    </button>
                    
                    <div style={{ maxHeight: '340px', overflowY: 'auto' }}>
                        {mapping.pairs.map(pair => (
                            <div key={pair.id} className="card card-body p-3 mb-2 shadow-sm">
                                <p className="small mb-1">
                                    <strong>Req:</strong> {reqTextMap.get(pair.feature_id)}
                                </p>
                                <p className="small mb-2">
                                    <strong>Maps to:</strong> {expTextMap.get(pair.experience_id)}
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