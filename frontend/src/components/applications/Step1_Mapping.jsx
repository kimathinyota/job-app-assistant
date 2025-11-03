// frontend/src/components/applications/Step1_Mapping.jsx
import React, { useState, useMemo } from 'react';
import { addMappingPair, deleteMappingPair } from '../../api/applicationClient';
import CVItemPreviewModal from './CVItemPreviewModal';

const Step1_Mapping = ({ job, cv, mapping, onMappingChanged, onNext }) => {
    const [selectedReqId, setSelectedReqId] = useState(null);
    const [selectedContextId, setSelectedContextId] = useState(null);
    const [selectedContextType, setSelectedContextType] = useState(null);
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    // --- 2. ADD MODAL STATE ---
    const [previewItem, setPreviewItem] = useState(null); // e.g., { item: {...}, type: 'experiences' }
    const [isModalOpen, setIsModalOpen] = useState(false);


    // --- 3. UPDATE cvEvidenceList to include the full item ---
    const cvEvidenceList = useMemo(() => [
        ...cv.experiences.map(item => ({
            id: item.id,
            type: 'experiences',
            text: `${item.title} @ ${item.company}`,
            item: item // <-- ADD THE FULL ITEM OBJECT
        })),
        ...cv.projects.map(item => ({
            id: item.id,
            type: 'projects',
            text: `${item.title} (Project)`,
            item: item // <-- ADD THE FULL ITEM OBJECT
        })),
        ...cv.education.map(item => ({
            id: item.id,
            type: 'education',
            text: `${item.degree} @ ${item.institution}`,
            item: item // <-- ADD THE FULL ITEM OBJECT
        })),
        ...cv.hobbies.map(item => ({
            id: item.id,
            type: 'hobbies',
            text: `${item.name} (Hobby)`,
            item: item // <-- ADD THE FULL ITEM OBJECT
        })),
    ], [cv.experiences, cv.projects, cv.education, cv.hobbies]);


    // --- Text Lookup Maps (Unchanged) ---
    const reqTextMap = useMemo(() => 
        new Map(job.features.map(f => [f.id, f.description])), 
        [job.features]
    );
    const contextItemTextMap = useMemo(() => 
        new Map(cvEvidenceList.map(item => [item.id, item.text])),
        [cvEvidenceList]
    );

    // --- Disabling Logic (This is what you described) ---
    const existingPairSet = useMemo(() => 
        new Set(
            mapping.pairs.map(p => `${p.feature_id}_${p.context_item_id || p.experience_id}`)
        ),
        [mapping.pairs]
    );

    // Find all evidence items paired with the *currently selected requirement*
    const disabledEvidenceIds = useMemo(() => {
        if (!selectedReqId) return new Set(); // If no req is selected, disable nothing
        return new Set(
            mapping.pairs
                .filter(p => p.feature_id === selectedReqId)
                .map(p => p.context_item_id || p.experience_id)
        );
    }, [mapping.pairs, selectedReqId]);

    // Find all requirement items paired with the *currently selected evidence*
    const disabledReqIds = useMemo(() => {
        if (!selectedContextId) return new Set(); // If no evidence is selected, disable nothing
        return new Set(
            mapping.pairs
                .filter(p => (p.context_item_id || p.experience_id) === selectedContextId)
                .map(p => p.feature_id)
        );
    }, [mapping.pairs, selectedContextId]);

    // --- 4. ADD MODAL HANDLERS ---
    const handlePreviewClick = (e, item, type) => {
        e.stopPropagation(); // Stop the click from selecting/deselecting the item
        setPreviewItem({ item, type });
        setIsModalOpen(true);
    };

    const handleCloseModal = () => {
        setIsModalOpen(false);
        setPreviewItem(null);
    };
    
    // --- FIX: This is the simple handler you need ---
    const handleSelectContextItem = (item) => {
        setSelectedContextId(item.id);
        setSelectedContextType(item.type);
        // We DO NOT clear selectedReqId here
    };

    const handleCreatePair = async () => {
        if (!selectedReqId || !selectedContextId || !selectedContextType) return;
        
        const pairKey = `${selectedReqId}_${selectedContextId}`;
        if (existingPairSet.has(pairKey) && !annotation.trim()) {
            alert("This pair already exists. Please add an annotation to create a duplicate link.");
            return;
        }
        
        setIsSubmitting(true);
        try {
            await addMappingPair(mapping.id, selectedReqId, selectedContextId, selectedContextType, annotation);
            await onMappingChanged(); // Refetch
            
            // Reset all selections
            setSelectedReqId(null);
            setSelectedContextId(null);
            setSelectedContextType(null);
            setAnnotation(""); // Clear annotation box
        } catch (err) {
            alert(`Failed to create pair: ${err.response?.data?.detail || err.message}`);
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDeletePair = async (pairId) => {
        try {
            await deleteMappingPair(mapping.id, pairId);
            await onMappingChanged(); // Refetch
        } catch (err) {
             alert(`Failed to delete pair: ${err.response?.data?.detail || err.message}`);
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
                        {job.features.map(req => {
                            // Item is disabled *only* if a context item is selected
                            // AND this req is paired with it
                            const isDisabled = disabledReqIds.has(req.id);
                            return (
                                <button
                                    key={req.id}
                                    type="button"
                                    className={`list-group-item list-group-item-action ${selectedReqId === req.id ? 'active' : ''} ${isDisabled ? 'list-group-item-light text-muted' : ''}`}
                                    // --- FIX: Use simple setter ---
                                    onClick={() => setSelectedReqId(req.id)}
                                    disabled={isDisabled}
                                >
                                    {req.description}
                                    {isDisabled && <span className="ms-2 small">(Paired)</span>}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Panel 2: Your CV Evidence (MODIFIED) */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">Your CV Evidence</h6>
                    <div className="list-group" style={{ maxHeight: '400px', overflowY: 'auto' }}>
                        {/* --- 5. MODIFY THE RENDER LOOP --- */}
                        {cvEvidenceList.map(item => {
                            const isDisabled = disabledEvidenceIds.has(item.id);
                            const isActive = selectedContextId === item.id;
                            
                            return (
                                <div
                                    key={item.id}
                                    // Use 'd-flex' to position the icon
                                    className={`list-group-item list-group-item-action d-flex justify-content-between align-items-center ${isActive ? 'active' : ''} ${isDisabled ? 'list-group-item-light text-muted' : ''}`}
                                    onClick={() => !isDisabled && handleSelectContextItem(item)}
                                    style={{ cursor: isDisabled ? 'not-allowed' : 'pointer' }}
                                >
                                    {/* Text content (make it take up available space) */}
                                    <span style={{ flex: 1, marginRight: '10px' }}>
                                        {item.text}
                                        {isDisabled && <span className="ms-2 small">(Paired)</span>}
                                    </span>
                                    
                                    {/* Preview Button */}
                                    <button
                                        type="button"
                                        // Use 'btn-outline-light' when active for better contrast
                                        className={`btn btn-sm ${isActive ? 'btn-outline-light' : 'btn-outline-secondary'}`}
                                        onClick={(e) => handlePreviewClick(e, item.item, item.type)}
                                        title="Preview Item"
                                        // zIndex ensures it's clickable over the parent div's click handler
                                        style={{ zIndex: 5 }} 
                                    >
                                        👁️
                                    </button>
                                </div>
                            );
                        })}
                        {/* --- END MODIFICATION --- */}
                    </div>
                </div>

                {/* Panel 3: Mapped Pairs */}
                <div className="col-4">
                    <h6 className="border-bottom pb-2">✅ Mapped Pairs</h6>
                    
                    <div className="mb-2">
                        <textarea
                            className="form-control form-control-sm"
                            rows="2"
                            placeholder="Optional: Add annotation..."
                            value={annotation}
                            onChange={(e) => setAnnotation(e.target.value)}
                        ></textarea>
                    </div>

                    <button 
                        className="btn btn-success w-100 mb-3"
                        // Button is active if one from each side is selected
                        disabled={!selectedReqId || !selectedContextId || isSubmitting}
                        onClick={handleCreatePair}
                    >
                        {isSubmitting ? 'Pairing...' : 'Create Pair'}
                    </button>
                    
                    <div style={{ maxHeight: '340px', overflowY: 'auto' }}>
                        {mapping.pairs.map(pair => (
                            <div key={pair.id} className="card card-body p-3 mb-2 shadow-sm">
                                <p className="small mb-1">
                                    <strong>Req:</strong> {reqTextMap.get(pair.feature_id) || pair.feature_text}
                                </p>
                                <p className="small mb-2">
                                    <strong>Maps to: </strong> 
                                    {/* This backwards-compatible check is still necessary */}
                                    {/* Once your backend is fixed, this will work */}
                                    {contextItemTextMap.get(pair.context_item_id || pair.experience_id) || pair.context_item_text || pair.experience_text}
                                </p>
                                
                                {pair.annotation && (
                                    <p className="small fst-italic border-top pt-2 mt-2 mb-2">
                                        <strong>Note:</strong> {pair.annotation}
                                    </p>
                                )}
                                
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
            
            {/* The "Next" button */}
            <div className="text-end mt-4">
                <button className="btn btn-primary" onClick={onNext}>
                    Next: Review CV &gt;
                </button>
            </div>

            {/* --- 6. RENDER THE MODAL --- */}
            <CVItemPreviewModal
                isOpen={isModalOpen}
                onClose={handleCloseModal}
                itemToPreview={previewItem}
                // Pass all the data from the main CV object
                allSkills={cv.skills}
                allAchievements={cv.achievements}
                allExperiences={cv.experiences}
                allEducation={cv.education}
            />
        </div>
    );
};

export default Step1_Mapping;