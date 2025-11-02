// frontend/src/components/applications/JobEditorModal.jsx
import React, { useState, useEffect } from 'react';
import { 
    fetchJobDetails, 
    addJobFeature, 
    updateJob, 
    deleteJobFeature // We will add this to the API client
} from '../../api/applicationClient';

const JobEditorModal = ({ jobId, isOpen, onClose, onJobUpdated }) => {
    const [job, setJob] = useState(null);
    const [loading, setLoading] = useState(false);
    
    // State for editing the job's core details
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    
    // State for adding a new feature/requirement
    const [newFeatureText, setNewFeatureText] = useState('');
    const [newFeatureType, setNewFeatureType] = useState('requirement');

    const loadJobData = async () => {
        if (!jobId) return;
        setLoading(true);
        try {
            const res = await fetchJobDetails(jobId);
            setJob(res.data);
            setTitle(res.data.title);
            setCompany(res.data.company);
        } catch (err) {
            console.error("Failed to fetch job details", err);
            onClose(); // Close if we can't load
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (isOpen) {
            loadJobData();
        }
    }, [isOpen, jobId]);

    const handleUpdateJobDetails = async () => {
        try {
            await updateJob(jobId, { title, company });
            onJobUpdated(); // Refresh the main list
        } catch (err) {
            alert("Failed to update job details.");
        }
    };

    const handleAddFeature = async (e) => {
        e.preventDefault();
        if (!newFeatureText.trim()) return;
        try {
            await addJobFeature(jobId, newFeatureText, newFeatureType);
            setNewFeatureText('');
            loadJobData(); // Refresh the modal's data
        } catch (err) {
            alert("Failed to add feature.");
        }
    };
    
    const handleDeleteFeature = async (featureId) => {
        if (!window.confirm("Delete this requirement?")) return;
        try {
            await deleteJobFeature(jobId, featureId);
            loadJobData(); // Refresh the modal's data
        } catch (err) {
            alert("Failed to delete feature.");
        }
    };

    if (!isOpen) return null;

    return (
        <div className="modal" style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}>
            <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <div className="modal-content">
                    <div className="modal-header">
                        <h5 className="modal-title">Edit Job</h5>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>
                    
                    <div className="modal-body">
                        {loading || !job ? (
                            <p>Loading job details...</p>
                        ) : (
                            <>
                                {/* Part 1: Edit Title/Company */}
                                <div className="mb-3 p-3 border rounded">
                                    <h6 className="h5">Core Details</h6>
                                    <div className="row g-2 mb-2">
                                        <div className="col-md-6">
                                            <label className="form-label">Job Title</label>
                                            <input 
                                                type="text" 
                                                className="form-control"
                                                value={title}
                                                onChange={(e) => setTitle(e.target.value)}
                                            />
                                        </div>
                                        <div className="col-md-6">
                                            <label className="form-label">Company</label>
                                            <input 
                                                type="text" 
                                                className="form-control"
                                                value={company}
                                                onChange={(e) => setCompany(e.target.value)}
                                            />
                                        </div>
                                    </div>
                                    <button 
                                        className="btn btn-primary btn-sm" 
                                        onClick={handleUpdateJobDetails}
                                    >
                                        Save Details
                                    </button>
                                </div>
                                
                                {/* Part 2: Edit Requirements/Features */}
                                <div className="mt-4">
                                    <h6 className="h5">Requirements & Features ({job.features.length})</h6>
                                    <ul className="list-group mb-3" style={{maxHeight: '200px', overflowY: 'auto'}}>
                                        {job.features.map(f => (
                                            <li key={f.id} className="list-group-item d-flex justify-content-between align-items-center">
                                                <div>
                                                    <span className="badge bg-secondary me-2">{f.type}</span>
                                                    {f.description}
                                                </div>
                                                <button 
                                                    className="btn btn-danger btn-sm"
                                                    onClick={() => handleDeleteFeature(f.id)}
                                                >
                                                    Delete
                                                </button>
                                            </li>
                                        ))}
                                    </ul>
                                    
                                    {/* Add New Feature Form */}
                                    <form onSubmit={handleAddFeature} className="border p-3 rounded bg-light">
                                        <label className="form-label fw-medium">Add New Feature</label>
                                        <textarea 
                                            className="form-control mb-2"
                                            rows="2"
                                            value={newFeatureText}
                                            onChange={(e) => setNewFeatureText(e.target.value)}
                                            placeholder="e.g., 5+ years of React experience"
                                        />
                                        <div className="d-flex justify-content-between">
                                            <select 
                                                className="form-select w-auto"
                                                value={newFeatureType}
                                                onChange={(e) => setNewFeatureType(e.target.value)}
                                            >
                                                <option value="requirement">Requirement</option>
                                                <option value="responsibility">Responsibility</option>
                                                <option value="nice_to_have">Nice to Have</option>
                                                <option value="qualification">Qualification</option>
                                            </select>
                                            <button type="submit" className="btn btn-success">Add Feature</button>
                                        </div>
                                    </form>
                                </div>
                            </>
                        )}
                    </div>
                    
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>Done</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default JobEditorModal;