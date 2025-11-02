// frontend/src/components/applications/JobModal.jsx
import React, { useState, useEffect } from 'react';
import { 
    fetchJobDetails, 
    upsertJob 
} from '../../api/applicationClient';

// This is a small, reusable component for the feature list
const FeatureItem = ({ feature, onDelete }) => (
    <div className="list-group-item d-flex justify-content-between align-items-center">
        <div>
            <span className={`badge bg-secondary me-2 text-capitalize`}>
                {feature.type.replace('_', ' ')}
            </span>
            {feature.description}
        </div>
        <button 
            type="button"
            className="btn-close"
            title="Delete feature"
            onClick={() => onDelete(feature.id)}
        ></button>
    </div>
);

// The main modal component
const JobModal = ({ initialJobId, isOpen, onClose, onJobUpdated }) => {
    const [isLoading, setIsLoading] = useState(false);
    const [isFetching, setIsFetching] = useState(false);
    const [jobId, setJobId] = useState(null);

    // --- Form State ---
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [job_url, setJobUrl] = useState('');
    const [application_end_date, setApplicationEndDate] = useState('');
    const [location, setLocation] = useState('');
    const [salary_range, setSalaryRange] = useState('');
    const [notes, setNotes] = useState('');
    
    // State for the list of features
    const [features, setFeatures] = useState([]);

    // State for the "add new feature" form
    const [newFeatureText, setNewFeatureText] = useState('');
    const [newFeatureType, setNewFeatureType] = useState('requirement');

    // Effect to load data when modal opens in "Edit" mode
    useEffect(() => {
        if (isOpen) {
            if (initialJobId) {
                // --- EDIT MODE ---
                setIsFetching(true);
                setJobId(initialJobId);
                fetchJobDetails(initialJobId)
                    .then(res => {
                        const job = res.data;
                        setTitle(job.title || '');
                        setCompany(job.company || '');
                        setJobUrl(job.job_url || '');
                        setApplicationEndDate(job.application_end_date || '');
                        setLocation(job.location || '');
                        setSalaryRange(job.salary_range || '');
                        setNotes(job.notes || '');
                        setFeatures(job.features || []);
                    })
                    .catch(err => {
                        console.error("Failed to load job", err);
                        alert("Failed to load job data.");
                        handleClose();
                    })
                    .finally(() => setIsFetching(false));
            } else {
                // --- ADD MODE ---
                // Reset all fields
                setJobId(null);
                setTitle('');
                setCompany('');
                setJobUrl('');
                setApplicationEndDate('');
                setLocation('');
                setSalaryRange('');
                setNotes('');
                setFeatures([]);
            }
        }
    }, [isOpen, initialJobId]);

    // --- Feature List Handlers ---
    const handleAddNewFeature = (e) => {
        e.preventDefault();
        if (!newFeatureText.trim()) return;
        
        const newFeature = {
            id: `temp_${Date.now()}`, // Temporary ID for React key
            description: newFeatureText,
            type: newFeatureType
        };
        
        setFeatures([...features, newFeature]);
        setNewFeatureText('');
        setNewFeatureType('requirement');
    };

    const handleDeleteFeature = (idToDelete) => {
        setFeatures(features.filter(f => f.id !== idToDelete));
    };

    // --- Main Submit Handler ---
    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);

        // 1. Create the payload
        const jobPayload = {
            id: jobId, // Will be null for "create", or an ID for "update"
            title,
            company,
            job_url,
            application_end_date,
            location,
            salary_range,
            notes,
            // Clean up temporary IDs before sending
            features: features.map(f => ({
                id: String(f.id).startsWith('temp_') ? null : f.id,
                description: f.description,
                type: f.type
            }))
        };

        try {
            // 2. Send the single "upsert" request
            await upsertJob(jobPayload);
            
            // 3. Notify parent and close
            onJobUpdated();
            handleClose();

        } catch (err) {
            alert(`Failed to ${jobId ? 'update' : 'create'} job.`);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const handleClose = () => {
        setIsLoading(false);
        setIsFetching(false);
        onClose(); // Call the parent's close handler
    };

    if (!isOpen) return null;

    return (
        <div className="modal" style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}>
            <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <form className="modal-content" onSubmit={handleSubmit}>
                    <div className="modal-header">
                        <h5 className="modal-title">
                            {initialJobId ? 'Edit Job' : 'Add New Job'}
                        </h5>
                        <button type="button" className="btn-close" onClick={handleClose}></button>
                    </div>
                    
                    <div className="modal-body">
                        {isFetching ? (
                            <p>Loading job details...</p>
                        ) : (
                            <>
                                {/* --- Core Details --- */}
                                <h6 className="h5">Core Details</h6>
                                <div className="row g-3 mb-3">
                                    <div className="col-md-6">
                                        <label className="form-label">Job Title*</label>
                                        <input type="text" className="form-control" value={title}
                                            onChange={(e) => setTitle(e.target.value)} required />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Company*</label>
                                        <input type="text" className="form-control" value={company}
                                            onChange={(e) => setCompany(e.target.value)} required />
                                    </div>
                                    <div className="col-md-12">
                                        <label className="form-label">Job URL</label>
                                        <input type="url" className="form-control" value={job_url}
                                            onChange={(e) => setJobUrl(e.target.value)} placeholder="https://..." />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Location</label>
                                        <input type="text" className="form-control" value={location}
                                            onChange={(e) => setLocation(e.target.value)} placeholder="e.g., Remote, New York" />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Salary Range</label>
                                        <input type="text" className="form-control" value={salary_range}
                                            onChange={(e) => setSalaryRange(e.target.value)} placeholder="e.g., $120k - $150k" />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label">Application End Date</label>
                                        <input type="date" className="form-control" value={application_end_date}
                                            onChange={(e) => setApplicationEndDate(e.target.value)} />
                                    </div>
                                    <div className="col-md-12">
                                        <label className="form-label">Notes</label>
                                        <textarea className="form-control" rows="2" value={notes}
                                            onChange={(e) => setNotes(e.target.value)} />
                                    </div>
                                </div>

                                <hr className="my-4" />

                                {/* --- Features/Requirements --- */}
                                <h6 className="h5">Requirements & Features ({features.length})</h6>
                                <div className="list-group mb-3" style={{maxHeight: '200px', overflowY: 'auto'}}>
                                    {features.length === 0 && (
                                        <div className="list-group-item text-muted fst-italic">No requirements added yet.</div>
                                    )}
                                    {features.map(f => (
                                        <FeatureItem key={f.id} feature={f} onDelete={handleDeleteFeature} />
                                    ))}
                                </div>
                                
                                {/* Add New Feature Form */}
                                <div className="border p-3 rounded bg-light">
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
                                        <button type="button" className="btn btn-success" onClick={handleAddNewFeature}>
                                            + Add Feature
                                        </button>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                    
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={handleClose}>Cancel</button>
                        <button type="submit" className="btn btn-primary" disabled={isLoading || isFetching}>
                            {isLoading ? "Saving..." : (jobId ? "Save Changes" : "Create Job")}
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default JobModal;