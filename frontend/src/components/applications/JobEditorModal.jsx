// frontend/src/components/applications/JobEditorModal.jsx
import React, { useState, useEffect } from 'react';
import { 
    Save, 
    Plus, 
    Trash2, 
    X, 
    Briefcase, 
    Building2, 
    AlertCircle, 
    Star, 
    ListChecks, 
    FileText,
    Check
} from 'lucide-react';
import { 
    fetchJobDetails, 
    addJobFeature, 
    updateJob, 
    deleteJobFeature 
} from '../../api/applicationClient';

const JobEditorModal = ({ jobId, isOpen, onClose, onJobUpdated }) => {
    const [job, setJob] = useState(null);
    const [loading, setLoading] = useState(false);
    const [savingCore, setSavingCore] = useState(false);
    
    // State for editing the job's core details
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    
    // State for adding a new feature/requirement
    const [newFeatureText, setNewFeatureText] = useState('');
    const [newFeatureType, setNewFeatureType] = useState('requirement');
    const [addingFeature, setAddingFeature] = useState(false);

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
            onClose(); 
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
        setSavingCore(true);
        try {
            await updateJob(jobId, { title, company });
            onJobUpdated(); 
            // Optional: Show a small "Saved" toast or feedback here
        } catch (err) {
            alert("Failed to update job details.");
        } finally {
            setSavingCore(false);
        }
    };

    const handleAddFeature = async (e) => {
        e.preventDefault();
        if (!newFeatureText.trim()) return;
        setAddingFeature(true);
        try {
            await addJobFeature(jobId, newFeatureText, newFeatureType);
            setNewFeatureText('');
            await loadJobData(); 
        } catch (err) {
            alert("Failed to add feature.");
        } finally {
            setAddingFeature(false);
        }
    };
    
    const handleDeleteFeature = async (featureId) => {
        if (!window.confirm("Delete this requirement?")) return;
        try {
            await deleteJobFeature(jobId, featureId);
            await loadJobData(); 
        } catch (err) {
            alert("Failed to delete feature.");
        }
    };

    // Helper for feature styling (matches JobCard)
    const getFeatureStyle = (type) => {
        const t = type ? type.toLowerCase() : '';
        if (t.includes('require') || t.includes('must')) {
            return { 
                label: 'Must Have', 
                icon: AlertCircle,
                badgeClass: 'text-danger bg-danger bg-opacity-10 border-danger-subtle' 
            };
        }
        if (t.includes('nice') || t.includes('bonus')) {
            return { 
                label: 'Bonus', 
                icon: Star,
                badgeClass: 'text-success bg-success bg-opacity-10 border-success-subtle' 
            };
        }
        if (t.includes('responsibility')) {
            return { 
                label: 'Responsibility', 
                icon: ListChecks,
                badgeClass: 'text-primary bg-primary bg-opacity-10 border-primary-subtle' 
            };
        }
        return { 
            label: type ? type.replace(/_/g, ' ') : 'Info', 
            icon: FileText,
            badgeClass: 'text-secondary bg-secondary bg-opacity-10 border-secondary-subtle' 
        };
    };

    if (!isOpen) return null;

    return (
        <div className="modal show d-block" style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }} tabIndex="-1">
            <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <div className="modal-content border-0 shadow-lg rounded-xl overflow-hidden">
                    
                    {/* Header */}
                    <div className="modal-header border-bottom-0 bg-white pb-0">
                        <div className="d-flex align-items-center gap-3">
                            <div className="bg-primary bg-opacity-10 p-2 rounded-circle text-primary">
                                <Briefcase size={20} />
                            </div>
                            <h5 className="modal-title fw-bold text-dark">Edit Job Details</h5>
                        </div>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>
                    
                    <div className="modal-body p-4">
                        {loading || !job ? (
                            <div className="text-center py-5">
                                <div className="spinner-border text-primary mb-3" role="status"></div>
                                <p className="text-muted">Loading job data...</p>
                            </div>
                        ) : (
                            <div className="d-flex flex-column gap-4">
                                {/* Part 1: Core Details Card */}
                                <div className="card border-0 shadow-sm bg-light bg-opacity-50">
                                    <div className="card-body p-3">
                                        <h6 className="text-uppercase text-muted small fw-bold mb-3 d-flex align-items-center gap-2">
                                            <Building2 size={14}/> Core Information
                                        </h6>
                                        <div className="row g-3">
                                            <div className="col-md-6">
                                                <label className="form-label small text-muted fw-medium">Job Title</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control border-0 shadow-sm"
                                                    value={title}
                                                    onChange={(e) => setTitle(e.target.value)}
                                                    placeholder="e.g. Senior Frontend Engineer"
                                                />
                                            </div>
                                            <div className="col-md-6">
                                                <label className="form-label small text-muted fw-medium">Company</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control border-0 shadow-sm"
                                                    value={company}
                                                    onChange={(e) => setCompany(e.target.value)}
                                                    placeholder="e.g. Tech Corp"
                                                />
                                            </div>
                                        </div>
                                        <div className="d-flex justify-content-end mt-3">
                                            <button 
                                                className="btn btn-dark btn-sm d-flex align-items-center gap-2 px-3" 
                                                onClick={handleUpdateJobDetails}
                                                disabled={savingCore}
                                            >
                                                {savingCore ? (
                                                    <>Saving...</>
                                                ) : (
                                                    <><Save size={14}/> Save Changes</>
                                                )}
                                            </button>
                                        </div>
                                    </div>
                                </div>
                                
                                <hr className="my-0 border-light"/>

                                {/* Part 2: Features List */}
                                <div>
                                    <div className="d-flex justify-content-between align-items-center mb-3">
                                        <h6 className="h6 fw-bold text-dark mb-0">
                                            Requirements & Highlights
                                            <span className="badge bg-light text-muted border ms-2 rounded-pill">{job.features.length}</span>
                                        </h6>
                                    </div>

                                    {/* Scrollable List */}
                                    <div className="border rounded-3 bg-white mb-3 overflow-hidden">
                                        <div className="custom-scroll p-2" style={{maxHeight: '240px', overflowY: 'auto'}}>
                                            {job.features.length === 0 ? (
                                                <div className="text-center py-4 text-muted small opacity-75">
                                                    No features added yet. Add one below.
                                                </div>
                                            ) : (
                                                <ul className="list-unstyled mb-0 d-flex flex-column gap-2">
                                                    {job.features.map(f => {
                                                        const style = getFeatureStyle(f.type);
                                                        return (
                                                            <li key={f.id} className="d-flex align-items-start justify-content-between p-2 rounded hover-bg-light transition-all group">
                                                                <div className="d-flex align-items-start gap-3">
                                                                    <span className={`badge border ${style.badgeClass} text-xs fw-medium d-flex align-items-center gap-1 mt-1`} style={{minWidth: '100px', justifyContent: 'center'}}>
                                                                        <style.icon size={10}/> {style.label}
                                                                    </span>
                                                                    <span className="text-dark small pt-1">{f.description}</span>
                                                                </div>
                                                                <button 
                                                                    className="btn btn-link text-danger p-1 opacity-25 hover-opacity-100 transition-all"
                                                                    onClick={() => handleDeleteFeature(f.id)}
                                                                    title="Remove requirement"
                                                                >
                                                                    <Trash2 size={14} />
                                                                </button>
                                                            </li>
                                                        );
                                                    })}
                                                </ul>
                                            )}
                                        </div>
                                    </div>
                                    
                                    {/* Add New Feature Form */}
                                    <form onSubmit={handleAddFeature} className="bg-light bg-opacity-50 p-3 rounded-3 border border-dashed">
                                        <label className="form-label small fw-bold text-muted text-uppercase">Add New Highlight</label>
                                        <div className="d-flex flex-column gap-2">
                                            <textarea 
                                                className="form-control border-0 shadow-sm"
                                                rows="2"
                                                value={newFeatureText}
                                                onChange={(e) => setNewFeatureText(e.target.value)}
                                                placeholder="Paste a requirement from the job description..."
                                            />
                                            <div className="d-flex gap-2">
                                                <select 
                                                    className="form-select form-select-sm w-auto border-0 shadow-sm"
                                                    value={newFeatureType}
                                                    onChange={(e) => setNewFeatureType(e.target.value)}
                                                >
                                                    <option value="requirement">Requirement</option>
                                                    <option value="responsibility">Responsibility</option>
                                                    <option value="nice_to_have">Nice to Have</option>
                                                    <option value="qualification">Qualification</option>
                                                </select>
                                                <button 
                                                    type="submit" 
                                                    className="btn btn-primary btn-sm d-flex align-items-center gap-2 px-3 flex-grow-1 justify-content-center"
                                                    disabled={addingFeature || !newFeatureText.trim()}
                                                >
                                                    {addingFeature ? 'Adding...' : <><Plus size={16}/> Add Feature</>}
                                                </button>
                                            </div>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        )}
                    </div>
                    
                    <div className="modal-footer border-top-0 bg-light bg-opacity-25">
                        <button type="button" className="btn btn-white border shadow-sm px-4" onClick={onClose}>Done</button>
                    </div>
                </div>
            </div>
            
            {/* Inline Styles for this component specific tweaks */}
            <style>{`
                .hover-bg-light:hover { background-color: #f8fafc; }
                .hover-opacity-100:hover { opacity: 1 !important; }
                .custom-scroll::-webkit-scrollbar { width: 4px; }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 4px; }
            `}</style>
        </div>
    );
};

export default JobEditorModal;