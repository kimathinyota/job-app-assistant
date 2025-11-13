// frontend/src/components/applications/JobModal.jsx
import React, { useState, useEffect } from 'react';
import { 
    Save, 
    Plus, 
    Trash2, 
    Briefcase, 
    Building2, 
    AlertCircle, 
    Star, 
    ListChecks, 
    FileText,
    MapPin,
    Banknote,
    Calendar,
    Link as LinkIcon,
    AlignLeft
} from 'lucide-react';
import { 
    fetchJobDetails, 
    upsertJob 
} from '../../api/applicationClient';

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
    
    // Feature State
    const [features, setFeatures] = useState([]);
    const [newFeatureText, setNewFeatureText] = useState('');
    const [newFeatureType, setNewFeatureType] = useState('requirement');

    // Helper for feature styling (Consistent with JobCard)
    const getFeatureStyle = (type) => {
        const t = type ? type.toLowerCase() : '';
        if (t.includes('require') || t.includes('must')) {
            return { 
                label: 'Must Have', 
                icon: AlertCircle, 
                badgeClass: 'text-danger bg-danger bg-opacity-10 border-danger-subtle' 
            };
        }
        if (t.includes('nice') || t.includes('bonus') || t.includes('plus')) {
            return { 
                label: 'Bonus', 
                icon: Star, 
                badgeClass: 'text-success bg-success bg-opacity-10 border-success-subtle' 
            };
        }
        if (t.includes('responsibility') || t.includes('task')) {
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

    // Effect to load data
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
                        onClose();
                    })
                    .finally(() => setIsFetching(false));
            } else {
                // --- ADD MODE ---
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

    // --- Handlers ---
    const handleAddNewFeature = (e) => {
        e.preventDefault();
        if (!newFeatureText.trim()) return;
        
        const newFeature = {
            id: `temp_${Date.now()}`, 
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

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);

        const jobPayload = {
            id: jobId, 
            title,
            company,
            job_url,
            application_end_date,
            location,
            salary_range,
            notes,
            features: features.map(f => ({
                id: String(f.id).startsWith('temp_') ? null : f.id,
                description: f.description,
                type: f.type
            }))
        };

        try {
            await upsertJob(jobPayload);
            onJobUpdated();
            handleClose();
        } catch (err) {
            alert(`Failed to ${jobId ? 'update' : 'create'} job.`);
        } finally {
            setIsLoading(false);
        }
    };

    const handleClose = () => {
        setIsLoading(false);
        setIsFetching(false);
        onClose();
    };

    if (!isOpen) return null;

    return (
        <div className="modal show d-block" style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }} tabIndex="-1">
            <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <form className="modal-content border-0 shadow-lg rounded-xl overflow-hidden" onSubmit={handleSubmit}>
                    
                    {/* Header */}
                    <div className="modal-header border-bottom-0 bg-white pb-0">
                        <div className="d-flex align-items-center gap-3">
                            <div className="bg-primary bg-opacity-10 p-2 rounded-circle text-primary">
                                <Briefcase size={20} />
                            </div>
                            <h5 className="modal-title fw-bold text-dark">
                                {initialJobId ? 'Edit Job Details' : 'Add New Job'}
                            </h5>
                        </div>
                        <button type="button" className="btn-close" onClick={handleClose}></button>
                    </div>
                    
                    <div className="modal-body p-4">
                        {isFetching ? (
                            <div className="text-center py-5">
                                <div className="spinner-border text-primary mb-3" role="status"></div>
                                <p className="text-muted">Loading job info...</p>
                            </div>
                        ) : (
                            <div className="d-flex flex-column gap-4">
                                
                                {/* 1. Core Identity Card */}
                                <div className="card border-0 shadow-sm bg-light bg-opacity-50">
                                    <div className="card-body p-3">
                                        <h6 className="text-uppercase text-muted small fw-bold mb-3 d-flex align-items-center gap-2">
                                            <Building2 size={14}/> Core Identity
                                        </h6>
                                        <div className="row g-3">
                                            <div className="col-md-6">
                                                <label className="form-label small text-muted fw-medium">Job Title*</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control border-0 shadow-sm"
                                                    value={title}
                                                    onChange={(e) => setTitle(e.target.value)}
                                                    required
                                                    placeholder="e.g. Senior Product Designer"
                                                />
                                            </div>
                                            <div className="col-md-6">
                                                <label className="form-label small text-muted fw-medium">Company Name*</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control border-0 shadow-sm"
                                                    value={company}
                                                    onChange={(e) => setCompany(e.target.value)}
                                                    required
                                                    placeholder="e.g. Acme Corp"
                                                />
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* 2. Logistics Grid */}
                                <div className="row g-3">
                                    <div className="col-md-6">
                                        <label className="form-label small text-muted fw-medium d-flex align-items-center gap-1">
                                            <MapPin size={12}/> Location
                                        </label>
                                        <input type="text" className="form-control border shadow-sm" value={location}
                                            onChange={(e) => setLocation(e.target.value)} placeholder="e.g. Remote / London" />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label small text-muted fw-medium d-flex align-items-center gap-1">
                                            <Banknote size={12}/> Salary Range
                                        </label>
                                        <input type="text" className="form-control border shadow-sm" value={salary_range}
                                            onChange={(e) => setSalaryRange(e.target.value)} placeholder="e.g. $120k - $150k" />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label small text-muted fw-medium d-flex align-items-center gap-1">
                                            <LinkIcon size={12}/> Job Posting URL
                                        </label>
                                        <input type="url" className="form-control border shadow-sm" value={job_url}
                                            onChange={(e) => setJobUrl(e.target.value)} placeholder="https://linkedin.com/jobs..." />
                                    </div>
                                    <div className="col-md-6">
                                        <label className="form-label small text-muted fw-medium d-flex align-items-center gap-1">
                                            <Calendar size={12}/> Apply By
                                        </label>
                                        <input type="date" className="form-control border shadow-sm" value={application_end_date}
                                            onChange={(e) => setApplicationEndDate(e.target.value)} />
                                    </div>
                                </div>

                                {/* 3. Notes Section */}
                                <div>
                                    <label className="form-label small text-muted fw-medium d-flex align-items-center gap-1">
                                        <AlignLeft size={12}/> Private Notes
                                    </label>
                                    <textarea className="form-control border shadow-sm" rows="2" value={notes}
                                        onChange={(e) => setNotes(e.target.value)} placeholder="Thoughts on the role, referral details, etc." />
                                </div>

                                <hr className="my-0 border-light"/>

                                {/* 4. Highlights / Features */}
                                <div>
                                    <div className="d-flex justify-content-between align-items-center mb-3">
                                        <h6 className="h6 fw-bold text-dark mb-0">
                                            Highlights & Requirements
                                            <span className="badge bg-light text-muted border ms-2 rounded-pill">{features.length}</span>
                                        </h6>
                                    </div>

                                    {/* Feature List */}
                                    <div className="border rounded-3 bg-white mb-3 overflow-hidden">
                                        <div className="custom-scroll p-2" style={{maxHeight: '200px', overflowY: 'auto'}}>
                                            {features.length === 0 ? (
                                                <div className="text-center py-4 text-muted small opacity-75">
                                                    No highlights added yet. Paste requirements below.
                                                </div>
                                            ) : (
                                                <ul className="list-unstyled mb-0 d-flex flex-column gap-2">
                                                    {features.map(f => {
                                                        const style = getFeatureStyle(f.type);
                                                        return (
                                                            <li key={f.id} className="d-flex align-items-start justify-content-between p-2 rounded hover-bg-light transition-all">
                                                                <div className="d-flex align-items-start gap-3">
                                                                    <span className={`badge border ${style.badgeClass} text-xs fw-medium d-flex align-items-center gap-1 mt-1`} 
                                                                          style={{minWidth: '90px', justifyContent: 'center'}}>
                                                                        <style.icon size={10}/> {style.label}
                                                                    </span>
                                                                    <span className="text-dark small pt-1">{f.description}</span>
                                                                </div>
                                                                <button 
                                                                    type="button"
                                                                    className="btn btn-link text-danger p-1 opacity-25 hover-opacity-100"
                                                                    onClick={() => handleDeleteFeature(f.id)}
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

                                    {/* Add Feature Form */}
                                    <div className="bg-light bg-opacity-50 p-3 rounded-3 border border-dashed">
                                        <div className="d-flex flex-column gap-2">
                                            <textarea 
                                                className="form-control border-0 shadow-sm"
                                                rows="2"
                                                value={newFeatureText}
                                                onChange={(e) => setNewFeatureText(e.target.value)}
                                                placeholder="Paste a requirement (e.g. '5+ years of Python')..."
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
                                                    type="button" 
                                                    className="btn btn-primary btn-sm d-flex align-items-center gap-2 px-3 flex-grow-1 justify-content-center"
                                                    onClick={handleAddNewFeature}
                                                    disabled={!newFeatureText.trim()}
                                                >
                                                    <Plus size={16}/> Add
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                    
                    {/* Footer */}
                    <div className="modal-footer border-top-0 bg-light bg-opacity-25">
                        <button type="button" className="btn btn-white border shadow-sm px-4" onClick={handleClose}>Cancel</button>
                        <button type="submit" className="btn btn-primary px-4 shadow-sm d-flex align-items-center gap-2" disabled={isLoading || isFetching}>
                            {isLoading ? "Saving..." : <><Save size={16}/> {jobId ? "Save Changes" : "Create Job"}</>}
                        </button>
                    </div>
                </form>
            </div>
            
            <style>{`
                .hover-bg-light:hover { background-color: #f8fafc; }
                .hover-opacity-100:hover { opacity: 1 !important; }
                .custom-scroll::-webkit-scrollbar { width: 4px; }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 4px; }
            `}</style>
        </div>
    );
};

export default JobModal;