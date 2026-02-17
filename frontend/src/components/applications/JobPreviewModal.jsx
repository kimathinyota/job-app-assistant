// frontend/src/components/applications/JobPreviewModal.jsx
import React from 'react';
import { 
    MapPin, Banknote, Calendar, Globe, FileText, 
    ListChecks, CheckCircle2, Zap, Plus 
} from 'lucide-react';

/**
 * A read-only modal to display the full job description details.
 */
const JobPreviewModal = ({ job, isOpen, onClose }) => {
    if (!isOpen || !job) return null;

    const descriptionContent = job.displayed_description || job.description || "No description content available.";

    // --- GROUPING LOGIC ---
    // 1. Responsibilities (Tasks)
    const responsibilities = job.features?.filter(f => f.type.includes('responsibility')) || [];
    // 2. Requirements (Must Haves)
    const requirements = job.features?.filter(f => f.type.includes('require') || f.type.includes('must')) || [];
    // 3. Skills (Hard & Soft)
    const skills = job.features?.filter(f => f.type.includes('skill') || f.type.includes('tech')) || [];
    // 4. Bonus (Nice to have)
    const bonus = job.features?.filter(f => f.type.includes('nice') || f.type.includes('bonus')) || [];

    // Helper to render a list section
    const renderListSection = (title, items, icon, colorClass) => {
        if (items.length === 0) return null;
        return (
            <div className="mb-4">
                <h6 className={`small fw-bold text-uppercase ${colorClass} mb-2 d-flex align-items-center gap-2`}>
                    {icon} {title}
                </h6>
                <ul className="list-unstyled mb-0 ps-1">
                    {items.map(f => (
                        <li key={f.id} className="mb-2 d-flex align-items-start gap-2">
                            <span className={`mt-1 ${colorClass}`} style={{opacity: 0.5}}>•</span>
                            <span className="text-secondary small" style={{lineHeight: '1.5'}}>{f.description}</span>
                        </li>
                    ))}
                </ul>
            </div>
        );
    };

    return (
        <div
            className="modal fade show"
            style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(2px)' }}
            onClick={onClose}
            tabIndex="-1"
        >
            <div
                className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="modal-content shadow-lg border-0">
                    
                    {/* --- HEADER --- */}
                    <div className="modal-header bg-light border-bottom-0 pb-2">
                        <div className="w-100">
                            <div className="d-flex justify-content-between align-items-start">
                                <div>
                                    <h5 className="modal-title fw-bold text-dark mb-1">{job.title}</h5>
                                    <div className="text-primary fw-medium small d-flex align-items-center gap-2">
                                        <BuildingIcon company={job.company} />
                                        {job.company}
                                    </div>
                                </div>
                                <button type="button" className="btn-close" onClick={onClose} aria-label="Close"></button>
                            </div>
                        </div>
                    </div>

                    {/* --- BODY --- */}
                    <div className="modal-body pt-2">
                        
                        {/* 1. Metadata Pills */}
                        <div className="d-flex flex-wrap gap-2 mb-4 text-xs">
                            {job.location && (
                                <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-1 py-2 px-3">
                                    <MapPin size={14} className="text-primary opacity-75"/> {job.location}
                                </span>
                            )}
                            {job.salary_range && (
                                <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-1 py-2 px-3">
                                    <Banknote size={14} className="text-success opacity-75"/> {job.salary_range}
                                </span>
                            )}
                            {job.application_end_date && (
                                <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-1 py-2 px-3">
                                    <Calendar size={14} className="text-warning opacity-75"/> Apply by {job.application_end_date}
                                </span>
                            )}
                            {job.job_url && (
                                <a 
                                    href={job.job_url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="badge bg-primary bg-opacity-10 text-primary border border-primary border-opacity-25 fw-medium d-flex align-items-center gap-1 py-2 px-3 text-decoration-none hover-shadow"
                                >
                                    <Globe size={14}/> Original Post
                                </a>
                            )}
                        </div>

                        <div className="row">
                            {/* LEFT COLUMN: Main Description */}
                            <div className="col-lg-7 pe-lg-4 border-end-lg">
                                <div className="mb-4">
                                    <h6 className="small fw-bold text-uppercase text-muted mb-3 d-flex align-items-center gap-2 border-bottom pb-2">
                                        <FileText size={16}/> Description
                                    </h6>
                                    <div 
                                        className="text-dark opacity-75" 
                                        style={{ 
                                            whiteSpace: 'pre-wrap', 
                                            fontSize: '0.9rem', 
                                            lineHeight: '1.6',
                                            fontFamily: 'inherit'
                                        }}
                                    >
                                        {descriptionContent}
                                    </div>
                                </div>
                                
                                {/* User Notes */}
                                {job.notes && (
                                    <div className="mt-4 alert alert-warning border-warning bg-warning bg-opacity-10 small">
                                        <strong className="text-warning-emphasis d-block mb-1">Your Notes:</strong>
                                        <span style={{ whiteSpace: 'pre-wrap' }}>{job.notes}</span>
                                    </div>
                                )}
                            </div>

                            {/* RIGHT COLUMN: Structured Extracted Features */}
                            <div className="col-lg-5 ps-lg-4 mt-4 mt-lg-0">
                                <div className="bg-light bg-opacity-50 p-3 rounded-3 border border-light h-100">
                                    <h5 className="fw-bold text-dark mb-3 small text-uppercase border-bottom pb-2">
                                        AI Extraction
                                    </h5>

                                    {/* A. Responsibilities */}
                                    {renderListSection("Key Responsibilities", responsibilities, <ListChecks size={14}/>, "text-primary")}

                                    {/* B. Requirements */}
                                    {renderListSection("Must Haves", requirements, <CheckCircle2 size={14}/>, "text-danger")}

                                    {/* C. Skills (Rendered as Pills) */}
                                    {skills.length > 0 && (
                                        <div className="mb-4">
                                            <h6 className="small fw-bold text-uppercase text-dark mb-2 d-flex align-items-center gap-2">
                                                <Zap size={14}/> Skills
                                            </h6>
                                            <div className="d-flex flex-wrap gap-1">
                                                {skills.map(s => (
                                                    <span key={s.id} className="badge bg-white border text-secondary fw-normal">
                                                        {s.description}
                                                    </span>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* D. Bonus */}
                                    {renderListSection("Nice to Have", bonus, <Plus size={14}/>, "text-success")}

                                    {(!job.features || job.features.length === 0) && (
                                        <div className="text-center text-muted py-4 small fst-italic">
                                            No structured features extracted yet.
                                        </div>
                                    )}
                                </div>
                            </div>
                        </div>

                    </div>

                    {/* --- FOOTER --- */}
                    <div className="modal-footer border-top-0 pt-0">
                        <button type="button" className="btn btn-light border" onClick={onClose}>Close</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

// Helper for Company Icon
const BuildingIcon = ({ company }) => {
    if (!company) return <MapPin size={16}/>;
    return (
        <div className="d-flex align-items-center justify-content-center bg-primary bg-opacity-10 rounded text-primary fw-bold" 
             style={{ width: '24px', height: '24px', fontSize: '12px' }}>
            {company.charAt(0).toUpperCase()}
        </div>
    );
};

export default JobPreviewModal;