// frontend/src/components/applications/JobCard.jsx
import React, { useState } from 'react';
import { 
    MapPin, 
    Banknote, 
    Calendar, 
    ExternalLink, 
    MoreVertical, 
    Building2, 
    CheckCircle2, 
    Trash2,
    Edit,
    FileText,
    Briefcase,
    Star,
    AlertCircle,
    ListChecks,
    Clock, // <--- Added
    Eye    // <--- Added
} from 'lucide-react';

const JobCard = ({ 
    job, 
    cvs = [], 
    defaultCvId, 
    application, 
    onStartApplication, 
    onEdit,
    onDelete,
    onViewDescription // <--- New Prop
}) => {

    const hasApplication = Boolean(application);

    // Local state for the CV dropdown
    const [selectedCvId, setSelectedCvId] = useState(
        hasApplication ? application.base_cv_id : (defaultCvId || '')
    );

    const handleStartClick = () => {
        if (selectedCvId) {
            onStartApplication(job.id, selectedCvId);
        }
    };
    
    const getSelectedCvName = () => {
        if (!application) return "Unknown CV";
        const foundCv = cvs.find(cv => cv.id === application.base_cv_id);
        return foundCv ? foundCv.name : "Unknown CV"; 
    };

    // --- DATE LOGIC ---
    // Prefer the explicitly scraped 'date_closing', fallback to legacy 'application_end_date'
    const closingDate = job.date_closing || job.application_end_date;
    const postedDate = job.date_posted;

    // Helper to style feature types
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
        // Default
        return { 
            label: type ? type.replace(/_/g, ' ') : 'Info', 
            icon: FileText,
            badgeClass: 'text-secondary bg-secondary bg-opacity-10 border-secondary-subtle' 
        };
    };

    return (
        <div className={`card h-100 transition-all hover-shadow-md ${hasApplication ? 'border-success border-2 shadow-sm' : 'border shadow-sm'}`}>
            <style>
                {`
                .text-xs { font-size: 0.75rem; }
                .hover-shadow-md:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important; }
                
                /* Custom scrollbar for the features list */
                .custom-scroll::-webkit-scrollbar { width: 4px; }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #e2e8f0; border-radius: 4px; }
                `}
            </style>
            
            {/* --- Header: Company & Actions --- */}
            <div className="card-body p-4 pb-0 d-flex flex-column">
                <div className="d-flex justify-content-between align-items-start mb-4">
                    <div className="d-flex gap-3 align-items-center overflow-hidden">
                        {/* Company Logo Placeholder */}
                        <div className="rounded-3 d-flex align-items-center justify-content-center bg-light text-primary fw-bold border flex-shrink-0" 
                             style={{ width: '52px', height: '52px', fontSize: '22px' }}>
                            {job.company ? job.company.charAt(0).toUpperCase() : <Building2 size={24}/>}
                        </div>
                        <div className="min-w-0">
                            <h5 className="card-title fw-bold text-dark mb-1 text-truncate" title={job.title}>
                                {job.title}
                            </h5>
                            <div className="text-muted small fw-medium text-truncate">{job.company}</div>
                        </div>
                    </div>

                    {/* Dropdown Menu */}
                    <div className="dropdown">
                        <button className="btn btn-link text-muted p-0 opacity-50 hover-opacity-100" type="button" data-bs-toggle="dropdown" aria-expanded="false">
                            <MoreVertical size={20} />
                        </button>
                        <ul className="dropdown-menu dropdown-menu-end shadow-sm border-0">
                            <li>
                                <button
                                    className="dropdown-item d-flex align-items-center gap-2 small"
                                    onClick={onEdit}
                                    disabled={hasApplication}
                                >
                                    <Edit size={14}/> Edit Details
                                </button>
                            </li>
                            <li><hr className="dropdown-divider"/></li>
                            <li>
                                <button
                                    className="dropdown-item text-danger d-flex align-items-center gap-2 small"
                                    onClick={onDelete}
                                >
                                    <Trash2 size={14}/> Delete Job
                                </button>
                            </li>
                        </ul>
                    </div>
                </div>

                {/* --- Tag Visuals --- */}
                <div className="d-flex flex-wrap gap-2 mb-4">
                    {job.location && (
                        <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-2 py-2 px-3 rounded-pill">
                            <MapPin size={14} className="text-primary opacity-75" /> {job.location}
                        </span>
                    )}
                    {job.salary_range && (
                        <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-2 py-2 px-3 rounded-pill">
                            <Banknote size={14} className="text-success opacity-75" /> {job.salary_range}
                        </span>
                    )}
                    
                    {/* UPDATED DATE LOGIC */}
                    {postedDate && (
                        <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-2 py-2 px-3 rounded-pill" title="Date Posted">
                            <Clock size={14} className="text-info opacity-75" /> 
                            <span className="text-muted opacity-75 me-1">Posted:</span>
                            {new Date(postedDate).toLocaleDateString(undefined, {month:'short', day:'numeric'})}
                        </span>
                    )}
                    {closingDate && (
                        <span className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-2 py-2 px-3 rounded-pill" title="Closing Date">
                            <Calendar size={14} className="text-warning opacity-75" /> 
                            <span className="text-muted opacity-75 me-1">Due:</span>
                            {new Date(closingDate).toLocaleDateString(undefined, {month:'short', day:'numeric'})}
                        </span>
                    )}
                </div>

                {/* --- Features Snippet (Fixed Types) --- */}
                <div className="flex-grow-1 mb-3">
                    <div className="d-flex justify-content-between align-items-center mb-2">
                        <h6 className="text-xs fw-bold text-uppercase text-muted mb-0 d-flex align-items-center gap-1">
                            <FileText size={12}/> Key Highlights
                        </h6>
                        {/* VIEW DESCRIPTION BUTTON */}
                        {(job.displayed_description || job.description) && (
                            <button 
                                onClick={() => onViewDescription(job)}
                                className="btn btn-link p-0 text-primary text-xs text-decoration-none d-flex align-items-center gap-1 hover-underline"
                            >
                                <Eye size={12} /> View Desc
                            </button>
                        )}
                    </div>

                    {job.features && job.features.length > 0 ? (
                        <div className="bg-light bg-opacity-50 rounded-3 p-3 custom-scroll border border-dashed" style={{ maxHeight: '140px', overflowY: 'auto' }}>
                            <ul className="list-unstyled mb-0">
                                {job.features.map(feature => {
                                    const style = getFeatureStyle(feature.type);
                                    return (
                                        <li key={feature.id} className="mb-2 d-flex align-items-start gap-2">
                                            {/* Type Badge */}
                                            <span className={`badge border ${style.badgeClass} text-xs fw-medium d-flex align-items-center gap-1 flex-shrink-0`} 
                                                  style={{marginTop: '1px', minWidth: '85px', justifyContent: 'center'}}>
                                                {style.label}
                                            </span>
                                            {/* Description */}
                                            <span className="text-muted opacity-75 small" style={{lineHeight: '1.4'}}>
                                                {feature.description}
                                            </span>
                                        </li>
                                    );
                                })}
                            </ul>
                        </div>
                    ) : (
                        <div className="p-4 text-center text-muted small bg-light bg-opacity-25 rounded-3 border border-dashed">
                            <span className="opacity-50">No highlights added.</span>
                        </div>
                    )}
                </div>

                 {/* Link */}
                 {job.job_url && (
                    <div className="mb-3 mt-auto">
                         <a href={job.job_url} target="_blank" rel="noopener noreferrer" className="small text-primary d-flex align-items-center gap-1 text-decoration-none hover-underline">
                            <ExternalLink size={14}/> View Original Job Post
                         </a>
                    </div>
                )}
            </div>

            {/* --- Footer: Action Area --- */}
            <div className={`card-footer p-3 border-top-0 ${hasApplication ? 'bg-success bg-opacity-10' : 'bg-light'}`}>
                {hasApplication ? (
                    <div className="d-flex justify-content-between align-items-center">
                        <div className="d-flex align-items-center gap-2 text-success fw-bold small">
                            <CheckCircle2 size={16} /> Application Started
                        </div>
                        <div className="badge bg-white text-success border border-success-subtle fw-normal">
                             CV: {getSelectedCvName()}
                        </div>
                    </div>
                ) : (
                    <div className="d-flex gap-2 align-items-center">
                        <div className="flex-grow-1 position-relative">
                            <Briefcase size={14} className="text-muted position-absolute top-50 start-0 translate-middle-y ms-3"/>
                            <select
                                className="form-select form-select-sm border-0 shadow-none bg-white ps-5"
                                value={selectedCvId || ''} 
                                onChange={(e) => setSelectedCvId(e.target.value)}
                                disabled={cvs.length === 0}
                                style={{fontSize: '0.9rem'}}
                            >
                                <option value="" disabled>Select Base CV...</option>
                                {cvs.map(cv => (
                                    <option key={cv.id} value={cv.id}>{cv.name}</option>
                                ))}
                            </select>
                        </div>
                        <button 
                            className="btn btn-sm btn-primary px-4 fw-medium shadow-sm d-flex align-items-center gap-1"
                            onClick={handleStartClick}
                            disabled={!selectedCvId}
                        >
                            Start
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
};

export default JobCard;