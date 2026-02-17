import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom'; // [1] Import Navigation Hook
import { 
    MapPin, Banknote, Calendar, MoreVertical, Building2, 
    CheckCircle2, Trash2, Edit, TrendingUp, Sparkles, 
    AlertTriangle, Briefcase, Eye, Clock, XCircle, 
    Loader2, RefreshCw, ExternalLink
} from 'lucide-react';

import { fetchMatchPreview } from '../../api/applicationClient';

const JobCard = ({ 
    job, 
    cvs = [], 
    defaultCvId, 
    application, 
    onStartApplication, 
    onViewApplication,
    onEdit,
    onDelete,
    onViewDescription, // This opens the Modal
    matchScore = 0,
    badges = [] 
}) => {
    const navigate = useNavigate(); // [2] Init Hook
    const hasApplication = Boolean(application);
    const [selectedCvId, setSelectedCvId] = useState(
        hasApplication ? application.base_cv_id : (defaultCvId || '')
    );

    // --- PREVIEW STATE ---
    const [previewData, setPreviewData] = useState(null); 
    const [isLoadingPreview, setIsLoadingPreview] = useState(false);

    const activeScore = previewData ? previewData.score : matchScore;
    const activeBadges = previewData ? previewData.badges : badges;
    const isPreviewMode = !!previewData; 

    // --- EFFECT: LIVE PREVIEW ---
    useEffect(() => {
        if (hasApplication || !selectedCvId) return;
        if (selectedCvId === defaultCvId) { setPreviewData(null); return; }

        let isMounted = true;
        setIsLoadingPreview(true);

        fetchMatchPreview(job.id, selectedCvId)
            .then(data => { if (isMounted) setPreviewData(data); })
            .catch(err => console.error(err))
            .finally(() => { if (isMounted) setIsLoadingPreview(false); });

        return () => { isMounted = false; };
    }, [selectedCvId, defaultCvId, hasApplication, job.id]);

    // --- HANDLERS ---
    
    // [3] The Main Card Click -> Goes to Full Page
    const handleCardClick = (e) => {
        // We do NOT stop propagation here; this is the default action
        navigate(`/job/${job.id}`);
    };

    const handleStartClick = (e) => { 
        e.stopPropagation(); // Stop card click
        if (selectedCvId) onStartApplication(job.id, selectedCvId); 
    };
    
    const getSelectedCvName = () => {
        if (!application) return "Unknown CV";
        const foundCv = cvs.find(cv => cv.id === application.base_cv_id);
        return foundCv ? foundCv.name : "Unknown CV"; 
    };

    // --- 1. GRADE / THEME SYSTEM ---
    const getGradeTheme = (score) => {
        if (isPreviewMode) return { border: 'border-info', text: 'text-info', bg: 'bg-info-subtle', icon: RefreshCw, label: 'Simulated' };
        if (!score) return { border: 'border-secondary', text: 'text-secondary', bg: 'bg-light', icon: Briefcase, label: 'Unscored' };
        if (score >= 85) return { border: 'border-success', text: 'text-success', bg: 'bg-success-subtle', icon: Sparkles, label: 'Excellent' };
        if (score >= 60) return { border: 'border-primary', text: 'text-primary', bg: 'bg-primary-subtle', icon: TrendingUp, label: 'Good' };
        if (score >= 40) return { border: 'border-warning', text: 'text-warning', bg: 'bg-warning-subtle', icon: AlertTriangle, label: 'Weak' };
        return { border: 'border-danger', text: 'text-danger', bg: 'bg-danger-subtle', icon: XCircle, label: 'Poor' };
    };

    // --- 2. HIGHLIGHT PILL STYLES ---
    const getFeatureStyle = (type) => {
        const t = type ? type.toLowerCase() : '';
        if (t.includes('responsibility')) return { label: 'Task', badgeClass: 'bg-primary-subtle text-primary border-primary-subtle' };
        if (t.includes('require') || t.includes('must')) return { label: 'Must Have', badgeClass: 'bg-danger-subtle text-danger border-danger-subtle' };
        if (t.includes('hard_skill') || t.includes('tech')) return { label: 'Skill', badgeClass: 'bg-dark-subtle text-dark border-dark-subtle' };
        if (t.includes('soft_skill')) return { label: 'Soft Skill', badgeClass: 'bg-info-subtle text-info border-info-subtle' };
        if (t.includes('nice') || t.includes('bonus')) return { label: 'Bonus', badgeClass: 'bg-success-subtle text-success border-success-subtle' };
        return { label: t.replace('_', ' '), badgeClass: 'bg-secondary-subtle text-secondary border-secondary-subtle text-capitalize' };
    };

    const theme = getGradeTheme(activeScore);
    const ScoreIcon = theme.icon;
    const closingDate = job.date_closing || job.application_end_date;

    const displayBadges = (activeBadges && activeBadges.length > 0) 
        ? activeBadges 
        : (activeScore > 0 ? [theme.label] : []);

    const sortedFeatures = job.features ? [...job.features].sort((a, b) => {
        const priority = { 'responsibility': 1, 'requirement': 2, 'hard_skill': 3, 'soft_skill': 4, 'nice_to_have': 5, 'bonus': 5 };
        const getP = (type) => {
            const t = type.toLowerCase();
            for (const key in priority) if (t.includes(key)) return priority[key];
            return 99;
        };
        return getP(a.type) - getP(b.type);
    }) : [];

    return (
        // [4] Card is now a clickable container
        <div 
            className={`card h-100 transition-all hover-shadow-md border-2 cursor-pointer ${theme.border} ${hasApplication ? 'shadow-sm' : ''}`}
            onClick={handleCardClick}
        >
            <style>
                {`
                .text-xs { font-size: 0.75rem; }
                .hover-shadow-md:hover { box-shadow: 0 10px 20px rgba(0,0,0,0.08) !important; transform: translateY(-3px); }
                .custom-scroll::-webkit-scrollbar { width: 4px; }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #e2e8f0; border-radius: 4px; }
                .safe-pill { max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
                .spin-slow { animation: spin 2s linear infinite; }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                /* Critical: Ensure Dropdown works on top of clickable card */
                .dropdown-menu { z-index: 1050; } 
                `}
            </style>
            
            <div className="card-body p-4 pb-0 d-flex flex-column">
                
                {/* --- HEADER --- */}
                <div className="d-flex justify-content-between align-items-start mb-2 w-100">
                    <div className="d-flex gap-3 align-items-center overflow-hidden flex-grow-1" style={{ minWidth: 0 }}> 
                        <div className="rounded-3 d-flex align-items-center justify-content-center bg-light border flex-shrink-0 text-primary fw-bold fs-5 shadow-sm" 
                             style={{ width: '48px', height: '48px' }}>
                            {job.company ? job.company.charAt(0).toUpperCase() : <Building2 size={24}/>}
                        </div>
                        <div className="min-w-0">
                            <h5 className="card-title fw-bold text-dark mb-0 text-truncate" title={job.title}>{job.title}</h5>
                            <div className="text-muted small fw-medium text-truncate">{job.company}</div>
                        </div>
                    </div>

                    <div className="flex-shrink-0 ms-2 d-flex align-items-center gap-1">
                        
                        {/* External Link (Stop Propagation) */}
                        {job.job_url && (
                            <a 
                                href={job.job_url} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="btn btn-link text-muted p-1 hover-bg-light rounded-circle"
                                title="Open Original Post"
                                onClick={(e) => e.stopPropagation()} 
                            >
                                <ExternalLink size={16} />
                            </a>
                        )}

                        {/* Dropdown Menu (Stop Propagation) */}
                        <div className="dropdown" onClick={(e) => e.stopPropagation()}>
                            <button className="btn btn-link text-muted p-0 opacity-50 hover-opacity-100" type="button" data-bs-toggle="dropdown">
                                <MoreVertical size={20} />
                            </button>
                            <ul className="dropdown-menu dropdown-menu-end shadow-sm border-0">
                                {/* NOTE: Edit opens the Modal (via parent handler), not the page */}
                                <li>
                                    <button className="dropdown-item small" onClick={() => onEdit()}>
                                        <Edit size={14} className="me-2"/> Edit Details
                                    </button>
                                </li>
                                <li><hr className="dropdown-divider"/></li>
                                <li>
                                    <button className="dropdown-item small text-danger" onClick={() => onDelete()}>
                                        <Trash2 size={14} className="me-2"/> Delete Job
                                    </button>
                                </li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* --- METADATA --- */}
                <div className="d-flex flex-wrap align-items-center gap-2 mb-3 text-xs fw-medium">
                    {hasApplication && (
                        <span className="badge bg-success-subtle text-success border border-success-subtle d-flex align-items-center gap-1 py-1 px-2 rounded-pill">
                            <CheckCircle2 size={12}/> Applied
                        </span>
                    )}
                    {job.location && (
                        <span className="d-flex align-items-center gap-1 text-muted bg-light px-2 py-1 rounded">
                            <MapPin size={12}/> {job.location}
                        </span>
                    )}
                    {job.salary_range && (
                        <span className="d-flex align-items-center gap-1 text-success bg-light px-2 py-1 rounded">
                            <Banknote size={12}/> {job.salary_range}
                        </span>
                    )}
                    {closingDate && (
                        <span className="d-flex align-items-center gap-1 text-warning-emphasis bg-light px-2 py-1 rounded">
                            <Clock size={12}/> Due {new Date(closingDate).toLocaleDateString(undefined, {month:'short', day:'numeric'})}
                        </span>
                    )}
                </div>

                {/* --- MATCH INSIGHT --- */}
                <div className={`rounded-3 p-3 mb-3 d-flex align-items-center gap-3 border ${theme.bg} ${theme.border} bg-opacity-25`}>
                    <div className="d-flex align-items-center gap-3 pe-3 border-end border-dark border-opacity-10">
                        {isLoadingPreview ? (
                            <div className={`d-flex align-items-center justify-content-center rounded-circle bg-white shadow-sm ${theme.text}`} style={{width: '42px', height: '42px'}}><Loader2 size={20} className="spin-slow" /></div>
                        ) : (
                            <div className={`d-flex align-items-center justify-content-center rounded-circle bg-white shadow-sm ${theme.text}`} style={{width: '42px', height: '42px'}}><ScoreIcon size={20} strokeWidth={2.5} /></div>
                        )}
                        <div className={`d-flex flex-column ${isLoadingPreview ? 'opacity-50' : ''}`} style={{lineHeight: 1}}>
                            <span className={`fw-bold fs-5 ${theme.text}`}>{Math.round(activeScore)}%</span>
                            <span className="text-uppercase fw-bold opacity-75" style={{fontSize: '0.65rem'}}>{theme.label}</span>
                        </div>
                    </div>
                    <div className={`d-flex flex-wrap gap-1 align-items-center ${isLoadingPreview ? 'opacity-50' : ''}`}>
                        {displayBadges.map((badge, idx) => {
                            if (!badge || typeof badge !== 'string') return null;
                            return <span key={idx} className="badge bg-white text-secondary shadow-sm fw-normal border border-opacity-10 px-2 text-wrap text-start">{badge}</span>;
                        })}
                    </div>
                </div>

                {/* --- HIGHLIGHTS --- */}
                <div className="flex-grow-1 mb-3 pt-2 border-top border-light">
                    <div className="d-flex justify-content-between align-items-center mb-2">
                        <span className="text-xs fw-bold text-secondary text-uppercase tracking-wide">Key Highlights</span>
                        {(job.displayed_description || job.description) && (
                            <button 
                                type="button" 
                                onClick={(e) => { 
                                    e.stopPropagation(); // [5] Stop Prop: Open Modal, don't navigate
                                    onViewDescription(job); 
                                }}
                                className="btn btn-link p-0 text-primary text-xs text-decoration-none d-flex align-items-center gap-1 hover-underline"
                            >
                                <Eye size={12} /> View Desc
                            </button>
                        )}
                    </div>
                    
                    {sortedFeatures.length > 0 ? (
                        <div className="bg-light bg-opacity-50 rounded-3 p-3 custom-scroll border border-dashed" style={{ maxHeight: '140px', overflowY: 'auto' }}>
                            <ul className="list-unstyled mb-0">
                                {sortedFeatures.map(feature => {
                                    const style = getFeatureStyle(feature.type);
                                    return (
                                        <li key={feature.id} className="mb-2 d-flex align-items-start gap-2">
                                            <span className={`badge border ${style.badgeClass} text-xs fw-medium d-flex align-items-center gap-1 flex-shrink-0`} style={{marginTop: '1px', minWidth: '85px', justifyContent: 'center'}}>
                                                {style.label}
                                            </span>
                                            <span className="text-muted opacity-75 small text-truncate d-block" style={{lineHeight: '1.4', whiteSpace: 'normal'}}>
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
            </div>

            {/* --- FOOTER --- */}
            <div className={`card-footer p-3 border-top-0 ${hasApplication ? 'bg-success-subtle' : 'bg-white border-top'}`}>
                {hasApplication ? (
                    <div className="d-flex justify-content-between align-items-center">
                        <div className="text-success text-xs fw-medium">
                            Applying with: <strong>{getSelectedCvName()}</strong>
                        </div>
                        <button 
                            type="button"
                            className="btn btn-sm btn-white border border-success-subtle text-success shadow-sm fw-medium"
                            onClick={(e) => {
                                e.stopPropagation(); // [6] Stop Prop: Go to Application Dashboard
                                onViewApplication(application.id); 
                            }}
                        >
                            View Application
                        </button>
                    </div>
                ) : (
                    <div className="d-flex gap-2 align-items-center">
                        <div className="flex-grow-1 position-relative">
                            <div className="position-absolute top-50 start-0 translate-middle-y ms-3 pointer-events-none">
                                {isPreviewMode ? <RefreshCw size={14} className="text-primary"/> : <Briefcase size={14} className="text-muted"/>}
                            </div>
                            <select
                                className={`form-select form-select-sm shadow-none ps-5 ${isPreviewMode ? 'border-primary text-primary fw-medium bg-primary-subtle' : 'border-0 bg-light'}`}
                                value={selectedCvId || ''} 
                                onChange={(e) => setSelectedCvId(e.target.value)}
                                onClick={(e) => e.stopPropagation()} // [7] Stop Prop: Selecting Dropdown
                                disabled={cvs.length === 0}
                                style={{fontSize: '0.9rem', cursor: 'pointer'}}
                            >
                                <option value="" disabled>Select Base CV...</option>
                                {cvs.map(cv => <option key={cv.id} value={cv.id}>{cv.name}</option>)}
                            </select>
                        </div>
                        <button 
                            className="btn btn-sm btn-primary px-4 fw-medium shadow-sm d-flex align-items-center gap-1"
                            onClick={handleStartClick} // Already has stopPropagation
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