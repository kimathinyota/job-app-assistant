// frontend/src/components/applications/ApplicationDashboard.jsx
import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    Briefcase, FileText, ShieldCheck, Plus, 
    ChevronRight, MapPin, Calendar, ArrowRight, 
    Loader2, CheckCircle2, Microscope,
    AlertTriangle, GraduationCap, Trophy,
    Download, Sparkles, FileCheck
} from 'lucide-react';
import { 
    fetchApplicationDetails, fetchJobDetails, 
    createCoverLetter, updateApplication, fetchCoverLetterDetails,
    fetchForensicAnalysis, triggerApplicationAnalysis 
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient';
import { getCurrentUser } from '../../api/authClient'; 
import { API_BASE_URL } from '../../api/client';
import { useJobSocket } from '../../hooks/useJobSocket'; 
import NewDocumentModal from './NewDocumentModal';

const ApplicationDashboard = () => {
    const { applicationId } = useParams();
    const navigate = useNavigate();
    
    // --- STATE ---
    const [user, setUser] = useState(null);
    
    // 1. Critical Data (Must load before page renders)
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    
    // 2. Heavy Data (Loads lazily, doesn't block page)
    const [forensics, setForensics] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    
    // UI State
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isExportMenuOpen, setIsExportMenuOpen] = useState(false);
    
    const exportMenuRef = useRef(null);

    // --- 1. LOAD USER & SOCKET ---
    useEffect(() => {
        getCurrentUser().then(res => setUser(res.data || res)).catch(console.error);
    }, []);

    // WebSocket Listener: Updates forensics when background task finishes
    useJobSocket(user?.id, (event) => {
        if (event.type === 'APP_SCORED' && event.payload.app_id === applicationId) {
            console.log("Analysis Ready via Socket!");
            // Re-fetch full analysis to get charts/gaps
            fetchForensicAnalysis(applicationId)
                .then(res => setForensics(res.data))
                .catch(err => console.error("Failed to reload forensics", err))
                .finally(() => setIsAnalyzing(false));
        }
    });

    // --- 2. CRITICAL DATA LOAD (Fast) ---
    useEffect(() => {
        const loadCritical = async () => {
            try {
                // Fetch App Wrapper
                const appRes = await fetchApplicationDetails(applicationId);
                let app = appRes.data;
                
                // Parallel fetch Job & CV
                const [jobRes, cvData] = await Promise.all([
                    fetchJobDetails(app.job_id).catch(e => ({ data: null })),
                    fetchCVDetails(app.base_cv_id).catch(e => null), 
                ]);

                // Fetch Supporting Docs
                let docs = [];
                if (app.supporting_document_ids && app.supporting_document_ids.length > 0) {
                    for (const docId of app.supporting_document_ids) {
                         try {
                             const dRes = await fetchCoverLetterDetails(docId);
                             if (dRes.data) docs.push(dRes.data);
                         } catch (e) { console.warn("Failed to load doc", docId); }
                    }
                }

                // Smart Start: Create Cover Letter if missing (and app not locked)
                if (docs.length === 0 && !app.is_locked) {
                    try {
                        const res = await createCoverLetter(app.job_id, app.base_cv_id, app.mapping_id, "Cover Letter");
                        const newDoc = res.data;
                        docs.push(newDoc);
                        
                        const newIds = [...(app.supporting_document_ids || []), newDoc.id];
                        await updateApplication(app.id, { supporting_document_ids: newIds });
                        app = { ...app, supporting_document_ids: newIds };
                    } catch (e) {
                        console.error("Smart Start failed", e);
                    }
                }

                // Set Critical Data
                setData({ 
                    app, 
                    job: jobRes.data || {}, 
                    cv: cvData || { name: "Unknown CV" }, 
                    docs 
                });

            } catch (err) {
                console.error("Critical Load Failed", err);
            } finally {
                // UNBLOCK THE UI IMMEDIATELY
                setIsLoading(false);
            }
        };
        loadCritical();

        // Click outside listener for export menu
        const handleClickOutside = (event) => {
            if (exportMenuRef.current && !exportMenuRef.current.contains(event.target)) {
                setIsExportMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);

    }, [applicationId]);

    // --- 3. FORENSICS LOAD (Lazy / Async) ---
    useEffect(() => {
        // Wait for critical data first
        if (!data) return;

        const loadForensics = async () => {
            // If we already have forensics (from socket update), don't re-fetch
            if (forensics) return;

            setIsAnalyzing(true); // Show spinner ONLY on the card
            
            try {
                const res = await fetchForensicAnalysis(applicationId);
                const analysis = res.data;

                // Check validity (Has score?)
                if (analysis && analysis.stats && analysis.stats.overall_match_score > 0) {
                    setForensics(analysis);
                    setIsAnalyzing(false);
                } else {
                    // Empty or New? Trigger Background Task
                    console.log("Analysis missing or empty. Triggering background inference...");
                    triggerApplicationAnalysis(applicationId).catch(console.error);
                    // Keep isAnalyzing=true until Socket fires
                }
            } catch (err) {
                // 404 means no analysis exists yet
                console.log("Analysis not found (404). Triggering background inference...");
                triggerApplicationAnalysis(applicationId).catch(console.error);
            }
        };
        
        loadForensics();
    }, [data, applicationId]); // Runs once 'data' is loaded

    // --- HANDLERS ---

    const handleCreateDocClick = () => setIsModalOpen(true);

    const handleCreateDocSubmit = async (name, layout, questions) => {
        try {
            const res = await createCoverLetter(data.app.job_id, data.app.base_cv_id, data.app.mapping_id, name);
            const newDoc = res.data;
            
            const newIds = [...(data.app.supporting_document_ids || []), newDoc.id];
            await updateApplication(data.app.id, { supporting_document_ids: newIds });
            
            // Update local state to show new doc immediately
            setData(prev => ({
                ...prev,
                app: { ...prev.app, supporting_document_ids: newIds },
                docs: [...prev.docs, newDoc]
            }));
            
            setIsModalOpen(false);
            navigate(`/application/${applicationId}/doc/${newDoc.id}`);
            
        } catch (err) {
            console.error(err);
            alert("Failed to create document.");
        }
    };

    const handleSubmitApplication = async () => {
        if (!window.confirm("Submit Application?\n\nThis will LOCK your CV and Documents.")) return;
        setIsLoading(true); // Full page spinner for submission is okay
        try {
             // await client.post(`/application/${applicationId}/submit`);
             window.location.reload(); 
        } catch (err) {
            alert("Failed to submit application.");
            setIsLoading(false);
        }
    };

    const handleExportCV = (type) => {
        if (!data?.app?.base_cv_id) return;
        let url = "";
        
        if (type === 'base') {
            url = `${API_BASE_URL}/cv/${data.app.base_cv_id}/export/pdf`;
        } else {
            if (data.app.derived_cv_id) {
                url = `${API_BASE_URL}/cv/${data.app.derived_cv_id}/export/pdf`;
            } else {
                url = `${API_BASE_URL}/cv/${data.app.base_cv_id}/export/pdf?mapping_id=${data.app.mapping_id}&application_id=${data.app.id}`;
            }
        }
        window.open(url, '_blank');
        setIsExportMenuOpen(false);
    };

    // --- HELPERS ---
    const getGradeColor = (grade) => {
        switch(grade) {
            case 'A': return 'text-success bg-success-subtle';
            case 'B': return 'text-primary bg-primary-subtle';
            case 'C': return 'text-warning bg-warning-subtle';
            default: return 'text-danger bg-danger-subtle';
        }
    };

    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin text-primary" size={32}/></div>;
    if (!data || !data.app) return <div className="p-5">Failed to load application.</div>;

    // Destructure for render
    const { job, docs } = data;
    
    // Use independent forensics state
    const stats = forensics?.stats || {};
    const grade = forensics?.suggested_grade || 'D';
    const badges = forensics?.suggested_badges || [];

    return (
        <div className="container-xl py-4 animate-fade-in">
            
            {/* Header & Status */}
            <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-start mb-5 gap-3">
                <div>
                    <div className="d-flex align-items-center gap-2 text-muted mb-2">
                        <div className="bg-light p-1 rounded"><Briefcase size={14} /></div>
                        <span className="small fw-bold text-uppercase tracking-wide">{job?.company || "Unknown Company"}</span>
                    </div>
                    <h1 className="fw-bold text-dark mb-2 display-6">{job?.title || "Unknown Job"}</h1>
                    <div className="d-flex flex-wrap gap-3 text-muted small">
                        {job?.location && <span className="d-flex align-items-center gap-1"><MapPin size={14}/> {job.location}</span>}
                        {job?.application_end_date && <span className="d-flex align-items-center gap-1"><Calendar size={14}/> Due: {job.application_end_date}</span>}
                    </div>
                </div>

                <div className="d-flex align-items-center gap-3">
                    {data.app.is_locked ? (
                        <div className="px-4 py-2 bg-success-subtle text-success border border-success-subtle rounded-pill d-flex align-items-center gap-2 shadow-sm">
                            <ShieldCheck size={18} />
                            <span className="fw-bold">Applied & Locked</span>
                        </div>
                    ) : (
                        <button 
                            className="btn btn-dark px-4 py-2 rounded-pill shadow-sm hover-lift d-flex align-items-center gap-2"
                            onClick={handleSubmitApplication}
                        >
                            <span>Submit Application</span>
                            <ArrowRight size={16} />
                        </button>
                    )}
                </div>
            </div>

            {/* Main Grid */}
            <div className="row g-4">
                
                {/* LEFT COLUMN: STRATEGY & ASSETS */}
                <div className="col-lg-4">
                    <h6 className="fw-bold text-uppercase text-muted mb-3 small tracking-wide">Strategy & Fit</h6>
                    
                    {/* 1. RoleCase Card */}
                    <div 
                        className="card border-0 shadow-sm mb-3 hover-lift transition-all overflow-hidden cursor-pointer group position-relative"
                        onClick={() => !isAnalyzing && navigate(`/application/${applicationId}/rolecase`)}
                    >
                        {/* LOADING OVERLAY (Scoped to Card) */}
                        {isAnalyzing && (
                            <div className="position-absolute top-0 start-0 w-100 h-100 bg-white bg-opacity-90 d-flex flex-column align-items-center justify-content-center" style={{zIndex: 10}}>
                                <Loader2 className="animate-spin text-primary mb-2" size={32}/>
                                <span className="small fw-bold text-uppercase tracking-wide text-primary">Analyzing Fit...</span>
                            </div>
                        )}

                        <div className="card-body p-4">
                            <div className="d-flex align-items-center justify-content-between mb-4">
                                <div className="d-flex align-items-center gap-2 text-dark">
                                    <div className="bg-dark text-white p-2 rounded-circle">
                                        <Microscope size={18} />
                                    </div>
                                    <h6 className="fw-bold mb-0">RoleCase</h6>
                                </div>
                                {!isAnalyzing && <ChevronRight size={18} className="text-muted opacity-50 group-hover-translate transition-transform" />}
                            </div>

                            {/* Score & Grade */}
                            <div className="d-flex align-items-end gap-3 mb-4">
                                <div className="display-4 fw-bold text-dark lh-1">
                                    {stats.overall_match_score || 0}<span className="fs-4 text-muted">%</span>
                                </div>
                                {!isAnalyzing && (
                                    <div className={`px-3 py-1 rounded fw-bold mb-2 ${getGradeColor(grade)}`}>
                                        Grade {grade}
                                    </div>
                                )}
                            </div>

                            {/* Badges */}
                            <div className="d-flex flex-wrap gap-2 mb-4">
                                {badges.length > 0 ? badges.slice(0, 3).map((b, i) => (
                                    <span key={i} className="badge bg-light text-secondary border fw-normal py-2 px-3 rounded-pill">
                                        {b}
                                    </span>
                                )) : (
                                    !isAnalyzing && <span className="text-muted small fst-italic">No badges generated.</span>
                                )}
                            </div>

                            {/* Stats */}
                            {!isAnalyzing && (
                                <div className="border-top pt-3">
                                    <div className="d-flex justify-content-between align-items-center small text-muted">
                                        <span>Evidence Sources:</span>
                                    </div>
                                    <div className="d-flex gap-3 mt-2">
                                        {stats.evidence_sources?.['Professional'] > 0 && (
                                            <div className="d-flex align-items-center gap-1 small text-dark" title="Professional">
                                                <Briefcase size={14} className="text-primary"/>
                                                <span className="fw-bold">{stats.evidence_sources['Professional']}</span>
                                            </div>
                                        )}
                                        {stats.evidence_sources?.['Academic'] > 0 && (
                                            <div className="d-flex align-items-center gap-1 small text-dark" title="Academic">
                                                <GraduationCap size={14} className="text-info"/>
                                                <span className="fw-bold">{stats.evidence_sources['Academic']}</span>
                                            </div>
                                        )}
                                        {stats.evidence_sources?.['Other'] > 0 && (
                                            <div className="d-flex align-items-center gap-1 small text-dark" title="Other">
                                                <Trophy size={14} className="text-warning"/>
                                                <span className="fw-bold">{stats.evidence_sources['Other']}</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}
                        </div>
                        <div className="card-footer bg-light border-0 py-2 text-center">
                            <small className="text-primary fw-bold text-uppercase tracking-wide" style={{fontSize: '0.7rem'}}>
                                Tap to view full RoleCase
                            </small>
                        </div>
                    </div>

                    {/* 2. Tailored CV Card */}
                    <div className="card border-0 shadow-sm mb-3 position-relative">
                        {isAnalyzing && (
                             <div className="position-absolute top-0 start-0 w-100 h-100 bg-white bg-opacity-50" style={{zIndex: 5}}></div>
                        )}

                        <div className="card-header bg-white border-0 pt-3 pb-0">
                            <div className="d-flex align-items-center gap-2 mb-1">
                                <Sparkles size={16} className="text-warning fill-warning" />
                                <h6 className="fw-bold text-uppercase text-muted mb-0 small tracking-wide">Tailored Asset</h6>
                            </div>
                        </div>
                        <div className="card-body p-3">
                            <div className="d-flex align-items-center justify-content-between p-2 rounded-3 bg-light border border-light overflow-hidden">
                                <div className="d-flex align-items-center gap-3 overflow-hidden flex-grow-1">
                                    <div className="bg-white p-2 rounded shadow-sm text-primary flex-shrink-0">
                                        <FileCheck size={20}/>
                                    </div>
                                    <div style={{minWidth: 0}} className="flex-grow-1">
                                        <h6 className="fw-bold mb-0 text-dark text-truncate">Ready-Made CV</h6>
                                        <small className="text-muted d-block text-truncate" style={{fontSize: '0.75rem'}}>
                                            Optimized for {job?.company || "this job"}
                                        </small>
                                    </div>
                                </div>
                                <div className="position-relative flex-shrink-0 ms-2" ref={exportMenuRef}>
                                    <button 
                                        className="btn btn-white btn-sm border shadow-sm rounded-circle p-2 d-flex align-items-center justify-content-center hover-lift"
                                        onClick={() => setIsExportMenuOpen(!isExportMenuOpen)}
                                        title="Export Options"
                                    >
                                        <Download size={16} className="text-dark" />
                                    </button>
                                    {isExportMenuOpen && (
                                        <div className="position-absolute end-0 mt-2 bg-white border rounded-3 shadow-lg py-2 z-index-dropdown" style={{width: '240px', zIndex: 1000}}>
                                            <div className="px-3 py-2 border-bottom mb-1">
                                                <span className="text-muted small fw-bold text-uppercase">Export Options</span>
                                            </div>
                                            <button className="dropdown-item px-3 py-2 d-flex align-items-center gap-2" onClick={() => handleExportCV('tailored')}>
                                                <Sparkles size={14} className="text-warning" />
                                                <div className="d-flex flex-column align-items-start lh-sm">
                                                    <span className="fw-bold text-dark small">Tailored CV</span>
                                                    <span className="text-muted" style={{fontSize: '0.65rem'}}>Reordered & Highlighted</span>
                                                </div>
                                            </button>
                                            <button className="dropdown-item px-3 py-2 d-flex align-items-center gap-2" onClick={() => handleExportCV('base')}>
                                                <FileText size={14} className="text-muted" />
                                                <div className="d-flex flex-column align-items-start lh-sm">
                                                    <span className="fw-medium text-dark small">Original Base CV</span>
                                                    <span className="text-muted" style={{fontSize: '0.65rem'}}>Standard master copy</span>
                                                </div>
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>
                            {!data.app.is_locked && (
                                <div className="text-end mt-2">
                                    <span className="text-muted small cursor-pointer hover-text-primary" onClick={() => navigate(`/application/${applicationId}/tailored-cv`)}>
                                        Customize Preview <ChevronRight size={12} />
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* RIGHT COLUMN: SUPPORTING DOCS */}
                <div className="col-lg-8">
                    <div className="d-flex justify-content-between align-items-center mb-3">
                        <h6 className="fw-bold text-uppercase text-muted mb-0 small tracking-wide">Supporting Documents</h6>
                        {!data.app.is_locked && (
                            <button className="btn btn-sm btn-outline-dark rounded-pill d-flex align-items-center gap-2 px-3" onClick={handleCreateDocClick}>
                                <Plus size={14} /> New Document
                            </button>
                        )}
                    </div>

                    {data.docs.length === 0 ? (
                        <div className="text-center py-5 border border-dashed rounded-4 bg-light-subtle">
                            <div className="text-muted mb-2"><FileText size={32} className="opacity-25"/></div>
                            <p className="text-muted fw-medium">No documents created yet.</p>
                        </div>
                    ) : (
                        <div className="row g-3">
                            {data.docs.map(doc => (
                                <div key={doc.id} className="col-md-6">
                                    <div className="card h-100 border-0 shadow-sm cursor-pointer hover-border-primary transition-all group position-relative overflow-hidden" onClick={() => navigate(`/application/${applicationId}/doc/${doc.id}`)}>
                                        <div className={`position-absolute top-0 start-0 h-100 w-1 ${doc.is_locked ? 'bg-success' : 'bg-primary'}`} style={{width: '4px'}}></div>
                                        <div className="card-body p-4 ps-4">
                                            <div className="d-flex justify-content-between align-items-start mb-4">
                                                <div className="bg-light text-primary p-2 rounded-circle"><FileText size={20} /></div>
                                                <ChevronRight size={18} className="text-muted opacity-50 group-hover-translate transition-transform" />
                                            </div>
                                            <h5 className="fw-bold text-dark mb-2 text-truncate" title={doc.name}>{doc.name || "Untitled Document"}</h5>
                                            <div className="d-flex align-items-center gap-3 text-muted small">
                                                <span>{doc.paragraphs?.length || 0} Sections</span>
                                                <span className="opacity-25">•</span>
                                                <span>{doc.ideas?.length || 0} Arguments</span>
                                            </div>
                                            {doc.is_locked && (
                                                <div className="mt-3 pt-3 border-top d-flex align-items-center gap-2 text-success small fw-bold"><ShieldCheck size={14} /> Locked Snapshot</div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <NewDocumentModal isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} onCreate={handleCreateDocSubmit} />

            <style>{`
                .hover-lift:hover { transform: translateY(-2px); }
                .hover-border-primary:hover { box-shadow: 0 0 0 1px var(--bs-primary) !important; }
                .group:hover .group-hover-translate { transform: translateX(4px); }
                .hover-text-primary:hover { color: var(--bs-primary) !important; text-decoration: underline; }
                .dropdown-item:active { background-color: var(--bs-primary); color: white; }
                .dropdown-item:active .text-muted { color: rgba(255,255,255,0.8) !important; }
            `}</style>
        </div>
    );
};

export default ApplicationDashboard;