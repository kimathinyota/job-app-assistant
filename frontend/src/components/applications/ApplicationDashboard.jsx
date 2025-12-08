// frontend/src/components/applications/ApplicationDashboard.jsx
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    Briefcase, FileText, ShieldCheck, Plus, 
    ChevronRight, MapPin, Calendar, ArrowRight, 
    Layout, Loader2, CheckCircle2
} from 'lucide-react';
import { 
    fetchApplicationDetails, fetchJobDetails, fetchMappingDetails, 
    createCoverLetter, updateApplication, fetchCoverLetterDetails 
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient';
import NewDocumentModal from './NewDocumentModal'; // <--- Import the new Modal

const ApplicationDashboard = () => {
    const { applicationId } = useParams();
    const navigate = useNavigate();
    
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isModalOpen, setIsModalOpen] = useState(false); // <--- State for Modal
    
    // Note: activeDocId logic is removed because we now navigate to a separate route for docs

    // --- 1. LOAD & SMART START ---
    useEffect(() => {
        const load = async () => {
            try {
                const appRes = await fetchApplicationDetails(applicationId);
                let app = appRes.data;
                
                // Parallel fetch for dependencies
                // We catch individual errors so the dashboard doesn't crash if one part fails
                const [jobRes, cvData, mappingRes] = await Promise.all([
                    fetchJobDetails(app.job_id).catch(e => ({ data: null })),
                    fetchCVDetails(app.base_cv_id).catch(e => null), 
                    fetchMappingDetails(app.mapping_id).catch(e => ({ data: null }))
                ]);

                const job = jobRes.data || {};
                const cv = cvData || { name: "Unknown CV" }; // fetchCVDetails returns data directly
                const mapping = mappingRes.data || { pairs: [] };

                // Fetch Docs
                let docs = [];
                if (app.supporting_document_ids && app.supporting_document_ids.length > 0) {
                    for (const docId of app.supporting_document_ids) {
                         try {
                             const dRes = await fetchCoverLetterDetails(docId);
                             if (dRes.data) docs.push(dRes.data);
                         } catch (e) { console.warn("Failed to load doc", docId); }
                    }
                }

                // ** SMART START **: If no docs exist, create the default "Cover Letter"
                if (docs.length === 0 && !app.is_locked) {
                    try {
                        const res = await createCoverLetter(app.job_id, app.base_cv_id, app.mapping_id, "Cover Letter");
                        const newDoc = res.data;
                        docs.push(newDoc);
                        
                        // Link it
                        const newIds = [...(app.supporting_document_ids || []), newDoc.id];
                        await updateApplication(app.id, { supporting_document_ids: newIds });
                        app = { ...app, supporting_document_ids: newIds };
                    } catch (e) {
                        console.error("Smart Start failed", e);
                    }
                }

                setData({ app, job, cv, mapping, docs });

            } catch (err) {
                console.error(err);
            } finally {
                setIsLoading(false);
            }
        };
        load();
    }, [applicationId]);

    // --- HANDLERS ---

    const handleCreateDocClick = () => {
        setIsModalOpen(true);
    };

    const handleCreateDocSubmit = async (name, layout, questions) => {
        try {
            // 1. Create the base document
            const res = await createCoverLetter(data.app.job_id, data.app.base_cv_id, data.app.mapping_id, name);
            const newDoc = res.data;
            
            // 2. Apply the selected layout strategy
            if (layout === 'qa') {
                // Call the new Q&A endpoint
                await client.post(`/coverletter/${newDoc.id}/autofill_qa`, { questions: questions });
            } else if (layout !== 'blank') {
                // Call the existing autofill endpoint
                await client.post(`/coverletter/${newDoc.id}/autofill?strategy=${layout}&mode=reset`);
            }
            
            // 3. Link to Application
            const newIds = [...(data.app.supporting_document_ids || []), newDoc.id];
            await updateApplication(data.app.id, { supporting_document_ids: newIds });
            
            setIsModalOpen(false);
            
            // 4. Navigate
            navigate(`/application/${applicationId}/doc/${newDoc.id}`);
            
        } catch (err) {
            console.error(err);
            alert("Failed to create document.");
        }
    };

    const handleCreateDoc = async () => {
        const name = prompt("Document Name (e.g., 'Selection Criteria', 'Personal Statement'):", "Supporting Document");
        if (!name) return;
        
        setIsLoading(true);
        try {
            const res = await createCoverLetter(data.app.job_id, data.app.base_cv_id, data.app.mapping_id, name);
            const newDoc = res.data;
            
            const newIds = [...(data.app.supporting_document_ids || []), newDoc.id];
            await updateApplication(data.app.id, { supporting_document_ids: newIds });
            
            // Navigate immediately to the new doc
            navigate(`/application/${applicationId}/doc/${newDoc.id}`);
            
        } catch (err) {
            alert("Failed to create document.");
            setIsLoading(false);
        }
    };

    const handleSubmitApplication = async () => {
        if (!window.confirm("Submit Application?\n\nThis will LOCK your CV and Documents to create a permanent snapshot of what you applied with.")) return;
        
        setIsLoading(true);
        try {
             await client.post(`/application/${applicationId}/submit`);
             window.location.reload(); 
        } catch (err) {
            alert("Failed to submit application.");
            setIsLoading(false);
        }
    };

    // --- MAPPING CALCULATIONS ---
    const getMappingStats = () => {
        if (!data?.job?.features || !data?.mapping?.pairs) return { percent: 0, text: "No data" };
        
        const requirements = data.job.features.filter(f => f.type === 'requirement' || f.type === 'qualification');
        const totalReqs = requirements.length;
        if (totalReqs === 0) return { percent: 100, text: "No requirements found" };

        const mappedReqIds = new Set(data.mapping.pairs.map(p => p.feature_id));
        const coveredReqs = requirements.filter(r => mappedReqIds.has(r.id)).length;
        
        const percent = Math.round((coveredReqs / totalReqs) * 100);
        return {
            percent,
            status: percent === 100 ? 'Complete' : `${coveredReqs}/${totalReqs} Requirements Covered`
        };
    };

    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin text-primary" size={32}/></div>;
    // Safety check for data integrity
    if (!data || !data.app) return <div className="p-5">Failed to load application.</div>;

    const mappingStats = getMappingStats();

    // --- DASHBOARD VIEW ---
    return (
        <div className="container-xl py-4 animate-fade-in">
            
            {/* Header & Status */}
            <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-start mb-5 gap-3">
                <div>
                    <div className="d-flex align-items-center gap-2 text-muted mb-2">
                        <div className="bg-light p-1 rounded"><Briefcase size={14} /></div>
                        <span className="small fw-bold text-uppercase tracking-wide">{data.job?.company || "Unknown Company"}</span>
                    </div>
                    <h1 className="fw-bold text-dark mb-2 display-6">{data.job?.title || "Unknown Job"}</h1>
                    <div className="d-flex flex-wrap gap-3 text-muted small">
                        {data.job?.location && <span className="d-flex align-items-center gap-1"><MapPin size={14}/> {data.job.location}</span>}
                        {data.job?.application_end_date && <span className="d-flex align-items-center gap-1"><Calendar size={14}/> Due: {data.job.application_end_date}</span>}
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
                
                {/* LEFT: Strategy & CV (The Foundation) */}
                <div className="col-lg-4">
                    <h6 className="fw-bold text-uppercase text-muted mb-3 small tracking-wide">Foundation</h6>
                    
                    {/* Mapping Card */}
                    <div className="card border-0 shadow-sm mb-3 hover-lift transition-all overflow-hidden">
                        <div className="card-body p-4">
                            <div className="d-flex align-items-center justify-content-between mb-3">
                                <div className="d-flex align-items-center gap-2 text-primary">
                                    <Layout size={20} />
                                    <h6 className="fw-bold mb-0">Strategy Mapping</h6>
                                </div>
                                {!data.app.is_locked && (
                                    <button 
                                        className="btn btn-sm btn-light text-muted stretched-link"
                                        onClick={() => navigate(`/application/${applicationId}/mapping`)}
                                    >
                                        Edit
                                    </button>
                                )}
                            </div>
                            
                            {/* Progress Bar */}
                            <div className="mb-2">
                                <div className="d-flex justify-content-between small mb-1">
                                    <span className={`fw-bold ${mappingStats.percent === 100 ? 'text-success' : 'text-muted'}`}>
                                        {mappingStats.status}
                                    </span>
                                    <span className="text-muted">{mappingStats.percent}%</span>
                                </div>
                                <div className="progress" style={{height: '6px'}}>
                                    <div 
                                        className={`progress-bar ${mappingStats.percent === 100 ? 'bg-success' : 'bg-primary'}`} 
                                        role="progressbar" 
                                        style={{width: `${mappingStats.percent}%`}} 
                                    />
                                </div>
                            </div>
                            
                            <p className="text-muted small mb-0 mt-3">
                                {data.mapping?.pairs?.length || 0} total evidence points linked.
                            </p>
                        </div>
                    </div>

                    {/* CV Card */}
                    <div className="card border-0 shadow-sm mb-3 hover-lift transition-all">
                        <div className="card-body p-4">
                            <div className="d-flex align-items-center justify-content-between mb-3">
                                <div className="d-flex align-items-center gap-2 text-primary">
                                    <FileText size={20} />
                                    <h6 className="fw-bold mb-0">Tailored CV</h6>
                                </div>
                                {!data.app.is_locked && (
                                    <button 
                                        className="btn btn-sm btn-light text-muted stretched-link"
                                        onClick={() => navigate(`/application/${applicationId}/cv`)}
                                    >
                                        Edit
                                    </button>
                                )}
                            </div>
                            <div className="d-flex align-items-center gap-2 text-muted small">
                                <CheckCircle2 size={14} className="text-success" />
                                <span>Based on <span className="fw-medium text-dark">{data.cv?.name || "Master CV"}</span></span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* RIGHT: Supporting Documents (The Work) */}
                <div className="col-lg-8">
                    <div className="d-flex justify-content-between align-items-center mb-3">
                        <h6 className="fw-bold text-uppercase text-muted mb-0 small tracking-wide">Supporting Documents</h6>
                        {!data.app.is_locked && (
                            <button 
                                className="btn btn-sm btn-outline-dark rounded-pill d-flex align-items-center gap-2 px-3"
                                onClick={handleCreateDocClick}
                            >
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
                                    <div 
                                        className="card h-100 border-0 shadow-sm cursor-pointer hover-border-primary transition-all group position-relative overflow-hidden"
                                        onClick={() => navigate(`/application/${applicationId}/doc/${doc.id}`)}
                                    >
                                        {/* Status Stripe */}
                                        <div className={`position-absolute top-0 start-0 h-100 w-1 ${doc.is_locked ? 'bg-success' : 'bg-primary'}`} style={{width: '4px'}}></div>
                                        
                                        <div className="card-body p-4 ps-4">
                                            <div className="d-flex justify-content-between align-items-start mb-4">
                                                <div className="bg-light text-primary p-2 rounded-circle">
                                                    <FileText size={20} />
                                                </div>
                                                <ChevronRight size={18} className="text-muted opacity-50 group-hover-translate transition-transform" />
                                            </div>
                                            
                                            <h5 className="fw-bold text-dark mb-2 text-truncate" title={doc.name}>{doc.name || "Untitled Document"}</h5>
                                            
                                            <div className="d-flex align-items-center gap-3 text-muted small">
                                                <span>{doc.paragraphs?.length || 0} Sections</span>
                                                <span className="opacity-25">•</span>
                                                <span>{doc.ideas?.length || 0} Arguments</span>
                                            </div>
                                            
                                            {doc.is_locked && (
                                                <div className="mt-3 pt-3 border-top d-flex align-items-center gap-2 text-success small fw-bold">
                                                    <ShieldCheck size={14} /> Locked Snapshot
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Modal */}
            <NewDocumentModal 
                isOpen={isModalOpen} 
                onClose={() => setIsModalOpen(false)} 
                onCreate={handleCreateDocSubmit}
            />

            <style>{`
                .hover-lift:hover { transform: translateY(-2px); }
                .hover-border-primary:hover { box-shadow: 0 0 0 1px var(--bs-primary) !important; }
                .group:hover .group-hover-translate { transform: translateX(4px); }
            `}</style>
        </div>
    );
};

export default ApplicationDashboard;