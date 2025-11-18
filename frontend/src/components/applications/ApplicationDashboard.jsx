// frontend/src/components/applications/ApplicationDashboard.jsx
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    Briefcase, FileText, ShieldCheck, Plus, 
    ChevronRight, MapPin, Calendar, ArrowRight 
} from 'lucide-react';
import { 
    fetchApplicationDetails, fetchJobDetails, fetchMappingDetails, 
    createCoverLetter, updateApplication 
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient';
import SupportingDocStudio from './SupportingDocStudio'; // The Component above

const ApplicationDashboard = () => {
    const { applicationId } = useParams();
    const navigate = useNavigate();
    const [data, setData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [activeDocId, setActiveDocId] = useState(null); // If set, we show the editor

    // Load Data
    useEffect(() => {
        const load = async () => {
            const app = (await fetchApplicationDetails(applicationId)).data;
            const job = (await fetchJobDetails(app.job_id)).data;
            const cv = (await fetchCVDetails(app.base_cv_id));
            const mapping = (await fetchMappingDetails(app.mapping_id)).data;
            
            // Fetch all docs
            const docs = [];
            for(const docId of app.supporting_document_ids) {
                // You might need a bulk fetch endpoint, or just loop for now
                try {
                    const d = (await fetchCoverLetterDetails(docId)).data;
                    docs.push(d);
                } catch(e) {}
            }
            
            setData({ app, job, cv, mapping, docs });
            setIsLoading(false);
        };
        load();
    }, [applicationId]);

    const handleCreateDoc = async () => {
        const name = prompt("Document Name (e.g., 'Cover Letter', 'Selection Criteria'):", "Cover Letter");
        if(!name) return;
        
        // 1. Create Doc
        const res = await createCoverLetter(data.app.job_id, data.app.base_cv_id, data.app.mapping_id, name);
        const newDoc = res.data;
        
        // 2. Link to App (This logic should ideally be in backend create_cover_letter, or we do it here manually)
        // We need an endpoint to add doc to app, or use the update endpoint
        const newDocIds = [...data.app.supporting_document_ids, newDoc.id];
        await updateApplication(data.app.id, { supporting_document_ids: newDocIds });
        
        // 3. Refresh
        setData(prev => ({ ...prev, app: { ...prev.app, supporting_document_ids: newDocIds }, docs: [...prev.docs, newDoc] }));
    };

    const handleSubmitApplication = async () => {
        if(!window.confirm("Ready to apply? This will lock your CV and Documents.")) return;
        // Call the backend submit/lock endpoint
        // await client.post(`/application/${applicationId}/submit`);
        // For now, just mock:
        alert("Snapshot created! Application status set to Applied.");
        navigate('/applications');
    };

    if(isLoading) return <div className="p-5 text-center">Loading Dashboard...</div>;

    // If editing a document, show the Studio instead of the Dashboard
    if (activeDocId) {
        return (
            <SupportingDocStudio 
                documentId={activeDocId}
                job={data.job}
                mapping={data.mapping}
                fullCV={data.cv}
                isLocked={data.app.is_locked}
                onBack={() => setActiveDocId(null)}
            />
        );
    }

    // --- THE DASHBOARD ---
    return (
        <div className="container-xl py-4">
            
            {/* Header */}
            <div className="d-flex justify-content-between align-items-start mb-5">
                <div>
                    <div className="d-flex align-items-center gap-2 text-muted mb-1">
                        <Briefcase size={16} /> <span>{data.job.company}</span>
                    </div>
                    <h1 className="fw-bold text-dark mb-2">{data.job.title}</h1>
                    <div className="d-flex gap-3 text-muted small">
                        {data.job.location && <span className="d-flex align-items-center gap-1"><MapPin size={14}/> {data.job.location}</span>}
                        {data.job.application_end_date && <span className="d-flex align-items-center gap-1"><Calendar size={14}/> Due: {data.job.application_end_date}</span>}
                    </div>
                </div>
                <div className="d-flex gap-2">
                     {!data.app.is_locked && (
                        <button className="btn btn-dark px-4 py-2 d-flex align-items-center gap-2 shadow-sm" onClick={handleSubmitApplication}>
                            Submit Application <ArrowRight size={16} />
                        </button>
                     )}
                     {data.app.is_locked && (
                         <div className="badge bg-success-subtle text-success border border-success px-3 py-2 d-flex align-items-center gap-2">
                             <ShieldCheck size={16} /> Application Submitted
                         </div>
                     )}
                </div>
            </div>

            <div className="row g-4">
                {/* LEFT COL: Strategy */}
                <div className="col-lg-4">
                    <div className="card border-0 shadow-sm h-100">
                        <div className="card-header bg-white border-0 pt-4 px-4">
                            <h6 className="fw-bold text-uppercase text-muted mb-0" style={{letterSpacing: '0.05em'}}>Strategy</h6>
                        </div>
                        <div className="card-body px-4">
                            <div className="d-flex align-items-center justify-content-between mb-3">
                                <h5 className="fw-bold mb-0">Mapping</h5>
                                <button className="btn btn-sm btn-light">Edit</button>
                            </div>
                            <p className="text-muted small">
                                {data.mapping.pairs.length} evidence points mapped.
                            </p>
                            {/* ... Add stats chart here later ... */}
                        </div>
                    </div>
                </div>

                {/* RIGHT COL: Assets */}
                <div className="col-lg-8">
                    
                    {/* Tailored CV */}
                    <div className="mb-4">
                        <h6 className="fw-bold text-uppercase text-muted mb-3" style={{letterSpacing: '0.05em'}}>Application Assets</h6>
                        <div className="card border shadow-sm mb-3 cursor-pointer hover-lift transition-all">
                            <div className="card-body d-flex align-items-center gap-3">
                                <div className="bg-primary-subtle text-primary p-3 rounded-3">
                                    <FileText size={24} />
                                </div>
                                <div className="flex-grow-1">
                                    <h6 className="fw-bold mb-1">Tailored CV</h6>
                                    <small className="text-muted">Based on {data.cv.name}</small>
                                </div>
                                <button className="btn btn-sm btn-outline-primary">Edit CV</button>
                            </div>
                        </div>
                    </div>

                    {/* Supporting Docs Grid */}
                    <div className="d-flex justify-content-between align-items-center mb-3">
                        <h6 className="fw-bold text-uppercase text-muted mb-0" style={{letterSpacing: '0.05em'}}>Supporting Documents</h6>
                        {!data.app.is_locked && (
                            <button className="btn btn-sm btn-outline-dark d-flex align-items-center gap-2" onClick={handleCreateDoc}>
                                <Plus size={14} /> New Document
                            </button>
                        )}
                    </div>

                    <div className="row g-3">
                        {data.docs.map(doc => (
                            <div key={doc.id} className="col-md-6">
                                <div 
                                    className="card h-100 border shadow-sm cursor-pointer hover-border-primary transition-all group"
                                    onClick={() => setActiveDocId(doc.id)}
                                >
                                    <div className="card-body p-4 d-flex flex-column">
                                        <div className="d-flex justify-content-between align-items-start mb-3">
                                            <div className="bg-light text-dark p-2 rounded-circle">
                                                <FileText size={20} />
                                            </div>
                                            <ChevronRight size={18} className="text-muted group-hover-translate" />
                                        </div>
                                        <h6 className="fw-bold mb-2">{doc.name}</h6>
                                        <p className="text-muted small mb-0 flex-grow-1">
                                            {doc.paragraphs.length} sections • {doc.ideas.length} arguments
                                        </p>
                                        <div className="mt-3 pt-3 border-top">
                                            <span className="tiny text-muted text-uppercase fw-bold">Click to Edit</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                        
                        {data.docs.length === 0 && (
                            <div className="col-12">
                                <div className="text-center py-5 border border-dashed rounded-3 text-muted">
                                    No documents yet. Create a Cover Letter or Q&A doc.
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
             <style>{`
                .hover-lift:hover { transform: translateY(-2px); }
                .hover-border-primary:hover { border-color: var(--bs-primary) !important; }
                .group:hover .group-hover-translate { transform: translateX(3px); transition: transform 0.2s; }
            `}</style>
        </div>
    );
};

export default ApplicationDashboard;