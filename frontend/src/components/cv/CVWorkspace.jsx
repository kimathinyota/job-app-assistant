import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Outlet, Link, useLocation } from 'react-router-dom';
import { fetchCVDetails, updateBaseCV } from '../../api/cvClient';
import { ContactInfoManager } from './ContactInfoManager';
import ExportCVModal from './ExportCVModal';
import { Download, Edit2, ChevronLeft } from 'lucide-react';
import { getCVDisplayName } from '../../utils/cvHelpers';

const CVWorkspace = () => {
    const { cvId } = useParams();
    const navigate = useNavigate();
    const location = useLocation();
    
    const [cv, setCv] = useState(null);
    const [loading, setLoading] = useState(true);
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({});
    const [showExportModal, setShowExportModal] = useState(false);

    // --- Data Loading ---
    const loadCV = async () => {
        setLoading(true);
        try {
            const data = await fetchCVDetails(cvId);
            setCv(data);
        } catch (error) {
            console.error("Failed to load CV", error);
            alert("Could not load CV. It may have been deleted.");
            navigate('/cvs');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        if (cvId) loadCV();
    }, [cvId]);

    // --- Handlers ---
    const handleStartEditHeader = () => {
        setEditFormData({
            name: cv.name,
            first_name: cv.first_name || '',
            last_name: cv.last_name || '',
            title: cv.title || '',
            summary: cv.summary || ''
        });
        setIsEditingHeader(true);
    };

    const handleUpdateHeader = async (e) => {
        e.preventDefault();
        try {
            const updated = await updateBaseCV(cv.id, editFormData);
            setCv(updated);
            setIsEditingHeader(false);
        } catch (err) {
            alert("Failed to update header.");
        }
    };

    const handleUpdateContact = async (contactInfo) => {
        try {
            const updated = await updateBaseCV(cv.id, { contact_info: contactInfo });
            setCv(updated);
            // We don't need to alert on success, the UI update is enough feedback
        } catch (err) {
            alert("Failed to update contact info.");
        }
    };

    if (loading) return <div className="p-5 text-center text-muted">Loading CV Workspace...</div>;
    if (!cv) return null;

    const isDashboard = location.pathname.endsWith(`/cv/${cvId}`) || location.pathname.endsWith(`/cv/${cvId}/`);

    return (
        <div className="text-start pb-5">
            {/* --- Navigation --- */}
            <div className="mb-3 d-flex align-items-center gap-2 text-muted small">
                <Link to="/cvs" className="text-decoration-none text-secondary hover:text-primary">CV Library</Link> 
                <span>/</span>
                <span className="fw-bold text-dark">{cv.name}</span>
            </div>

            {/* --- Header Card --- */}
            <div className="bg-white rounded-xl border shadow-sm p-4 mb-4">
                {!isEditingHeader ? (
                    <div>
                        {/* Top Row: Title & Actions */}
                        <div className="d-flex justify-content-between align-items-start mb-3">
                            <div>
                                <div className="d-flex align-items-center gap-2 mb-1">
                                    <h3 className="h4 fw-bold text-primary mb-0">{getCVDisplayName(cv)}</h3>
                                    <span className="badge bg-light text-muted border">Master CV</span>
                                </div>
                                <p className="text-muted small mb-0">Internal ID: <span className="fw-medium text-dark">{cv.name}</span></p>
                            </div>
                            <div className="d-flex gap-2">
                                <button onClick={() => setShowExportModal(true)} className="btn btn-outline-success btn-sm d-flex align-items-center gap-2">
                                    <Download size={14}/> Export
                                </button>
                                <button onClick={handleStartEditHeader} className="btn btn-outline-secondary btn-sm d-flex align-items-center gap-2">
                                    <Edit2 size={14}/> Edit Header
                                </button>

                                <button 
                                    className="btn btn-sm btn-outline-secondary" 
                                    onClick={() => navigate(`/cv/${cv.id}/quick-edit`)}
                                >
                                    Preview
                                </button>
                            </div>
                        </div>

                        {/* Summary Section */}
                        <div className="mb-4">
                            <p className="text-muted mb-0" style={{whiteSpace: 'pre-wrap'}}>
                                {cv.summary || <span className="fst-italic opacity-50">No summary provided.</span>}
                            </p>
                        </div>

                        {/* Contact Info - Moved to its own full-width block */}
                        <div className="border-top pt-3">
                            <ContactInfoManager contactInfo={cv.contact_info} onSave={handleUpdateContact} />
                        </div>
                    </div>
                ) : (
                    <form onSubmit={handleUpdateHeader} className="bg-light p-3 rounded">
                        <div className="row g-3 mb-3">
                            <div className="col-md-6">
                                <label className="form-label small fw-bold">Internal Name</label>
                                <input type="text" className="form-control" value={editFormData.name} onChange={e => setEditFormData({...editFormData, name: e.target.value})} required />
                            </div>
                            <div className="col-md-6">
                                <label className="form-label small fw-bold">Title</label>
                                <input type="text" className="form-control" value={editFormData.title} onChange={e => setEditFormData({...editFormData, title: e.target.value})} />
                            </div>
                            <div className="col-md-6">
                                <label className="form-label small fw-bold">First Name</label>
                                <input type="text" className="form-control" value={editFormData.first_name} onChange={e => setEditFormData({...editFormData, first_name: e.target.value})} />
                            </div>
                            <div className="col-md-6">
                                <label className="form-label small fw-bold">Last Name</label>
                                <input type="text" className="form-control" value={editFormData.last_name} onChange={e => setEditFormData({...editFormData, last_name: e.target.value})} />
                            </div>
                            <div className="col-12">
                                <label className="form-label small fw-bold">Summary</label>
                                <textarea className="form-control" rows={3} value={editFormData.summary} onChange={e => setEditFormData({...editFormData, summary: e.target.value})} />
                            </div>
                        </div>
                        <div className="d-flex gap-2">
                            <button type="submit" className="btn btn-primary btn-sm">Save Changes</button>
                            <button type="button" className="btn btn-light btn-sm" onClick={() => setIsEditingHeader(false)}>Cancel</button>
                        </div>
                    </form>
                )}
            </div>

            {/* --- Main Content Area --- */}
            <div className="animate-fade-in">
                {!isDashboard && (
                    <div className="mb-3">
                         <Link to={`/cv/${cvId}`} className="btn btn-link text-decoration-none p-0 d-flex align-items-center gap-1 text-muted">
                            <ChevronLeft size={16}/> Back to Dashboard
                        </Link>
                    </div>
                )}
                
                <Outlet context={{ cv, refreshCV: loadCV }} />
            </div>

            {showExportModal && (
                <ExportCVModal cvId={cv.id} onClose={() => setShowExportModal(false)} />
            )}
        </div>
    );
};

export default CVWorkspace;