import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

// 1. API Imports
import { 
    fetchAllCVs, 
    createBaseCV, 
    deleteBaseCV, 
    importCV,
    setPrimaryCV 
} from '../api/cvClient';

// Import the existing Auth method
import { getCurrentUser } from '../api/authClient'; 

// 2. Component Imports
import CVSelector from './cv/CVList'; 
import ImportCVModal from './cv/ImportCVModal';
import { UploadCloud } from 'lucide-react';

const CVLibraryPage = () => {
    const navigate = useNavigate();
    
    // --- State ---
    const [cvs, setCvs] = useState([]);
    const [user, setUser] = useState(null); 
    const [loading, setLoading] = useState(true);
    
    // Modals
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showImportModal, setShowImportModal] = useState(false);
    
    // Forms
    const [createFormData, setCreateFormData] = useState({ 
        name: '', first_name: '', last_name: '', title: '', summary: '' 
    });

    // --- Data Loading ---
    const loadData = async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            // 1. Parallel Fetching
            // We catch user error silently so CVs still load even if auth checks are weird
            const [cvData, userRes] = await Promise.all([
                fetchAllCVs(),
                getCurrentUser().catch(err => null)
            ]);

            // 2. Handle Data
            const cvList = cvData || [];
            setCvs(cvList);

            console.log("Fetched CVs:", cvList);

            // IMPORTANT: Axios returns the payload in .data
            // If userRes is null (failed), userData is null.
            const userData = userRes ? userRes.data : null;

            if (userData) {
                // --- Auto-Set Default Logic ---
                // If user has NO primary CV, but DOES have CVs...
                if (!userData.primary_cv_id && cvList.length > 0) {
                    const fallbackId = cvList[0].id;
                    
                    // Update Local State immediately
                    userData.primary_cv_id = fallbackId; 
                    
                    // Update Backend quietly
                    setPrimaryCV(fallbackId).catch(err => console.warn("Auto-set failed", err));
                }
                // -----------------------------
                setUser(userData);
            }
            
        } catch (error) {
            console.error("Failed to fetch library data", error);
        } finally {
            if (!silent) setLoading(false);
        }
    };

    // Initial Load
    useEffect(() => { 
        loadData(false); 
    }, []);

    // Auto-Refresh
    useEffect(() => {
        const interval = setInterval(() => {
            loadData(true); 
        }, 5000); 
        return () => clearInterval(interval);
    }, []);

    // --- Handlers ---

    const handleSetPrimary = async (cvId) => {
        const previousPrimary = user?.primary_cv_id;
        
        // Optimistic Update
        setUser(prev => ({ ...prev, primary_cv_id: cvId }));

        try {
            await setPrimaryCV(cvId);
        } catch (error) {
            console.error("Failed to set primary CV", error);
            // Revert
            setUser(prev => ({ ...prev, primary_cv_id: previousPrimary }));
            alert("Failed to update default CV.");
        }
    };

    const handleStartImport = async (name, textData) => {
        try {
            setShowImportModal(false); 
            await importCV(name, textData);
            await loadData(true); 
        } catch (error) { 
            console.error("Import start failed", error);
            alert("Could not start import task."); 
        }
    };

    const handleCreateCV = async (e) => {
        e.preventDefault();
        try {
            const newCV = await createBaseCV(
                createFormData.name, 
                createFormData.first_name, 
                createFormData.last_name, 
                createFormData.title, 
                createFormData.summary
            );
            navigate(`/cv/${newCV.id}`);
        } catch (err) { 
            console.error(err);
            alert("Failed to create CV"); 
        }
    };

    const handleDeleteCV = async (cvId) => {
        if (window.confirm("Are you sure you want to delete this CV?")) {
            try {
                await deleteBaseCV(cvId);
                await loadData(true); 
            } catch (error) {
                alert("Failed to delete CV.");
            }
        }
    };

    // --- Render ---

    return (
        <div className="text-start pb-5">
            <div className="mb-4">
                <div className="d-flex justify-content-between align-items-center mb-4">
                    <h2 className="fw-bold text-dark mb-0">CV Library</h2>
                    <button 
                        onClick={() => setShowImportModal(true)} 
                        className="btn btn-outline-primary d-flex align-items-center gap-2"
                    >
                        <UploadCloud size={18} /> Import CV
                    </button>
                </div>
                
                {loading && !cvs.length ? (
                    <div className="d-flex justify-content-center py-5">
                        <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading...</span>
                        </div>
                    </div>
                ) : (
                    <CVSelector 
                        cvs={cvs} 
                        primaryCvId={user?.primary_cv_id} 
                        onSelect={(id) => navigate(`/cv/${id}`)} 
                        onCreate={() => setShowCreateModal(true)} 
                        onDelete={handleDeleteCV}
                        onSetPrimary={handleSetPrimary}
                    />
                )}
            </div>

            {/* Modals */}
            {showImportModal && (
                <ImportCVModal 
                    onClose={() => setShowImportModal(false)}
                    onStartBackgroundImport={handleStartImport}
                />
            )}

            {showCreateModal && (
                <div className="modal fade show d-block" style={{background: 'rgba(0,0,0,0.5)'}}>
                    <div className="modal-dialog modal-dialog-centered">
                        <div className="modal-content">
                            <div className="modal-header">
                                <h5 className="modal-title">Create New CV</h5>
                                <button type="button" className="btn-close" onClick={() => setShowCreateModal(false)}></button>
                            </div>
                            <form onSubmit={handleCreateCV}>
                                <div className="modal-body">
                                    <div className="mb-3">
                                        <label className="form-label">Internal Name <span className="text-danger">*</span></label>
                                        <input 
                                            type="text" 
                                            className="form-control" 
                                            placeholder="e.g. Master CV 2024"
                                            value={createFormData.name} 
                                            onChange={e => setCreateFormData({...createFormData, name: e.target.value})} 
                                            autoFocus 
                                            required
                                        />
                                    </div>
                                    <div className="row g-2">
                                        <div className="col-md-6 mb-3">
                                            <label className="form-label">First Name</label>
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                value={createFormData.first_name} 
                                                onChange={e => setCreateFormData({...createFormData, first_name: e.target.value})} 
                                            />
                                        </div>
                                        <div className="col-md-6 mb-3">
                                            <label className="form-label">Last Name</label>
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                value={createFormData.last_name} 
                                                onChange={e => setCreateFormData({...createFormData, last_name: e.target.value})} 
                                            />
                                        </div>
                                    </div>
                                    <div className="mb-3">
                                        <label className="form-label">Job Title</label>
                                        <input 
                                            type="text" 
                                            className="form-control" 
                                            placeholder="e.g. Senior Product Designer"
                                            value={createFormData.title} 
                                            onChange={e => setCreateFormData({...createFormData, title: e.target.value})} 
                                        />
                                    </div>
                                    <div className="mb-3">
                                        <label className="form-label">Summary</label>
                                        <textarea 
                                            className="form-control" 
                                            rows="3"
                                            placeholder="Brief professional summary..."
                                            value={createFormData.summary} 
                                            onChange={e => setCreateFormData({...createFormData, summary: e.target.value})} 
                                        ></textarea>
                                    </div>
                                </div>
                                <div className="modal-footer">
                                    <button type="button" className="btn btn-light" onClick={() => setShowCreateModal(false)}>Cancel</button>
                                    <button type="submit" className="btn btn-primary">Create</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CVLibraryPage;