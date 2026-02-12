import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    fetchAllCVs, 
    createBaseCV, 
    deleteBaseCV, 
    importCV
} from '../api/cvClient';

// Child Components
import CVSelector from './cv/CVList';
import ImportCVModal from './cv/ImportCVModal';

// Icons
import { UploadCloud } from 'lucide-react';

const CVLibraryPage = () => {
    const navigate = useNavigate();
    
    // --- State ---
    const [cvs, setCvs] = useState([]);
    const [loading, setLoading] = useState(true);
    
    // Modals
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [showImportModal, setShowImportModal] = useState(false);
    
    // Forms
    const [createFormData, setCreateFormData] = useState({ 
        name: '', 
        first_name: '', 
        last_name: '', 
        title: '', 
        summary: '' 
    });

    // --- Data Loading (Silent Support) ---
    // Added 'silent' param to prevent full-page spinner on auto-refresh
    const loadCvs = async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            const data = await fetchAllCVs();
            setCvs(data || []);
        } catch (error) {
            console.error("Failed to fetch CVs", error);
        } finally {
            if (!silent) setLoading(false);
        }
    };

    // 1. Initial Load
    useEffect(() => { 
        loadCvs(false); 
    }, []);

    // 2. Regular Auto-Refresh (Every 5 seconds)
    // This ensures the list stays in sync with background tasks
    useEffect(() => {
        const interval = setInterval(() => {
            loadCvs(true); // Silent refresh
        }, 5000); 
        return () => clearInterval(interval);
    }, []);

    // --- Handlers ---

    const handleStartImport = async (name, textData) => {
        try {
            setShowImportModal(false); 
            
            // Start backend task (creates DB placeholder)
            await importCV(name, textData);
            
            // Immediate refresh to show the new "Importing..." card
            await loadCvs(true); 
            
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
        if (window.confirm("Are you sure you want to delete this master CV?")) {
            try {
                await deleteBaseCV(cvId);
                await loadCvs(true); 
            } catch (error) {
                alert("Failed to delete CV.");
            }
        }
    };

    // --- Render ---

    return (
        <div className="text-start pb-5">
            <div className="mb-4">
                <h2 className="fw-bold text-dark mb-3">CV Library</h2>
                
                <button 
                    onClick={() => setShowImportModal(true)} 
                    className="btn btn-outline-primary d-flex align-items-center gap-2 mb-3"
                >
                    <UploadCloud size={18} /> Import CV
                </button>
                
                {loading ? (
                    <div className="d-flex justify-content-center py-5">
                        <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading...</span>
                        </div>
                    </div>
                ) : (
                    <CVSelector 
                        cvs={cvs} 
                        onSelect={(id) => navigate(`/cv/${id}`)} 
                        onCreate={() => setShowCreateModal(true)} 
                        onDelete={handleDeleteCV}
                    />
                )}
            </div>

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
                                    {/* Additional fields hidden for brevity */}
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