import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
    fetchAllCVs, 
    createBaseCV, 
    deleteBaseCV, 
    importCV, 
    checkTaskStatus 
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

    // Background Import State
    const [activeImport, setActiveImport] = useState(null); 
    const pollingInterval = useRef(null);

    // --- Data Loading ---
    const loadCvs = async () => {
        setLoading(true);
        try {
            const data = await fetchAllCVs();
            setCvs(data || []);
        } catch (error) {
            console.error("Failed to fetch CVs", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => { 
        loadCvs(); 
    }, []);

    // --- Background Task Polling ---
    useEffect(() => {
        if (activeImport && activeImport.status === 'processing') {
            // Poll every 2 seconds
            pollingInterval.current = setInterval(async () => {
                try {
                    const statusData = await checkTaskStatus(activeImport.taskId);
                    
                    if (statusData.status === 'finished') {
                        // Task Complete
                        clearInterval(pollingInterval.current);
                        setActiveImport(null); 
                        alert(`CV "${activeImport.name}" imported successfully!`);
                        await loadCvs(); // Refresh list to show new CV
                    } 
                    else if (statusData.status === 'failed') {
                        // Task Failed
                        clearInterval(pollingInterval.current);
                        setActiveImport(prev => ({ 
                            ...prev, 
                            status: 'error', 
                            error: statusData.error 
                        }));
                        alert("Import failed. See console or UI for details.");
                    }
                } catch (err) {
                    console.error("Polling error", err);
                }
            }, 2000);
        }
        
        // Cleanup interval on unmount or when activeImport changes
        return () => { 
            if (pollingInterval.current) clearInterval(pollingInterval.current); 
        };
    }, [activeImport]);

    // --- Handlers ---

    const handleStartImport = async (name, textData) => {
        try {
            setShowImportModal(false); // Close the input modal
            
            // Start the backend task
            const data = await importCV(name, textData);
            
            // Set local state to track progress (shows spinner in UI)
            setActiveImport({ 
                taskId: data.task_id, 
                name: name, 
                status: 'processing', 
                startTime: Date.now() 
            });
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
            // Navigate immediately to the new CV's workspace
            navigate(`/cv/${newCV.id}`);
        } catch (err) { 
            console.error(err);
            alert("Failed to create CV"); 
        }
    };

    const handleDeleteCV = async (cvId) => {
        if (window.confirm("Are you sure you want to delete this master CV? This cannot be undone.")) {
            try {
                await deleteBaseCV(cvId);
                await loadCvs(); // Refresh list
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
                
                {/* Import Button */}
                <button 
                    onClick={() => setShowImportModal(true)} 
                    className="btn btn-outline-primary d-flex align-items-center gap-2 mb-3"
                >
                    <UploadCloud size={18} /> Import CV
                </button>
                
                {/* CV List / Grid */}
                {loading ? (
                    <div className="d-flex justify-content-center py-5">
                        <div className="spinner-border text-primary" role="status">
                            <span className="visually-hidden">Loading...</span>
                        </div>
                    </div>
                ) : (
                    <CVSelector 
                        cvs={cvs} 
                        // Navigation Handler
                        onSelect={(id) => navigate(`/cv/${id}`)} 
                        selectedCVId={null} 
                        // Creation Handler
                        onCreate={() => setShowCreateModal(true)} 
                        // Pass background task state so the list can show a "Loading..." card
                        activeImport={activeImport}
                        onImportClick={() => setShowImportModal(true)}
                        // Delete Handler
                        onDelete={handleDeleteCV}
                    />
                )}
            </div>

            {/* --- Modals --- */}

            {showImportModal && (
                <ImportCVModal 
                    onClose={() => setShowImportModal(false)}
                    onStartBackgroundImport={handleStartImport}
                    activeImportTask={activeImport}
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
                                        <div className="form-text">Used to identify this CV in your library.</div>
                                    </div>
                                    {/* Optional fields could go here, but keeping it simple for quick creation */}
                                </div>
                                <div className="modal-footer">
                                    <button type="button" className="btn btn-light" onClick={() => setShowCreateModal(false)}>Cancel</button>
                                    <button type="submit" className="btn btn-primary">Create & Edit</button>
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