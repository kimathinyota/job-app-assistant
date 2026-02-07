// frontend/src/components/CVManagerPage.jsx
import React, { useState, useEffect,  useRef } from 'react';
// 1. Import useParams and useNavigate
import { useLocation, useParams, useNavigate } from 'react-router-dom'; 
import {
    fetchAllCVs, 
    deleteBaseCV,
    fetchCVDetails,
    updateBaseCV,
    createBaseCV,
    // --- NEW IMPORTS FOR BACKGROUND TASKS ---
    importCV,
    checkTaskStatus,
    // Complex Managers
    addExperienceComplex,
    updateExperienceComplex,
    addEducationComplex,
    updateEducationComplex,
    addHobbyComplex,
    updateHobbyComplex,
    addProjectComplex,
    updateProjectComplex,
    
    // Simple Managers
    addSkill,
    deleteNestedItem
} from '../api/cvClient';

// --- Icons & UI ---
import { 
    Briefcase, 
    BookOpen, 
    Cpu, 
    Layers, 
    Award, 
    Smile, 
    ChevronLeft, 
    Trash2, 
    Edit2,
    Download,
    UploadCloud,
    // --- NEW ICONS FOR CONTACT INFO ---
    Phone, 
    Mail, 
    Globe, 
    MapPin, 
    Linkedin, 
    Plus, 
    X, 
    Save
} from 'lucide-react';

import ImportCVModal from './cv/ImportCVModal';
import ExportCVModal from './cv/ExportCVModal'; 
import { ContactInfoManager } from './cv/ContactInfoManager';

import CVSelector from './cv/CVList';
import { getCVDisplayName } from '../utils/cvHelpers'; 

// --- Managers ---
import ExperienceManager from './cv/ExperienceManager';
import EducationManager from './cv/EducationManager';
import HobbyManager from './cv/HobbyManager';
import ProjectManager from './cv/ProjectManager';
import SkillsetManager from './cv/SkillsetManager';
import AchievementHub from './cv/AchievementHub';


// --- Professional Section Card ---
const SectionCard = ({ title, count, icon: Icon, colorClass, onClick }) => (
    <div onClick={onClick} className="col-md-4 mb-3">
        <div className="card border-0 shadow-sm h-100 hover-lift cursor-pointer transition-all">
            <div className="card-body d-flex align-items-center gap-3 p-4">
                <div className={`p-3 rounded-circle bg-opacity-10 ${colorClass.replace('text-', 'bg-')}`}>
                    <Icon size={24} className={colorClass} />
                </div>
                <div>
                    <h5 className="fw-bold text-dark mb-0">{title}</h5>
                    <span className="text-muted small">{count} items</span>
                </div>
            </div>
        </div>
    </div>
);

// --- CVSectionDashboard ---
const CVSectionDashboard = ({ cv, onSelectSection }) => (
    <div className="row g-3 py-2">
        <SectionCard title="Experience" count={cv.experiences?.length || 0} icon={Briefcase} colorClass="text-blue-600" onClick={() => onSelectSection('Experiences')} />
        <SectionCard title="Education" count={cv.education?.length || 0} icon={BookOpen} colorClass="text-indigo-600" onClick={() => onSelectSection('Education')} />
        <SectionCard title="Projects" count={cv.projects?.length || 0} icon={Cpu} colorClass="text-purple-600" onClick={() => onSelectSection('Projects')} />
        <SectionCard title="Master Skills" count={cv.skills?.length || 0} icon={Layers} colorClass="text-emerald-600" onClick={() => onSelectSection('Skills')} />
        <SectionCard title="Achievements" count={cv.achievements?.length || 0} icon={Award} colorClass="text-amber-500" onClick={() => onSelectSection('Achievements')} />
        <SectionCard title="Hobbies" count={cv.hobbies?.length || 0} icon={Smile} colorClass="text-pink-500" onClick={() => onSelectSection('Hobbies')} />
    </div>
);

const CVManagerPage = () => {
    
    const { cvId } = useParams(); 
    const navigate = useNavigate();
    const location = useLocation();

    // --- State ---
    const [cvs, setCvs] = useState([]);
    const [loadingCvs, setLoadingCvs] = useState(true);
    const initialSection = location.state?.initialSection;
    const [selectedCVId, setSelectedCVId] = useState(null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [activeSection, setActiveSection] = useState(null);
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({ name: '', first_name: '', last_name: '', title: '', summary: '' });
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createFormData, setCreateFormData] = useState({ name: '', first_name: '', last_name: '', title: '', summary: '' });

    const [showImportModal, setShowImportModal] = useState(false);
    const [showExportModal, setShowExportModal] = useState(false);

    // --- NEW: Background Import State ---
    const [activeImport, setActiveImport] = useState(null); 
    const pollingInterval = useRef(null);

    // --- DATA LOADING ---
    const reloadData = async () => {
        setLoadingCvs(true);
        let newSelectedCvId = null;
        try {
            const data = await fetchAllCVs();
            setCvs(data || []);
            
            if (data && data.length > 0) {
                const cvFromUrl = data.find(cv => cv.id === cvId);
                if (cvFromUrl) {
                    newSelectedCvId = cvFromUrl.id;
                } else {
                    newSelectedCvId = data[0].id;
                }
            }
        } catch (error) {
            console.error("Failed to reload CVs:", error);
        } finally {
            setLoadingCvs(false);
            setSelectedCVId(newSelectedCvId);
        }
    };
    
    useEffect(() => { reloadData(); }, []);

    const fetchAndSetDetails = async (id) => {
        setLoadingDetails(true); 
        try {
            const response = await fetchCVDetails(id);
            setDetailedCV(response); 
        } catch (error) {
            console.error("Failed to load CV details:", error);
            setDetailedCV(null); 
        } finally {
            setLoadingDetails(false); 
        }
    };

    useEffect(() => {
        setIsEditingHeader(false);
        if (initialSection) setActiveSection(initialSection);
        else setActiveSection(null);

        if (selectedCVId) {
            fetchAndSetDetails(selectedCVId);
            if (cvId !== selectedCVId) navigate(`/cv/${selectedCVId}`, { replace: true });
        } else if (!loadingCvs && cvs.length === 0) {
             setDetailedCV(null);
             if (cvId) navigate('/cv', { replace: true });
        }
    }, [selectedCVId, cvs.length, initialSection, loadingCvs, cvId, navigate]);


    // --- NEW: POLLING ENGINE ---
    useEffect(() => {
        if (activeImport && activeImport.status === 'processing') {
            // Check status every 2 seconds
            pollingInterval.current = setInterval(async () => {
                try {
                    const statusData = await checkTaskStatus(activeImport.taskId);
                    
                    if (statusData.status === 'finished') {
                        clearInterval(pollingInterval.current);
                        setActiveImport(null); 
                        alert(`CV "${activeImport.name}" imported successfully!`);
                        
                        await reloadData(); 
                        if (statusData.result && statusData.result.id) {
                            setSelectedCVId(statusData.result.id);
                        }
                    } 
                    else if (statusData.status === 'failed') {
                        clearInterval(pollingInterval.current);
                        setActiveImport(prev => ({ ...prev, status: 'error', error: statusData.error }));
                        alert("Import failed. Click the loading tab for details.");
                    }
                } catch (err) {
                    console.error("Polling error", err);
                }
            }, 2000);
        }
        return () => { if (pollingInterval.current) clearInterval(pollingInterval.current); };
    }, [activeImport]);


    // --- HANDLERS ---

    const handleStartImport = async (name, textData) => {
        try {
            setShowImportModal(false); // Close modal
            
            const data = await importCV(name, textData); // Start Task
            
            // Start Polling
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

    // ... (Keep existing edit/update/create handlers) ...
    const handleStartEditHeader = () => {
        setEditFormData({
            name: detailedCV.name, first_name: detailedCV.first_name || '', last_name: detailedCV.last_name || '', title: detailedCV.title || '', summary: detailedCV.summary || ''
        });
        setIsEditingHeader(true);
    };

     const handleUpdateCVHeader = async (e) => {
        e.preventDefault();
        if (!editFormData.name.trim()) return alert('CV Internal Name is required.');
        try {
            const updatedCV = await updateBaseCV(detailedCV.id, editFormData);
            setDetailedCV(updatedCV); setIsEditingHeader(false); await reloadData(); 
        } catch (error) { alert('Failed to update CV.'); }
    };

    const handleUpdateContactInfo = async (newContactInfo) => {
        try {
            const updatedCV = await updateBaseCV(detailedCV.id, { contact_info: newContactInfo });
            setDetailedCV(updatedCV);
        } catch (error) { alert("Failed to update contact info."); }
    };

    const handleCreateCV = async (e) => {
        e.preventDefault();
        if (!createFormData.name.trim()) return alert("Internal Name is required");
        try {
            const newCV = await createBaseCV(createFormData.name, createFormData.first_name, createFormData.last_name, createFormData.title, createFormData.summary);
            setSelectedCVId(newCV.id); await reloadData(); setShowCreateModal(false);
            setCreateFormData({ name: '', first_name: '', last_name: '', title: '', summary: '' });
        } catch (err) { alert("Failed to create CV"); }
    };

    const handleAddOrUpdateNestedItem = async (cvId, data, itemType) => {
        const isUpdating = Boolean(data.id);
        const itemId = data.id;
        const apiFunctions = {
            'Experience': { add: addExperienceComplex, update: updateExperienceComplex },
            'Education': { add: addEducationComplex, update: updateEducationComplex },
            'Hobby': { add: addHobbyComplex, update: updateHobbyComplex },
            'Project': { add: addProjectComplex, update: updateProjectComplex },
            'Skill': { add: addSkill, update: addSkill }, 
        };
        try {
            let apiFn;
            if (isUpdating) {
                apiFn = apiFunctions[itemType]?.update;
                if (itemType === 'Skill') await apiFn(cvId, data); else await apiFn(cvId, itemId, data);
                alert(`${itemType} updated successfully!`);
            } else {
                apiFn = apiFunctions[itemType]?.add;
                await apiFn(cvId, data);
                alert(`${itemType} added successfully!`);
            }
            await reloadData(); await fetchAndSetDetails(cvId); 
        } catch (error) { alert(`Failed to save ${itemType}.`); console.error(error); }
    };

    const handleDeleteCV = async (cvIdToDelete) => {
        if (window.confirm("Delete this master CV?")) {
            try { await deleteBaseCV(cvIdToDelete); alert("CV deleted!"); await reloadData(); } 
            catch (error) { alert("Failed to delete CV."); }
        }
    };

     const handleDeleteNested = async (cvIdToDeleteFrom, itemId, listName) => {
        if (window.confirm(`Permanently delete this item?`)) {
            try { await deleteNestedItem(cvIdToDeleteFrom, itemId, listName); fetchAndSetDetails(cvIdToDeleteFrom); } 
            catch (error) { alert(`Error deleting item.`); }
        }
    };

    const renderSectionDetail = () => {
         if (!activeSection) return null;
         const commonProps = {
            cvId: detailedCV.id, allSkills: detailedCV.skills || [], allAchievements: detailedCV.achievements || [],
            onSubmit: handleAddOrUpdateNestedItem, onDelete: handleDeleteNested, onBack: () => setActiveSection(null)
        };
        const sections = {
            'Experiences': <ExperienceManager {...commonProps} experiences={detailedCV.experiences || []} />,
            'Education': <EducationManager {...commonProps} education={detailedCV.education || []} />,
            'Hobbies': <HobbyManager {...commonProps} hobbies={detailedCV.hobbies || []} />,
            'Projects': <ProjectManager {...commonProps} projects={detailedCV.projects || []} allExperiences={detailedCV.experiences || []} allEducation={detailedCV.education || []} allHobbies={detailedCV.hobbies || []} />,
            'Skills': <SkillsetManager {...commonProps} allExperiences={detailedCV.experiences || []} allEducation={detailedCV.education || []} allProjects={detailedCV.projects || []} allHobbies={detailedCV.hobbies || []} />,
            'Achievements': <AchievementHub {...commonProps} allExperiences={detailedCV.experiences || []} allEducation={detailedCV.education || []} allProjects={detailedCV.projects || []} allHobbies={detailedCV.hobbies || []} />
        };
        return sections[activeSection] || <p>Section not found.</p>; 
    };

    // --- RENDER ---
    
    if (loadingCvs) {
        return (
            <div className="text-center py-5">
                <div className="spinner-border text-primary" role="status"><span className="visually-hidden">Loading...</span></div>
            </div>
        );
    }
    
    return (
        <div className="text-start pb-5">
            <style>{`.hover-lift:hover { transform: translateY(-4px); box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important; } .cursor-pointer { cursor: pointer; }`}</style>

            <div className="mb-4">
                <h2 className="fw-bold text-dark mb-3">CV Library</h2>

                <button onClick={() => setShowImportModal(true)} className="btn btn-outline-primary d-flex align-items-center gap-2">
                    <UploadCloud size={18} /> Import CV
                </button>

                {/* --- CV SELECTOR (Passing Background State) --- */}
                <CVSelector 
                    cvs={cvs} 
                    onSelect={setSelectedCVId} 
                    selectedCVId={selectedCVId} 
                    onCreate={() => setShowCreateModal(true)} 
                    
                    // NEW PROPS
                    activeImport={activeImport}
                    onImportClick={() => setShowImportModal(true)}
                />
            </div>
            
            <div className="mt-3">
                {loadingDetails ? (
                    <div className="text-center py-5 text-muted">Loading CV details...</div>
                ) : detailedCV ? (
                    <div className="animate-fade-in">
                        {/* Header */}
                        <div className="bg-white rounded-xl border shadow-sm p-4 mb-4">
                            {!isEditingHeader ? (
                                <div className="d-flex justify-content-between align-items-start">
                                    <div className="w-100 me-3">
                                        <div className="d-flex align-items-center gap-2 mb-1">
                                            <h3 className="h4 fw-bold text-primary mb-0">{getCVDisplayName(detailedCV)}</h3>
                                            <span className="badge bg-light text-muted border">Master</span>
                                        </div>
                                        <p className="text-muted small mb-1">Internal ID: <span className="fw-medium text-dark">{detailedCV.name}</span></p>
                                        <p className="text-muted mb-0" style={{whiteSpace: 'pre-wrap'}}>{detailedCV.summary || <span className="fst-italic opacity-50">No summary.</span>}</p>
                                        <ContactInfoManager contactInfo={detailedCV.contact_info} onSave={handleUpdateContactInfo} />
                                    </div>
                                    <div className="d-flex gap-2 flex-shrink-0">
                                        <button onClick={() => setShowExportModal(true)} className="btn btn-outline-success btn-sm d-flex align-items-center gap-2"><Download size={14}/> Export</button>
                                        <button onClick={handleStartEditHeader} className="btn btn-outline-secondary btn-sm d-flex align-items-center gap-2"><Edit2 size={14}/> Edit Header</button>
                                    </div>
                                </div>
                            ) : (
                                <form onSubmit={handleUpdateCVHeader} className="bg-light p-3 rounded">
                                    <div className="row g-3 mb-3">
                                        <div className="col-12"><label className="form-label">Internal Name</label><input type="text" className="form-control" value={editFormData.name} onChange={e => setEditFormData({...editFormData, name: e.target.value})} required /></div>
                                        {/* ... (Other fields can be added here if needed) ... */}
                                    </div>
                                    <div className="d-flex gap-2">
                                        <button type="submit" className="btn btn-primary btn-sm">Save</button>
                                        <button type="button" className="btn btn-light btn-sm" onClick={() => setIsEditingHeader(false)}>Cancel</button>
                                    </div>
                                </form>
                            )}
                        </div>

                        {/* Content */}
                        {activeSection === null ? (
                            <>
                                <CVSectionDashboard cv={detailedCV} onSelectSection={setActiveSection} />
                                <div className="mt-4 pt-3 border-top text-end">
                                    <button onClick={() => handleDeleteCV(detailedCV.id)} className="btn btn-outline-danger btn-sm d-flex align-items-center gap-2 ms-auto"><Trash2 size={14} /> Delete CV</button>
                                </div>
                            </>
                        ) : (
                            <div className="bg-white rounded-xl border shadow-sm p-4">
                                <div className="mb-3 d-flex justify-content-end"> 
                                    <button onClick={() => setActiveSection(null)} className="btn btn-link text-decoration-none p-0 d-flex align-items-center gap-1 text-muted"><ChevronLeft size={16}/> Back to Dashboard</button>
                                </div>
                                {renderSectionDetail()}
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="text-center py-5 border rounded-xl border-dashed bg-light">
                        <p className="text-muted mb-3">{cvs.length > 0 ? "No CV Selected." : "No CVs found."}</p>
                        <button onClick={() => setShowCreateModal(true)} className="btn btn-primary">Create your first CV</button>
                    </div>
                )}
            </div>

            {/* --- FIX IS HERE: PASS CORRECT PROPS TO MODAL --- */}
            {showImportModal && (
                <ImportCVModal 
                    onClose={() => setShowImportModal(false)}
                    // Pass the Background Function, NOT onSuccess
                    onStartBackgroundImport={handleStartImport}
                    // Pass the State Object for Progress View
                    activeImportTask={activeImport}
                />
            )}

            {showExportModal && detailedCV && (
                <ExportCVModal cvId={detailedCV.id} onClose={() => setShowExportModal(false)} />
            )}

            {showCreateModal && (
                <div className="modal fade show d-block" style={{background: 'rgba(0,0,0,0.5)'}}>
                    <div className="modal-dialog modal-dialog-centered">
                        <div className="modal-content">
                            <div className="modal-header"><h5 className="modal-title">Create CV</h5><button type="button" className="btn-close" onClick={() => setShowCreateModal(false)}></button></div>
                            <form onSubmit={handleCreateCV}>
                                <div className="modal-body">
                                    <div className="mb-3"><label className="form-label">Name</label><input type="text" className="form-control" value={createFormData.name} onChange={e => setCreateFormData({...createFormData, name: e.target.value})} autoFocus required/></div>
                                </div>
                                <div className="modal-footer"><button type="button" className="btn btn-light" onClick={() => setShowCreateModal(false)}>Cancel</button><button type="submit" className="btn btn-primary">Create</button></div>
                            </form>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CVManagerPage;