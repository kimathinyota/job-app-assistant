// frontend/src/components/CVManagerPage.jsx
import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useParams, useNavigate } from 'react-router-dom'; 
import {
    fetchAllCVs, deleteBaseCV, fetchCVDetails, updateBaseCV, createBaseCV,
    importCV, checkTaskStatus,
    addExperienceComplex, updateExperienceComplex, addEducationComplex, updateEducationComplex,
    addHobbyComplex, updateHobbyComplex, addProjectComplex, updateProjectComplex,
    addSkill, deleteNestedItem
} from '../api/cvClient';

import { 
    Briefcase, BookOpen, Cpu, Layers, Award, Smile, 
    ChevronLeft, Trash2, Edit2, Download, UploadCloud,
    Loader2
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

// --- Dashboard Grid Card ---
const SectionCard = ({ title, count, icon: Icon, colorClass, onClick }) => (
    <div onClick={onClick} className="col-12 col-md-6 col-lg-4 mb-3">
        <div className="card border-0 shadow-sm h-100 hover-lift cursor-pointer transition-all">
            <div className="card-body d-flex flex-column align-items-center justify-content-center text-center p-4">
                <div className={`p-3 rounded-circle bg-opacity-10 mb-3 ${colorClass.replace('text-', 'bg-')}`}>
                    <Icon size={28} className={colorClass} />
                </div>
                <h5 className="fw-bold text-dark mb-1">{title}</h5>
                <span className="text-muted small">{count} items</span>
            </div>
        </div>
    </div>
);

const CVSectionDashboard = ({ cv, onSelectSection }) => (
    <div className="row g-3">
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
    const [editFormData, setEditFormData] = useState({ name: '', first_name: '', last_name: '', title: '', summary: '', contact_info: {} });
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createFormData, setCreateFormData] = useState({ name: '', first_name: '', last_name: '', title: '', summary: '' });

    const [showImportModal, setShowImportModal] = useState(false);
    const [showExportModal, setShowExportModal] = useState(false);

    // --- Data Loading ---
    const reloadData = async () => {
        // Only set global loading on first load to prevent UI flicker during polling updates
        if (cvs.length === 0) setLoadingCvs(true);
        
        let newSelectedCvId = null;
        try {
            const data = await fetchAllCVs();
            setCvs(data || []);
            
            if (data && data.length > 0 && !selectedCVId) {
                const cvFromUrl = data.find(cv => cv.id === cvId);
                newSelectedCvId = cvFromUrl ? cvFromUrl.id : data[0].id;
            }
        } catch (error) {
            console.error("Failed to reload CVs:", error);
        } finally {
            setLoadingCvs(false);
            if (newSelectedCvId) setSelectedCVId(newSelectedCvId);
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
            // Don't fetch details if it's currently importing (it has no data yet)
            const isImporting = cvs.find(c => c.id === selectedCVId)?.is_importing;
            if (!isImporting) {
                fetchAndSetDetails(selectedCVId);
            } else {
                setDetailedCV(null);
            }
            
            if (cvId !== selectedCVId) navigate(`/cv/${selectedCVId}`, { replace: true });
        } else if (!loadingCvs && cvs.length === 0) {
             setDetailedCV(null);
             if (cvId) navigate('/cv', { replace: true });
        }
    }, [selectedCVId, cvs, initialSection, loadingCvs, cvId, navigate]);


    // --- ROBUST POLLING ENGINE ---
    // Watches the `cvs` list. If any CV has `is_importing: true`, it polls the task.
    // This survives page refreshes because the flag comes from the DB.
    useEffect(() => {
        const importingCv = cvs.find(c => c.is_importing);
        
        let intervalId = null;

        if (importingCv && importingCv.import_task_id) {
            intervalId = setInterval(async () => {
                try {
                    const statusData = await checkTaskStatus(importingCv.import_task_id);
                    
                    if (statusData.status === 'finished') {
                        // Task Complete
                        clearInterval(intervalId);
                        // Refresh data to get the final CV (is_importing will be false)
                        await reloadData(); 
                        setSelectedCVId(importingCv.id); // Auto-select the new CV
                        
                    } else if (statusData.status === 'failed') {
                        // Task Failed
                        clearInterval(intervalId);
                        alert(`Import failed: ${statusData.error}`);
                        await reloadData(); 
                    }
                    // If 'processing' or 'queued', simply continue polling
                } catch (err) { 
                    console.error("Polling error", err); 
                }
            }, 2000); // Poll every 2 seconds
        }

        return () => { 
            if (intervalId) clearInterval(intervalId); 
        };
    }, [cvs]); 


    // --- Handlers ---

    const handleStartImport = async (name, textData) => {
        try {
            setShowImportModal(false);
            // 1. Kick off import (Backend creates Placeholder CV immediately)
            await importCV(name, textData);
            
            // 2. Immediate reload to fetch the Placeholder CV
            // This adds the "Loading Card" to the UI instantly
            await reloadData(); 
            
        } catch (error) { 
            console.error(error); 
            alert("Could not start import task."); 
        }
    };

    const handleStartEditHeader = () => {
        setEditFormData({
            name: detailedCV.name, 
            first_name: detailedCV.first_name || '', 
            last_name: detailedCV.last_name || '', 
            title: detailedCV.title || '', 
            summary: detailedCV.summary || '',
            contact_info: detailedCV.contact_info || {}
        });
        setIsEditingHeader(true);
    };

     const handleUpdateCVHeader = async (e) => {
        e.preventDefault();
        if (!editFormData.name.trim()) return alert('CV Internal Name is required.');
        try {
            const updatedCV = await updateBaseCV(detailedCV.id, editFormData);
            setDetailedCV(updatedCV); 
            setIsEditingHeader(false); 
            await reloadData(); 
        } catch (error) { alert('Failed to update CV.'); }
    };

    const handleCreateCV = async (e) => {
        e.preventDefault();
        if (!createFormData.name.trim()) return alert("Internal Name is required");
        try {
            const newCV = await createBaseCV(createFormData.name, createFormData.first_name, createFormData.last_name, createFormData.title, createFormData.summary);
            setSelectedCVId(newCV.id); 
            await reloadData(); 
            setShowCreateModal(false);
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
            let apiFn = isUpdating ? apiFunctions[itemType]?.update : apiFunctions[itemType]?.add;
            if (itemType === 'Skill' && isUpdating) await apiFn(cvId, data); 
            else if (isUpdating) await apiFn(cvId, itemId, data);
            else await apiFn(cvId, data);
            
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

    if (loadingCvs) {
        return <div className="text-center py-5"><Loader2 className="animate-spin text-primary" size={32}/></div>;
    }
    
    return (
        <div className="text-start pb-5 w-100">
            <style>{`.hover-lift:hover { transform: translateY(-4px); box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important; } .cursor-pointer { cursor: pointer; }`}</style>

            <div className="mb-4">
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <h2 className="fw-bold text-dark mb-0">CV Library</h2>
                    <button onClick={() => setShowImportModal(true)} className="btn btn-outline-primary d-flex align-items-center gap-2">
                        <UploadCloud size={18} /> Import CV
                    </button>
                </div>

                <CVSelector 
                    cvs={cvs} 
                    onSelect={setSelectedCVId} 
                    selectedCVId={selectedCVId} 
                    // No need to pass activeImport prop anymore
                    onImportClick={() => setShowImportModal(true)}
                    onCreate={() => setShowCreateModal(true)} 
                />
            </div>
            
            <div className="mt-3">
                {loadingDetails ? (
                    <div className="text-center py-5 text-muted">Loading CV details...</div>
                ) : detailedCV ? (
                    <div className="animate-fade-in">
                        
                        <div className="row mb-4">
                            <div className="col-12">
                                <div className="bg-white rounded-xl border shadow-sm p-4">
                                    {!isEditingHeader ? (
                                        <div className="d-flex justify-content-between align-items-start">
                                            <div className="w-100 me-3">
                                                <div className="d-flex align-items-center gap-2 mb-1">
                                                    <h3 className="h4 fw-bold text-primary mb-0">{getCVDisplayName(detailedCV)}</h3>
                                                    <span className="badge bg-light text-muted border">Master</span>
                                                </div>
                                                <p className="text-muted small mb-1">Internal ID: <span className="fw-medium text-dark">{detailedCV.name}</span></p>
                                                <p className="text-muted mb-0" style={{whiteSpace: 'pre-wrap'}}>{detailedCV.summary || <span className="fst-italic opacity-50">No summary.</span>}</p>
                                                
                                                <ContactInfoManager 
                                                    contactInfo={detailedCV.contact_info} 
                                                    isEditing={false} 
                                                />
                                            </div>
                                            <div className="d-flex gap-2 flex-shrink-0">
                                                <button onClick={() => setShowExportModal(true)} className="btn btn-outline-success btn-sm d-flex align-items-center gap-2"><Download size={14}/> Export</button>
                                                <button onClick={handleStartEditHeader} className="btn btn-outline-secondary btn-sm d-flex align-items-center gap-2"><Edit2 size={14}/> Edit Header</button>
                                            </div>
                                        </div>
                                    ) : (
                                        <form onSubmit={handleUpdateCVHeader} className="bg-light p-3 rounded">
                                            <div className="row g-3 mb-3">
                                                <div className="col-12"><label className="form-label fw-bold small text-muted">Internal Name</label><input type="text" className="form-control" value={editFormData.name} onChange={e => setEditFormData({...editFormData, name: e.target.value})} required /></div>
                                                <div className="col-md-2"><label className="form-label fw-bold small text-muted">Title</label><input type="text" className="form-control" value={editFormData.title} onChange={e => setEditFormData({...editFormData, title: e.target.value})} /></div>
                                                <div className="col-md-5"><label className="form-label fw-bold small text-muted">First Name</label><input type="text" className="form-control" value={editFormData.first_name} onChange={e => setEditFormData({...editFormData, first_name: e.target.value})} /></div>
                                                <div className="col-md-5"><label className="form-label fw-bold small text-muted">Last Name</label><input type="text" className="form-control" value={editFormData.last_name} onChange={e => setEditFormData({...editFormData, last_name: e.target.value})} /></div>
                                                <div className="col-12"><label className="form-label fw-bold small text-muted">Summary</label><textarea className="form-control" rows="3" value={editFormData.summary} onChange={e => setEditFormData({...editFormData, summary: e.target.value})} /></div>
                                                
                                                <div className="col-12">
                                                    <ContactInfoManager 
                                                        contactInfo={editFormData.contact_info} 
                                                        isEditing={true} 
                                                        onChange={(newInfo) => setEditFormData(prev => ({ ...prev, contact_info: newInfo }))}
                                                    />
                                                </div>
                                            </div>
                                            <div className="d-flex gap-2 justify-content-end mt-3 border-top pt-3">
                                                <button type="button" className="btn btn-light" onClick={() => setIsEditingHeader(false)}>Cancel</button>
                                                <button type="submit" className="btn btn-primary px-4">Save Changes</button>
                                            </div>
                                        </form>
                                    )}
                                </div>
                            </div>
                        </div>

                        {activeSection === null ? (
                            <>
                                <CVSectionDashboard cv={detailedCV} onSelectSection={setActiveSection} />
                                <div className="mt-4 pt-3 border-top text-end">
                                    <button onClick={() => handleDeleteCV(detailedCV.id)} className="btn btn-outline-danger btn-sm d-flex align-items-center gap-2 ms-auto"><Trash2 size={14} /> Delete CV</button>
                                </div>
                            </>
                        ) : (
                            <div className="row">
                                <div className="col-12">
                                    <div className="bg-white rounded-xl border shadow-sm p-4">
                                        <div className="mb-3 d-flex justify-content-end"> 
                                            <button onClick={() => setActiveSection(null)} className="btn btn-link text-decoration-none p-0 d-flex align-items-center gap-1 text-muted"><ChevronLeft size={16}/> Back to Dashboard</button>
                                        </div>
                                        {renderSectionDetail()}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="text-center py-5 border rounded-xl border-dashed bg-light">
                        <p className="text-muted mb-3">{cvs.length > 0 ? "Select a CV from the list above." : "No CVs found."}</p>
                        <button onClick={() => setShowCreateModal(true)} className="btn btn-primary">Create your first CV</button>
                    </div>
                )}
            </div>

            {showImportModal && (
                <ImportCVModal 
                    onClose={() => setShowImportModal(false)}
                    onStartBackgroundImport={handleStartImport}
                />
            )}

            {showExportModal && detailedCV && (
                <ExportCVModal cvId={detailedCV.id} onClose={() => setShowExportModal(false)} />
            )}

            {showCreateModal && (
                <div className="modal fade show d-block" style={{background: 'rgba(0,0,0,0.5)'}}>
                    <div className="modal-dialog modal-dialog-centered">
                        <div className="modal-content shadow-lg border-0">
                            <div className="modal-header border-bottom-0 pb-0">
                                <h5 className="modal-title fw-bold">Create New Master CV</h5>
                                <button type="button" className="btn-close" onClick={() => setShowCreateModal(false)}></button>
                            </div>
                            <form onSubmit={handleCreateCV}>
                                <div className="modal-body pt-4">
                                    <div className="mb-3">
                                        <label className="form-label fw-bold small text-muted text-uppercase">Internal Name (Required)</label>
                                        <input type="text" className="form-control" placeholder="e.g. Full Stack 2025" value={createFormData.name} onChange={(e) => setCreateFormData({...createFormData, name: e.target.value})} required autoFocus />
                                    </div>
                                    <div className="row mb-3">
                                        <div className="col-2">
                                            <label className="form-label fw-bold small text-muted text-uppercase">Title</label>
                                            <input type="text" className="form-control" placeholder="Dr." value={createFormData.title} onChange={(e) => setCreateFormData({...createFormData, title: e.target.value})} />
                                        </div>
                                        <div className="col-5">
                                            <label className="form-label fw-bold small text-muted text-uppercase">First Name</label>
                                            <input type="text" className="form-control" value={createFormData.first_name} onChange={(e) => setCreateFormData({...createFormData, first_name: e.target.value})} />
                                        </div>
                                        <div className="col-5">
                                            <label className="form-label fw-bold small text-muted text-uppercase">Last Name</label>
                                            <input type="text" className="form-control" value={createFormData.last_name} onChange={(e) => setCreateFormData({...createFormData, last_name: e.target.value})} />
                                        </div>
                                    </div>
                                    <div className="mb-3">
                                        <label className="form-label fw-bold small text-muted text-uppercase">Summary</label>
                                        <textarea className="form-control" rows="3" placeholder="Professional summary..." value={createFormData.summary} onChange={(e) => setCreateFormData({...createFormData, summary: e.target.value})} />
                                    </div>
                                </div>
                                <div className="modal-footer border-top-0">
                                    <button type="button" className="btn btn-light" onClick={() => setShowCreateModal(false)}>Cancel</button>
                                    <button type="submit" className="btn btn-primary px-4">Create CV</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CVManagerPage;