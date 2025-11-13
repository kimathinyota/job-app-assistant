// frontend/src/components/CVManagerPage.jsx
import React, { useState, useEffect } from 'react';
import {
    deleteBaseCV,
    fetchCVDetails,
    updateBaseCV,
    createBaseCV, 
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
    Edit2 
} from 'lucide-react';

import CVSelector from './cv/CVList';
import { getCVDisplayName } from '../utils/cvHelpers'; // <--- IMPORT HELPER

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
        <SectionCard 
            title="Experience" 
            count={cv.experiences?.length || 0} 
            icon={Briefcase} 
            colorClass="text-blue-600"
            onClick={() => onSelectSection('Experiences')} 
        />
        <SectionCard 
            title="Education" 
            count={cv.education?.length || 0} 
            icon={BookOpen} 
            colorClass="text-indigo-600"
            onClick={() => onSelectSection('Education')} 
        />
        <SectionCard 
            title="Projects" 
            count={cv.projects?.length || 0} 
            icon={Cpu} 
            colorClass="text-purple-600"
            onClick={() => onSelectSection('Projects')} 
        />
        <SectionCard 
            title="Master Skills" 
            count={cv.skills?.length || 0} 
            icon={Layers} 
            colorClass="text-emerald-600"
            onClick={() => onSelectSection('Skills')} 
        />
        <SectionCard 
            title="Achievements" 
            count={cv.achievements?.length || 0} 
            icon={Award} 
            colorClass="text-amber-500"
            onClick={() => onSelectSection('Achievements')} 
        />
        <SectionCard 
            title="Hobbies" 
            count={cv.hobbies?.length || 0} 
            icon={Smile} 
            colorClass="text-pink-500"
            onClick={() => onSelectSection('Hobbies')} 
        />
    </div>
);

const CVManagerPage = ({ cvs, setActiveView, reloadData, initialSection }) => {
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [activeSection, setActiveSection] = useState(null);
    
    // Edit Header State (Added Title)
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({ name: '', first_name: '', last_name: '', title: '', summary: '' });

    // Create Modal State (Added Title)
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [createFormData, setCreateFormData] = useState({ name: '', first_name: '', last_name: '', title: '', summary: '' });

    // --- EFFECTS ---
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
        if (initialSection) {
            setActiveSection(initialSection);
        } else {
            setActiveSection(null);
        }

        if (selectedCVId) {
            fetchAndSetDetails(selectedCVId);
        } else if (cvs.length > 0) {
            setSelectedCVId(cvs[0].id);
        } else {
             setDetailedCV(null);
        }
    }, [selectedCVId, cvs.length, initialSection]);


    // --- HANDLERS: Header Edit ---
     const handleStartEditHeader = () => {
        setEditFormData({
            name: detailedCV.name,
            first_name: detailedCV.first_name || '',
            last_name: detailedCV.last_name || '',
            title: detailedCV.title || '', // <--- Load title
            summary: detailedCV.summary || ''
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
        } catch (error) {
            alert('Failed to update CV.');
            console.error(error);
        }
    };

    // --- HANDLERS: Create CV ---
    const handleCreateCV = async (e) => {
        e.preventDefault();
        if (!createFormData.name.trim()) return alert("Internal Name is required");
        
        try {
            const newCV = await createBaseCV(
                createFormData.name, 
                createFormData.first_name, 
                createFormData.last_name, 
                createFormData.title, // <--- Pass title
                createFormData.summary
            );
            await reloadData(); 
            setSelectedCVId(newCV.id); 
            setShowCreateModal(false);
            setCreateFormData({ name: '', first_name: '', last_name: '', title: '', summary: '' });
        } catch (err) {
            console.error(err);
            alert("Failed to create CV");
        }
    };

    // --- Unified Item Handler (No changes here) ---
    const handleAddOrUpdateNestedItem = async (cvId, data, itemType) => {
        // ... (Same as before) ...
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
                if (!apiFn) throw new Error(`Update API function not configured for ${itemType}`);
                if (itemType === 'Skill') await apiFn(cvId, data);
                else await apiFn(cvId, itemId, data);
                alert(`${itemType} updated successfully!`);
            } else {
                apiFn = apiFunctions[itemType]?.add;
                if (!apiFn) throw new Error(`Create API function not configured for ${itemType}`);
                await apiFn(cvId, data);
                alert(`${itemType} added successfully!`);
            }
            await reloadData(); 
            await fetchAndSetDetails(cvId); 
        } catch (error) {
            alert(`Failed to ${isUpdating ? 'update' : 'add'} ${itemType}. Check console.`);
            console.error(error);
        }
    };

    // --- Delete Handlers (No changes here) ---
    const handleDeleteCV = async (cvIdToDelete) => {
        if (window.confirm("Delete this master CV? This action cannot be undone.")) {
            try {
                await deleteBaseCV(cvIdToDelete);
                alert("CV deleted!");
                await reloadData();
                setSelectedCVId(null);
                setDetailedCV(null);
            } catch (error) {
                alert("Failed to delete CV."); console.error(error);
            }
        }
    };

     const handleDeleteNested = async (cvIdToDeleteFrom, itemId, listName) => {
        if (window.confirm(`Permanently delete this item?`)) {
            try {
                await deleteNestedItem(cvIdToDeleteFrom, itemId, listName);
                fetchAndSetDetails(cvIdToDeleteFrom); 
            } catch (error) {
                alert(`Error deleting item.`); console.error(error);
            }
        }
    };

    const masterSkills = detailedCV?.skills || [];
    const masterAchievements = detailedCV?.achievements || [];
    const masterExperiences = detailedCV?.experiences || [];
    const masterEducation = detailedCV?.education || [];
    const masterProjects = detailedCV?.projects || [];
    const masterHobbies = detailedCV?.hobbies || [];

    const renderSectionDetail = () => {
         if (!activeSection) return null;
        const commonProps = {
            cvId: detailedCV.id,
            allSkills: masterSkills,
            allAchievements: masterAchievements,
            onSubmit: handleAddOrUpdateNestedItem,
            onDelete: handleDeleteNested,
            onBack: () => setActiveSection(null)
        };
        const sections = {
            'Experiences': <ExperienceManager {...commonProps} experiences={masterExperiences} />,
            'Education': <EducationManager {...commonProps} education={masterEducation} />,
            'Hobbies': <HobbyManager {...commonProps} hobbies={masterHobbies} />,
            'Projects': <ProjectManager {...commonProps} projects={masterProjects} allExperiences={masterExperiences} allEducation={masterEducation} allHobbies={masterHobbies} />,
            'Skills': <SkillsetManager {...commonProps} allExperiences={masterExperiences} allEducation={masterEducation} allProjects={masterProjects} allHobbies={masterHobbies} />,
            'Achievements': <AchievementHub {...commonProps} allExperiences={masterExperiences} allEducation={masterEducation} allProjects={masterProjects} allHobbies={masterHobbies} />
        };
        return sections[activeSection] || <p>Section not found.</p>; 
    };

    // --- Main Render ---
    return (
        <div className="text-start pb-5">
            <style>
                {`
                .hover-lift { transition: transform 0.2s ease, box-shadow 0.2s ease; }
                .hover-lift:hover { transform: translateY(-4px); box-shadow: 0 10px 20px rgba(0,0,0,0.1) !important; }
                .cursor-pointer { cursor: pointer; }
                `}
            </style>

            <div className="mb-4">
                <h2 className="fw-bold text-dark mb-3">CV Library</h2>
                <CVSelector 
                    cvs={cvs} 
                    onSelect={setSelectedCVId} 
                    selectedCVId={selectedCVId} 
                    onCreate={() => setShowCreateModal(true)} 
                />
            </div>
            
            <div className="mt-3">
                {loadingDetails ? (
                    <div className="text-center py-5 text-muted">Loading CV details...</div>
                ) : detailedCV ? (
                    <div className="animate-fade-in">
                        
                        {/* CV Header Card */}
                        <div className="bg-white rounded-xl border shadow-sm p-4 mb-4">
                            {!isEditingHeader ? (
                                <div className="d-flex justify-content-between align-items-start">
                                    <div>
                                        <div className="d-flex align-items-center gap-2 mb-1">
                                            {/* Use Display Name */}
                                            <h3 className="h4 fw-bold text-primary mb-0">
                                                {getCVDisplayName(detailedCV)}
                                            </h3>
                                            <span className="badge bg-light text-muted border">Master Version</span>
                                        </div>
                                        {/* Show Internal Name if different from display name */}
                                        <p className="text-muted small mb-1">
                                            Internal ID: <span className="fw-medium text-dark">{detailedCV.name}</span>
                                        </p>
                                        <p className="text-muted mb-0" style={{whiteSpace: 'pre-wrap'}}>
                                            {detailedCV.summary || <span className="fst-italic opacity-50">No summary provided. Click edit to add one.</span>}
                                        </p>
                                    </div>
                                    <button 
                                        onClick={handleStartEditHeader} 
                                        className="btn btn-outline-secondary btn-sm d-flex align-items-center gap-2"
                                    >
                                        <Edit2 size={14}/> Edit Header
                                    </button>
                                </div>
                            ) : (
                                <form onSubmit={handleUpdateCVHeader} className="bg-light p-3 rounded">
                                    <div className="row g-3 mb-3">
                                        <div className="col-md-12">
                                            <label className="form-label fw-bold small text-uppercase">Internal Name (Private)</label>
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                value={editFormData.name} 
                                                onChange={e => setEditFormData({...editFormData, name: e.target.value})} 
                                                required 
                                            />
                                        </div>
                                        <div className="col-md-2">
                                            <label className="form-label fw-bold small text-uppercase">Title</label>
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                placeholder="Dr. / Senior"
                                                value={editFormData.title} 
                                                onChange={e => setEditFormData({...editFormData, title: e.target.value})} 
                                            />
                                        </div>
                                        <div className="col-md-5">
                                            <label className="form-label fw-bold small text-uppercase">First Name</label>
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                value={editFormData.first_name} 
                                                onChange={e => setEditFormData({...editFormData, first_name: e.target.value})} 
                                            />
                                        </div>
                                        <div className="col-md-5">
                                            <label className="form-label fw-bold small text-uppercase">Last Name</label>
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                value={editFormData.last_name} 
                                                onChange={e => setEditFormData({...editFormData, last_name: e.target.value})} 
                                            />
                                        </div>
                                        <div className="col-12">
                                            <label className="form-label fw-bold small text-uppercase">Summary / Bio</label>
                                            <textarea 
                                                className="form-control" 
                                                rows="3" 
                                                value={editFormData.summary} 
                                                onChange={e => setEditFormData({...editFormData, summary: e.target.value})} 
                                            />
                                        </div>
                                    </div>
                                    <div className="d-flex gap-2">
                                        <button type="submit" className="btn btn-primary btn-sm">Save Changes</button> 
                                        <button type="button" className="btn btn-light btn-sm border" onClick={() => setIsEditingHeader(false)}>Cancel</button>
                                    </div>
                                </form>
                            )}
                        </div>

                        {/* Content */}
                        {activeSection === null ? (
                            <>
                                <CVSectionDashboard cv={detailedCV} onSelectSection={setActiveSection} />
                                <div className="mt-4 pt-3 border-top text-end">
                                    <button onClick={() => handleDeleteCV(detailedCV.id)} className="btn btn-outline-danger btn-sm d-flex align-items-center gap-2 ms-auto">
                                        <Trash2 size={14} /> Delete CV
                                    </button>
                                </div>
                            </>
                        ) : (
                            <div className="bg-white rounded-xl border shadow-sm p-4">
                                <div className="mb-3 d-flex justify-content-end"> 
                                    <button 
                                        onClick={() => setActiveSection(null)} 
                                        className="btn btn-link text-decoration-none p-0 d-flex align-items-center gap-1 text-muted"
                                    >
                                        <ChevronLeft size={16}/> Back to Dashboard
                                    </button>
                                </div>
                                {renderSectionDetail()}
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="text-center py-5 border rounded-xl border-dashed bg-light">
                        <p className="text-muted mb-3">No CV Selected.</p>
                        <button onClick={() => setShowCreateModal(true)} className="btn btn-primary">Create your first CV</button>
                    </div>
                )}
            </div>

            {/* Create CV Modal */}
            {showCreateModal && (
                <>
                    <div className="modal-backdrop fade show"></div>
                    <div className="modal fade show d-block" tabIndex="-1">
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
                                            <input 
                                                type="text" 
                                                className="form-control" 
                                                placeholder="e.g. Full Stack 2025" 
                                                value={createFormData.name} 
                                                onChange={(e) => setCreateFormData({...createFormData, name: e.target.value})} 
                                                required 
                                                autoFocus
                                            />
                                        </div>
                                        <div className="row mb-3">
                                            <div className="col-2">
                                                <label className="form-label fw-bold small text-muted text-uppercase">Title</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control" 
                                                    placeholder="Dr."
                                                    value={createFormData.title} 
                                                    onChange={(e) => setCreateFormData({...createFormData, title: e.target.value})} 
                                                />
                                            </div>
                                            <div className="col-5">
                                                <label className="form-label fw-bold small text-muted text-uppercase">First Name</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control" 
                                                    value={createFormData.first_name} 
                                                    onChange={(e) => setCreateFormData({...createFormData, first_name: e.target.value})} 
                                                />
                                            </div>
                                            <div className="col-5">
                                                <label className="form-label fw-bold small text-muted text-uppercase">Last Name</label>
                                                <input 
                                                    type="text" 
                                                    className="form-control" 
                                                    value={createFormData.last_name} 
                                                    onChange={(e) => setCreateFormData({...createFormData, last_name: e.target.value})} 
                                                />
                                            </div>
                                        </div>
                                        <div className="mb-3">
                                            <label className="form-label fw-bold small text-muted text-uppercase">Summary</label>
                                            <textarea 
                                                className="form-control" 
                                                rows="3" 
                                                placeholder="Professional summary..."
                                                value={createFormData.summary} 
                                                onChange={(e) => setCreateFormData({...createFormData, summary: e.target.value})} 
                                            />
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
                </>
            )}
        </div>
    );
};

export default CVManagerPage;