// frontend/src/components/CVManagerPage.jsx
import React, { useState, useEffect } from 'react';
import {
    deleteBaseCV,
    fetchCVDetails,
    updateBaseCV,
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
    // addAchievement, // No longer adding/editing achievements
    
    deleteNestedItem
} from '../api/cvClient';

// --- Component Imports ---
import CVSelector from './cv/CVList';
// import NestedList from './cv//NestedList'; // No longer needed for Achievements

// Managers
import ExperienceManager from './cv/ExperienceManager';
import EducationManager from './cv/EducationManager';
import HobbyManager from './cv/HobbyManager';
import ProjectManager from './cv/ProjectManager';
import SkillsetManager from './cv/SkillsetManager';
import AchievementHub from './cv/AchievementHub'; // <-- NEW

// Simple Forms
// import AchievementForm from './cv/AchievementForm'; // <-- REMOVED


// --- SectionButton (Bootstrap Version) ---
const SectionButton = ({ title, count, onClick }) => (
    <button
        onClick={onClick}
        className="btn btn-light text-start p-3 shadow-sm"
        style={{ flex: '1 1 250px', minHeight: '100px' }}
    >
        <span className="fs-5 fw-bold text-primary d-block">{title}</span>
        <span className="text-muted">({count} items)</span>
    </button>
);

// --- CVSectionDashboard (Bootstrap Version) ---
const CVSectionDashboard = ({ cv, onSelectSection }) => (
    <div className="d-flex flex-wrap gap-3 justify-content-center py-3">
        <SectionButton title="Experiences" count={cv.experiences?.length || 0} onClick={() => onSelectSection('Experiences')} />
        <SectionButton title="Education" count={cv.education?.length || 0} onClick={() => onSelectSection('Education')} />
        <SectionButton title="Projects" count={cv.projects?.length || 0} onClick={() => onSelectSection('Projects')} />
        <SectionButton title="Master Skills" count={cv.skills?.length || 0} onClick={() => onSelectSection('Skills')} />
        <SectionButton title="Master Achievements" count={cv.achievements?.length || 0} onClick={() => onSelectSection('Achievements')} />
        <SectionButton title="Hobbies" count={cv.hobbies?.length || 0} onClick={() => onSelectSection('Hobbies')} />
    </div>
);


const CVManagerPage = ({ cvs, setActiveView, reloadData }) => {
    // --- STATE DECLARATIONS ---
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [activeSection, setActiveSection] = useState(null);
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({ name: '', summary: '' });
    
    // This state is no longer used
    // const [editingItem, setEditingItem] = useState(null); 

    // --- Data Fetching Logic ---
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

    // --- useEffect Hook ---
     useEffect(() => {
        setActiveSection(null);
        setIsEditingHeader(false);
        // setEditingItem(null); // No longer needed

        if (selectedCVId) {
            fetchAndSetDetails(selectedCVId);
        } else if (cvs.length > 0) {
            setSelectedCVId(cvs[0].id);
        } else {
             setDetailedCV(null);
        }
    }, [selectedCVId, cvs.length]);


    // --- CRUD Handlers ---
     const handleStartEditHeader = () => {
        setEditFormData({
            name: detailedCV.name,
            summary: detailedCV.summary || ''
        });
        setIsEditingHeader(true);
    };

    const handleCancelEditHeader = () => setIsEditingHeader(false);

     const handleUpdateCVHeader = async (e) => {
        e.preventDefault();
        if (!editFormData.name.trim()) return alert('CV Name is required.');
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

    // This handler is no longer needed for Achievements
    // const handleStartEditItem = (item, sectionName) => { ... };

    const handleCancelEdit = () => {
        // This clears state for ALL sections
        // setEditingItem(null); // No longer needed
        setActiveSection(null); // Go back to dashboard on cancel
    };

    // --- This handler now manages all complex types AND simple types ---
    const handleAddOrUpdateNestedItem = async (cvId, data, itemType) => {
        const isUpdating = Boolean(data.id);
        const itemId = data.id;

        // --- UPDATED: 'Achievement' is removed ---
        const apiFunctions = {
            'Experience': { add: addExperienceComplex, update: updateExperienceComplex },
            'Education': { add: addEducationComplex, update: updateEducationComplex },
            'Hobby': { add: addHobbyComplex, update: updateHobbyComplex },
            'Project': { add: addProjectComplex, update: updateProjectComplex },
            'Skill': { add: addSkill, update: addSkill }, // Skill still uses simple add/upsert
        };
        
        console.log(`[CVManager] Starting ${isUpdating ? 'update' : 'create'} for ${itemType}.`, "Sending data to backend:", data);

        try {
            let apiFn;
            if (isUpdating) {
                apiFn = apiFunctions[itemType]?.update;
                if (!apiFn) throw new Error(`Update API function not configured for ${itemType}`);
                
                if (itemType === 'Skill') {
                    console.log(`[CVManager] Calling simple update API (via POST) for ${itemType} (${itemId})...`);
                    await apiFn(cvId, data); // Calls addSkill(cvId, data)
                } else {
                    console.log(`[CVManager] Calling complex update API (via PATCH) for ${itemType} (${itemId})...`);
                    await apiFn(cvId, itemId, data); // Calls update...Complex(cvId, itemId, data)
                }
                alert(`${itemType} updated successfully!`);
                // setEditingItem(null); // No longer needed
            } else {
                apiFn = apiFunctions[itemType]?.add;
                if (!apiFn) throw new Error(`Create API function not configured for ${itemType}`);

                console.log(`[CVManager] Calling create API for ${itemType}...`);
                await apiFn(cvId, data);
                alert(`${itemType} added successfully!`);
            }
            
            console.log("[CVManager] Process complete. Refetching details and reloading core data.");
            await reloadData(); 
            await fetchAndSetDetails(cvId); // Final refetch

        } catch (error) {
            alert(`Failed to ${isUpdating ? 'update' : 'add'} ${itemType}. Check console.`);
            console.error(`[CVManager] API call FAILED:`, error);
        }
    };
    // --- *** END OF MODIFIED FUNCTION *** ---

    const handleDeleteCV = async (cvIdToDelete) => {
        if (window.confirm("Delete this master CV?")) {
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
        // We can ask for confirmation again here
        if (window.confirm(`Are you sure you want to permanently delete this ${listName} item?`)) {
            try {
                await deleteNestedItem(cvIdToDeleteFrom, itemId, listName);
                fetchAndSetDetails(cvIdToDeleteFrom); // Refetch details
            } catch (error) {
                alert(`Error deleting item.`); console.error(error);
            }
        }
    };

    // --- RENDER LOGIC ---
    const masterSkills = detailedCV?.skills || [];
    const masterAchievements = detailedCV?.achievements || [];
    const masterExperiences = detailedCV?.experiences || [];
    const masterEducation = detailedCV?.education || [];
    const masterProjects = detailedCV?.projects || [];
    const masterHobbies = detailedCV?.hobbies || [];


    if (cvs.length === 0) {
        return (
             <div className="text-center p-5 border border-primary-subtle rounded bg-primary-light-subtle">
                <h3 className="text-primary">No CVs Found</h3>
                <p>Please use the form on the Dashboard Home to create your first base CV.</p>
                <button className="btn btn-primary" onClick={() => setActiveView('Dashboard')}>Go to Dashboard</button>
            </div>
        );
    }

    const renderSectionDetail = () => {
         if (!activeSection) return null;

        // --- Manager Components ---
        if (activeSection === 'Experiences') {
            return (
                <ExperienceManager
                    cvId={detailedCV.id}
                    experiences={masterExperiences}
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    onSubmit={handleAddOrUpdateNestedItem}
                    onDelete={handleDeleteNested}
                    onBack={handleCancelEdit}
                />
            );
        }
        if (activeSection === 'Education') {
            return (
                <EducationManager
                    cvId={detailedCV.id}
                    education={masterEducation}
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    onSubmit={handleAddOrUpdateNestedItem}
                    onDelete={handleDeleteNested}
                    onBack={handleCancelEdit}
                />
            );
        }
        if (activeSection === 'Hobbies') {
            return (
                <HobbyManager
                    cvId={detailedCV.id}
                    hobbies={masterHobbies}
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    onSubmit={handleAddOrUpdateNestedItem}
                    onDelete={handleDeleteNested}
                    onBack={handleCancelEdit}
                />
            );
        }
        if (activeSection === 'Projects') {
            return (
                <ProjectManager
                    cvId={detailedCV.id}
                    projects={masterProjects}
                    allExperiences={masterExperiences}
                    allEducation={masterEducation}
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    onSubmit={handleAddOrUpdateNestedItem}
                    onDelete={handleDeleteNested}
                    onBack={handleCancelEdit}
                />
            );
        }
        
        if (activeSection === 'Skills') {
            return (
                <SkillsetManager
                    cvId={detailedCV.id}
                    // Pass ALL CV data to build the map
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    allExperiences={masterExperiences}
                    allEducation={masterEducation}
                    allProjects={masterProjects}
                    allHobbies={masterHobbies}
                    onSubmit={handleAddOrUpdateNestedItem}
                    onDelete={handleDeleteNested}
                    onBack={handleCancelEdit}
                />
            );
        }
        
        // --- *** NEW: Special Case for Achievements *** ---
        if (activeSection === 'Achievements') {
            return (
                <AchievementHub
                    cvId={detailedCV.id}
                    // Pass ALL CV data to build the map
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    allExperiences={masterExperiences}
                    allEducation={masterEducation}
                    allProjects={masterProjects}
                    allHobbies={masterHobbies}
                    onDelete={handleDeleteNested}
                    onBack={handleCancelEdit}
                />
            );
        }
        // --- *** END NEW CASE *** ---

        // This section is now empty, as all sections are handled by managers.
        const sectionMap = {};
        const current = sectionMap[activeSection];
        if (!current) return <p>Section not found.</p>; 

        // ... (Old split-screen logic is no longer needed) ...
    };


    // --- Final Return Block (unchanged) ---
    return (
        <div className="text-start">
            <h2>CV Manager & Editor</h2>
            <CVSelector cvs={cvs} onSelect={setSelectedCVId} selectedCVId={selectedCVId} />
            
            {/* Main content card */}
            <div className="card shadow-sm mt-3">
                <div className="card-body p-4">
                    {loadingDetails ? (
                        <p className="text-center text-primary">Loading CV details...</p>
                    ) : detailedCV ? (
                        <>
                            {/* CV Header */}
                            {!isEditingHeader ? (
                                <div className="mb-4 pb-4 border-bottom border-primary-subtle position-relative">
                                    <h3 className="h4 m-0 text-primary">{detailedCV.name}</h3>
                                    <p className="m-0 mt-1 fst-italic text-muted" style={{whiteSpace: 'pre-wrap'}}>
                                        {detailedCV.summary || "No summary."}
                                    </p>
                                    <button 
                                        onClick={handleStartEditHeader} 
                                        className="btn btn-outline-secondary btn-sm position-absolute"
                                        style={{ top: '0', right: '0' }}
                                    >
                                        Edit Details
                                    </button>
                                </div>
                            ) : (
                                <form onSubmit={handleUpdateCVHeader} className="mb-4 p-3 border border-primary-subtle rounded bg-light-subtle">
                                    <div className="mb-2">
                                        <label htmlFor="cvName" className="form-label fw-bold">Name:</label>
                                        <input 
                                            id="cvName"
                                            type="text" 
                                            value={editFormData.name} 
                                            onChange={(e) => setEditFormData({...editFormData, name: e.target.value})} 
                                            required 
                                            className="form-control"
                                        />
                                    </div>
                                    <div className="mb-3">
                                        <label htmlFor="cvSummary" className="form-label fw-bold">Summary:</label>
                                        <textarea 
                                            id="cvSummary"
                                            value={editFormData.summary} 
                                            onChange={(e) => setEditFormData({...editFormData, summary: e.target.value})} 
                                            className="form-control"
                                            rows="3"
                                        />
                                    </div>
                                    <button type="submit" className="btn btn-primary me-2">Save</button> 
                                    <button type="button" className="btn btn-secondary" onClick={handleCancelEditHeader}>Cancel</button>
                                </form>
                            )}

                            {/* CV Body */}
                            {activeSection === null ? (
                                <CVSectionDashboard cv={detailedCV} onSelectSection={setActiveSection} />
                            ) : (
                                renderSectionDetail()
                            )}

                            {/* Footer */}
                            <div className="border-top mt-4 pt-4 text-end">
                                <button onClick={() => handleDeleteCV(detailedCV.id)} className="btn btn-danger">
                                    Delete This Entire CV
                                </button>
                            </div>
                        </>
                    ) : (
                        selectedCVId ? <p>Loading CV details...</p> : <p className="text-center text-muted">Select a CV to manage.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CVManagerPage;