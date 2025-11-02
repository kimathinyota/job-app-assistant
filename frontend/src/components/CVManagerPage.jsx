// frontend/src/components/CVManagerPage.jsx
import React, { useState, useEffect } from 'react';
import {
    deleteBaseCV,
    fetchCVDetails,
    updateBaseCV,
    // *** MODIFICATION: Import new complex functions ***
    addExperienceComplex,
    updateExperienceComplex,
    // *** END MODIFICATION ***
    addSkill,
    addAchievement,
    addEducation,
    addProject,
    addHobby,
    deleteNestedItem
} from '../api/cvClient';

// --- Component Imports ---
import CVSelector from './cv/CVList';
import NestedList from './cv/NestedList';
import ExperienceManager from './cv/ExperienceManager';
import EducationForm from './cv/EducationForm';
import ProjectForm from './cv/ProjectForm';
import SkillForm from './cv/SkillForm';
import AchievementForm from './cv/AchievementForm';
import HobbyForm from './cv/HobbyForm';


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
    
    // This state is still used by OTHER forms
    const [editingItem, setEditingItem] = useState(null); 

    // --- Data Fetching Logic (unchanged) ---
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
        setEditingItem(null);

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
        // ... (unchanged)
        setEditFormData({
            name: detailedCV.name,
            summary: detailedCV.summary || ''
        });
        setIsEditingHeader(true);
    };

    const handleCancelEditHeader = () => setIsEditingHeader(false);

     const handleUpdateCVHeader = async (e) => {
        // ... (unchanged)
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

    const handleStartEditItem = (item, sectionName) => {
        // This handler is now ONLY for non-experience sections
        setEditingItem(item);
        setActiveSection(sectionName);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const handleCancelEdit = () => {
        // This clears state for ALL sections
        setEditingItem(null);
        setActiveSection(null); // Go back to dashboard on cancel
    };

    // --- *** THIS IS THE NEW, SIMPLIFIED FUNCTION *** ---
    const handleAddOrUpdateNestedItem = async (cvId, data, itemType) => {
        const isUpdating = Boolean(data.id);
        const itemId = data.id;

        const apiFunctions = {
            // *** MODIFICATION: Use complex functions for Experience ***
            'Experience': { add: addExperienceComplex, update: updateExperienceComplex },
            // *** END MODIFICATION ***
            'Education': { add: addEducation, update: /* updateEducation */ addEducation },
            'Project': { add: addProject, update: /* updateProject */ addProject },
            'Hobby': { add: addHobby, update: /* updateHobby */ addHobby },
            'Achievement': { add: addAchievement, update: /* updateAchievement */ addAchievement },
            'Skill': { add: addSkill, update: /* updateSkill */ addSkill },
        };
        
        console.log(`[CVManager] Starting ${isUpdating ? 'update' : 'create'} for ${itemType}.`, "Sending data to backend:", data);

        try {
            const { add: addFn, update: updateFn } = apiFunctions[itemType] || {};
            if (!addFn || !updateFn) throw new Error(`API functions not configured for ${itemType}`);

            if (isUpdating) {
                console.log(`[CVManager] Calling update API for ${itemType} (${itemId})...`);
                // For Experience, 'data' is the full complex payload
                // For other items, it's the simple { name: ... } payload
                await updateFn(cvId, itemId, data);
                alert(`${itemType} updated successfully!`);
                setEditingItem(null); 
            } else {
                console.log(`[CVManager] Calling create API for ${itemType}...`);
                // For Experience, 'data' is the full complex payload
                // For other items, it's the simple { name: ... } payload
                await addFn(cvId, data);
                alert(`${itemType} added successfully!`);
            }
            
            console.log("[CVManager] Process complete. Refetching details and reloading core data.");
            // We reload *all* data here to ensure subsequent actions (like in ExperienceManager)
            // have the absolute latest state for skills/achievements
            await reloadData(); 
            await fetchAndSetDetails(cvId); // Final refetch

        } catch (error) {
            alert(`Failed to ${isUpdating ? 'update' : 'add'} ${itemType}. Check console.`);
            console.error(`[CVManager] API call FAILED:`, error);
        }
    };
    // --- *** END OF NEW FUNCTION *** ---

    const handleDeleteCV = async (cvIdToDelete) => {
        // ... (unchanged)
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
        // ... (unchanged)
        if (window.confirm(`Delete this item from ${listName}?`)) {
            try {
                await deleteNestedItem(cvIdToDeleteFrom, itemId, listName);
                fetchAndSetDetails(cvIdToDeleteFrom);
            } catch (error) {
                alert(`Error deleting item.`); console.error(error);
            }
        }
    };

    // --- RENDER LOGIC ---
    const masterSkills = detailedCV?.skills || [];
    const masterAchievements = detailedCV?.achievements || [];

    if (cvs.length === 0) {
        // ... (unchanged)
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

        // --- Special Case for Experiences ---
        if (activeSection === 'Experiences') {
            return (
                <ExperienceManager
                    cvId={detailedCV.id}
                    experiences={detailedCV.experiences || []}
                    allSkills={masterSkills}
                    allAchievements={masterAchievements}
                    onSubmit={handleAddOrUpdateNestedItem} // Pass the main handler
                    onDelete={handleDeleteNested}         // Pass the main handler
                    onBack={handleCancelEdit} // Use the main cancel handler
                />
            );
        }

        // --- Default logic for all OTHER sections ---
        const sectionMap = {
            'Education': { Form: EducationForm, items: detailedCV?.education || [], listName: 'education', formProps: { allSkills: masterSkills } },
            'Projects': { Form: ProjectForm, items: detailedCV?.projects || [], listName: 'projects', formProps: { allSkills: masterSkills, allAchievements: masterAchievements } },
            'Skills': { Form: SkillForm, items: detailedCV?.skills || [], listName: 'skills', formProps: {} },
            'Achievements': { Form: AchievementForm, items: detailedCV?.achievements || [], listName: 'achievements', formProps: { allSkills: masterSkills } },
            'Hobbies': { Form: HobbyForm, items: detailedCV?.hobbies || [], listName: 'hobbies', formProps: { allSkills: masterSkills, allAchievements: masterAchievements } },
        };

        const current = sectionMap[activeSection];
        if (!current) return <p>Section not found.</p>;

        const { Form, items, listName, formProps } = current;
        const currentEditingItem = (editingItem && editingItem.id && items.some(i => i.id === editingItem.id)) ? editingItem : null;
        const noun = listName.slice(0, -1); 

        return (
            <div>
                 <button onClick={handleCancelEdit} className="btn btn-secondary mb-3">
                    &larr; Back to CV Dashboard
                </button>

                {/* Use Bootstrap's grid system for layout */}
                <div className="row g-4 mt-1">
                    <div className="col-lg-5">
                        <h3 className="h4 border-bottom pb-2 text-capitalize">
                            {currentEditingItem ? `Edit ${noun}` : `Add New ${noun}`}
                        </h3>
                        <Form
                            key={currentEditingItem?.id || 'new-item-form'}
                            onSubmit={handleAddOrUpdateNestedItem}
                            cvId={detailedCV.id}
                            initialData={currentEditingItem}
                            onCancelEdit={handleCancelEdit}
                            {...formProps}
                        />
                    </div>
                    <div className="col-lg-7">
                        {!currentEditingItem && <h3 className="h4 border-bottom pb-2 text-capitalize">Existing {listName}</h3>}
                        
                        <NestedList
                            cvId={detailedCV.id}
                            items={items}
                            listName={listName}
                            onDelete={handleDeleteNested}
                            onEdit={(item) => handleStartEditItem(item, activeSection)}
                            allSkills={masterSkills}
                            allAchievements={masterAchievements}
                        />
                    </div>
                </div>
            </div>
        );
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
