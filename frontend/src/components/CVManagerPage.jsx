import React, { useState, useEffect } from 'react';
import { 
    deleteBaseCV, 
    fetchCVDetails, 
    addExperience, 
    addSkill,
    addAchievement,
    addEducation, 
    addProject,   
    deleteNestedItem
} from '../api/cvClient'; 


// --- Component Imports ---
// import CVSelector from './cv/CVList'; 
// import NestedList from './cv/NestedList'; 
import ExperienceForm from './cv/ExperienceForm';
import EducationForm from './cv/EducationForm'; 
import ProjectForm from './cv/ProjectForm';     
import SkillForm from './cv/SkillForm'; 
import AchievementForm from './cv/AchievementForm';


// Main CV Manager Component
const CVManagerPage = ({ cvs, setActiveView, reloadData }) => {
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);

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

    useEffect(() => {
        if (selectedCVId) {
            fetchAndSetDetails(selectedCVId);
        } else if (cvs.length > 0) {
            setSelectedCVId(cvs[0].id);
        } else {
             setDetailedCV(null);
        }
    }, [selectedCVId, cvs.length]);


    // --- CRUD Handlers ---

    // NEW: Handles creation of a new master skill from within the nested forms (Experience/Project)
    const handleCreateSkillAndLink = async (cvId, skillData) => {
        try {
            // 1. Create the new skill in the master list
            const newSkill = await addSkill(cvId, skillData);
            alert(`New Skill "${newSkill.name}" created successfully!`);
            
            // 2. Refresh all data (master list and details) so the new skill appears in the linker
            await reloadData(); // Reloads master list in App.jsx
            fetchAndSetDetails(cvId); // Reloads the detailed CV to get the new skill list
        } catch (error) {
            alert('Failed to create new skill. Check console.');
            console.error(error);
        }
    };


    const handleDeleteCV = async (cvId) => {
        if (window.confirm("Are you sure you want to delete this master CV? This is irreversible.")) {
            try {
                await deleteBaseCV(cvId);
                alert("CV deleted successfully!");
                await reloadData();
                setSelectedCVId(null);
                setDetailedCV(null);
            } catch (error) {
                alert("Failed to delete CV.");
                console.error(error);
            }
        }
    };

    // Generic handler for nested ADD actions (Experience, Skill, etc.)
    const handleAddNestedItem = async (cvId, data, addFunction, itemType) => {
        try {
            await addFunction(cvId, data);
            alert(`${itemType} added successfully!`);
            
            // If the added item is a top-level resource (like Achievement/Skill), refresh everything.
            if (itemType === 'Achievement' || itemType === 'Skill') {
                 await reloadData();
            } else {
                 fetchAndSetDetails(cvId); // Just reload the current CV's details
            }
        } catch (error) {
            alert(`Failed to add ${itemType}. Check console.`);
            console.error(error);
        }
    };

    // Handler for deleting a nested item
    const handleDeleteNested = async (cvId, itemId, listName) => {
        if (window.confirm(`Delete this item from ${listName}?`)) {
            try {
                await deleteNestedItem(cvId, itemId, listName);
                fetchAndSetDetails(cvId);
            } catch (error) {
                alert(`Error deleting item from ${listName}. Check console.`);
                console.error(error);
            }
        }
    };


    // --- RENDER LOGIC ---

    const masterSkills = detailedCV ? detailedCV.skills : []; 

    if (cvs.length === 0) {
        return (
            <div style={{ textAlign: 'center', padding: '50px', border: '2px dashed #007bff', borderRadius: '8px' }}>
                <h3 style={{ color: '#007bff' }}>No CVs Found</h3>
                <p>Please use the form on the Dashboard Home to create your first base CV.</p>
                <button onClick={() => setActiveView('Dashboard')}>Go to Dashboard</button>
            </div>
        );
    }

    return (
        <div style={{ textAlign: 'left' }}>
            <h2>CV Manager & Editor</h2>
            
            {/* 1. CV Selector (Use CVList) */}
            {/* ... */}

            {/* Detail/Editor Area */}
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px', backgroundColor: '#fff' }}>
                {loadingDetails ? (
                    <p style={{ textAlign: 'center' }}>Loading detailed components...</p>
                ) : detailedCV ? (
                    <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap' }}>
                        
                        {/* --- LEFT COLUMN: INPUT FORMS (350px) --- */}
                        <div style={{ flex: '1 1 350px', minWidth: '350px' }}>
                            <h3 style={{ borderBottom: '1px solid #eee', paddingBottom: '10px' }}>Add New Components</h3>
                            
                            {/* Experience Form (CONTEXTUAL SKILL CREATION/LINKING) */}
                            <ExperienceForm 
                                onSubmit={handleAddNestedItem} 
                                cvId={detailedCV.id}
                                allSkills={masterSkills} 
                                onSkillCreate={handleCreateSkillAndLink} 
                            />
                            {/* Education Form */}
                            <EducationForm 
                                onSubmit={handleAddNestedItem} 
                                cvId={detailedCV.id}
                                allSkills={masterSkills}
                                onSkillCreate={handleCreateSkillAndLink} 
                            />
                            {/* Project Form (CONTEXTUAL SKILL CREATION/LINKING) */}
                            <ProjectForm 
                                onSubmit={handleAddNestedItem} 
                                cvId={detailedCV.id}
                                allSkills={masterSkills}
                                onSkillCreate={handleCreateSkillAndLink}
                            />
                            {/* Master Skill Form (STANDALONE) */}
                            <SkillForm 
                                onSubmit={(cvId, data) => handleAddNestedItem(cvId, data, addSkill, 'Skill')} 
                                cvId={detailedCV.id}
                            />
                            {/* Master Achievement Form */}
                             <AchievementForm 
                                onSubmit={(cvId, data) => handleAddNestedItem(cvId, data, addAchievement, 'Achievement')} 
                                cvId={detailedCV.id}
                            />
                        </div>
                        
                        {/* --- RIGHT COLUMN: DATA LISTS (600px+) --- */}
                        <div style={{ flex: '2 1 600px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
                            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                                <div style={{ flex: '1 1 45%' }}>
                                    <NestedList 
                                        cvId={detailedCV.id}
                                        items={detailedCV.experiences}
                                        title="Experiences"
                                        listName="experiences"
                                        onRefresh={fetchAndSetDetails}
                                        onDelete={handleDeleteNested}
                                    />
                                </div>
                                <div style={{ flex: '1 1 45%' }}>
                                    <NestedList 
                                        cvId={detailedCV.id}
                                        items={detailedCV.education}
                                        title="Education"
                                        listName="education"
                                        onRefresh={fetchAndSetDetails}
                                        onDelete={handleDeleteNested}
                                    />
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                                <div style={{ flex: '1 1 45%' }}>
                                    <NestedList 
                                        cvId={detailedCV.id}
                                        items={detailedCV.projects}
                                        title="Projects"
                                        listName="projects"
                                        onRefresh={fetchAndSetDetails}
                                        onDelete={handleDeleteNested}
                                    />
                                </div>
                                <div style={{ flex: '1 1 45%' }}>
                                    <NestedList 
                                        cvId={detailedCV.id}
                                        items={detailedCV.skills}
                                        title="Master Skills"
                                        listName="skills"
                                        onRefresh={fetchAndSetDetails}
                                        onDelete={handleDeleteNested}
                                    />
                                </div>
                            </div>
                            
                            <div style={{ display: 'flex', gap: '20px', flexWrap: 'wrap' }}>
                                <div style={{ flex: '1 1 45%' }}>
                                     <NestedList 
                                        cvId={detailedCV.id}
                                        items={detailedCV.achievements}
                                        title="Master Achievements"
                                        listName="achievements"
                                        onRefresh={fetchAndSetDetails}
                                        onDelete={handleDeleteNested}
                                    />
                                </div>
                            </div>

                            <button onClick={() => handleDeleteCV(detailedCV.id)} style={{ backgroundColor: '#dc3545', color: 'white', padding: '10px' }}>
                                Delete Current CV
                            </button>
                        </div>
                    </div>
                ) : (
                    <p style={{ textAlign: 'center', color: '#777' }}>Select a CV above to begin editing its components.</p>
                )}
            </div>
        </div>
    );
};

export default CVManagerPage;