import React, { useState, useEffect } from 'react';
import { 
    deleteBaseCV, 
    fetchCVDetails, 
    updateBaseCV,
    addExperience, 
    addSkill,
    addAchievement,
    addEducation, 
    addProject,   
    addHobby, // <-- Includes Hobbies
    deleteNestedItem
} from '../api/cvClient'; 

// --- Component Imports ---
import CVSelector from './cv/CVList'; 
import NestedList from './cv/NestedList'; 
import ExperienceForm from './cv/ExperienceForm';
import EducationForm from './cv/EducationForm'; 
import ProjectForm from './cv/ProjectForm';     
import SkillForm from './cv/SkillForm'; 
import AchievementForm from './cv/AchievementForm';
import HobbyForm from './cv/HobbyForm'; // <-- Includes Hobbies

// --- STYLED HELPER COMPONENTS (for the new dashboard) ---

const SectionButton = ({ title, count, onClick }) => (
    <button 
        onClick={onClick} 
        style={{ 
            flex: '1 1 250px', 
            minHeight: '100px',
            textAlign: 'left',
            padding: '15px',
            backgroundColor: '#fff',
            border: '1px solid #ccc',
            borderRadius: '8px',
            cursor: 'pointer',
            transition: 'all 0.2s',
            boxShadow: '0 2px 4px rgba(0,0,0,0.05)',
        }}
        onMouseOver={e => e.currentTarget.style.borderColor = '#007bff'}
        onMouseOut={e => e.currentTarget.style.borderColor = '#ccc'}
    >
        <span style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#007bff', display: 'block' }}>{title}</span>
        <span style={{ fontSize: '0.9em', color: '#555' }}>({count} items)</span>
    </button>
);

// --- CV SECTION DASHBOARD (LEVEL 1) ---
const CVSectionDashboard = ({ cv, onSelectSection }) => (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'center', padding: '20px 0' }}>
        <SectionButton title="Experiences" count={cv.experiences.length} onClick={() => onSelectSection('Experiences')} />
        <SectionButton title="Education" count={cv.education.length} onClick={() => onSelectSection('Education')} />
        <SectionButton title="Projects" count={cv.projects.length} onClick={() => onSelectSection('Projects')} />
        <SectionButton title="Master Skills" count={cv.skills.length} onClick={() => onSelectSection('Skills')} />
        <SectionButton title="Master Achievements" count={cv.achievements.length} onClick={() => onSelectSection('Achievements')} />
        <SectionButton title="Hobbies" count={cv.hobbies.length} onClick={() => onSelectSection('Hobbies')} /> {/* <-- Includes Hobbies */}
    </div>
);


// --- Main CV Manager Component ---
const CVManagerPage = ({ cvs, setActiveView, reloadData }) => {
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);

    // --- NEW STATE for layout and editing ---
    const [activeSection, setActiveSection] = useState(null); // null | 'Experiences' | 'Education' etc.
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({ name: '', summary: '' });

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
        // When selected CV changes, reset the view
        setActiveSection(null);
        setIsEditingHeader(false);

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

    const handleCancelEditHeader = () => {
        setIsEditingHeader(false);
    };

    const handleUpdateCVHeader = async (e) => {
        e.preventDefault();
        if (!editFormData.name.trim()) {
            alert('CV Name is required.');
            return;
        }
        try {
            const updatedCV = await updateBaseCV(detailedCV.id, editFormData);
            setDetailedCV(updatedCV); // Update local state with returned data
            setIsEditingHeader(false);
            await reloadData(); // Reloads master CV list in App.jsx to show new name in selector
        } catch (error) {
            alert('Failed to update CV details. Check console.');
            console.error(error);
        }
    };

    /**
     * --- *** THIS IS THE CORRECT, FIXED VERSION *** ---
     * This function now creates the skill, updates the local state,
     * and returns the new skill object so the SkillLinker can auto-select it.
     */
    const handleCreateSkillAndLink = async (cvId, skillData) => {
        try {
            // 1. Create the new skill in the master list
            const response = await addSkill(cvId, skillData);
            const newSkill = response.data; // Get the new skill from the API response
            alert(`New Skill "${newSkill.name}" created successfully!`);
            
            // 2. Update local state instead of full reload
            setDetailedCV(prevCV => ({
                ...prevCV,
                skills: [...prevCV.skills, newSkill] // Add new skill to master list
            }));

            // 3. Return the new skill object
            return newSkill; 
            
        } catch (error) {
            alert('Failed to create new skill. Check console.');
            console.error(error);
            return null; // Return null on failure
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

    /**
     * --- *** THIS IS THE CORRECT, FIXED VERSION *** ---
     * It now re-fetches details *after* adding an item, which is
     * necessary to show the newly created item (and its skill links).
     */
    const handleAddNestedItem = async (cvId, data, addFunction, itemType) => {
        try {
            await addFunction(cvId, data);
            alert(`${itemType} added successfully!`);
            
            // If the added item is a top-level resource (like Achievement/Skill), refresh everything.
            if (itemType === 'Achievement' || itemType === 'Skill') {
                 await reloadData(); // Reloads master list
                 fetchAndSetDetails(cvId); // Reloads details
            } else {
                 // For Experience, Project, Education, Hobby, just reload the details
                 // to show the new item in its list.
                 fetchAndSetDetails(cvId); 
            }
        } catch (error) {
            alert(`Failed to add ${itemType}. Check console.`);
            console.error(error);
        }
    };

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

    // --- Helper function to render the active section (LEVEL 2) ---
    const renderSectionDetail = () => {
        if (!activeSection) return null;

        // Find the correct data for the active section
        const sectionMap = {
            'Experiences': { 
                Form: ExperienceForm, 
                items: detailedCV.experiences, 
                listName: 'experiences',
                addFn: addExperience
            },
            'Education': { 
                Form: EducationForm, 
                items: detailedCV.education, 
                listName: 'education',
                addFn: addEducation
            },
            'Projects': { 
                Form: ProjectForm, 
                items: detailedCV.projects, 
                listName: 'projects',
                addFn: addProject
            },
            'Skills': { 
                Form: SkillForm, 
                items: detailedCV.skills, 
                listName: 'skills',
                addFn: addSkill
            },
            'Achievements': { 
                Form: AchievementForm, 
                items: detailedCV.achievements, 
                listName: 'achievements',
                addFn: addAchievement
            },
            // <-- Includes Hobbies -->
            'Hobbies': {
                Form: HobbyForm,
                items: detailedCV.hobbies,
                listName: 'hobbies',
                addFn: addHobby
            },
        };

        const current = sectionMap[activeSection];
        if (!current) return <p>Section not found.</p>;

        const { Form, items, listName, addFn } = current;

        return (
            <div>
                <button onClick={() => setActiveSection(null)} style={{ marginBottom: '20px', backgroundColor: '#6c757d', color: 'white' }}>
                    &larr; Back to CV Dashboard
                </button>
                
                <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap', marginTop: '10px' }}>
                    
                    {/* Left Column: The Form */}
                    <div style={{ flex: '1 1 350px', minWidth: '350px' }}>
                        <h3 style={{ borderBottom: '1px solid #eee', paddingBottom: '10px' }}>Add New {listName.slice(0, -1)}</h3>
                        <Form
                            onSubmit={(cvId, data) => handleAddNestedItem(cvId, data, addFn, activeSection.slice(0, -1))} 
                            cvId={detailedCV.id}
                            allSkills={masterSkills}
                            onSkillCreate={handleCreateSkillAndLink} 
                        />
                    </div>

                    {/* Right Column: The List */}
                    <div style={{ flex: '2 1 600px' }}>
                        <NestedList 
                            cvId={detailedCV.id}
                            items={items}
                            title={`Existing ${listName}`}
                            listName={listName}
                            onRefresh={fetchAndSetDetails}
                            onDelete={handleDeleteNested}
                        />
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div style={{ textAlign: 'left' }}>
            <h2>CV Manager & Editor</h2>
            
            <CVSelector 
                cvs={cvs}
                onSelect={setSelectedCVId}
                selectedCVId={selectedCVId}
            />

            {/* Detail/Editor Area */}
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px', backgroundColor: '#fff' }}>
                {loadingDetails ? (
                    <p style={{ textAlign: 'center' }}>Loading detailed components...</p>
                ) : detailedCV ? (
                    <>
                        {/* --- 1. CV HEADER (NOW EDITABLE) --- */}
                        {!isEditingHeader ? (
                            <div style={{ marginBottom: '20px', paddingBottom: '20px', borderBottom: '2px solid #007bff', position: 'relative' }}>
                                <h3 style={{ margin: 0, color: '#007bff' }}>{detailedCV.name}</h3>
                                <p style={{ margin: '5px 0 0 0', fontStyle: 'italic', color: '#555', whiteSpace: 'pre-wrap' }}>
                                    {detailedCV.summary || "No summary provided."}
                                </p>
                                <button onClick={handleStartEditHeader} style={{ position: 'absolute', top: '10px', right: '10px', fontSize: '0.8em' }}>
                                    Edit Details
                                </button>
                            </div>
                        ) : (
                            <form onSubmit={handleUpdateCVHeader} style={{ marginBottom: '20px', padding: '15px', border: '2px dashed #007bff', borderRadius: '5px' }}>
                                <div>
                                    <label style={{ fontWeight: 'bold' }}>CV Name:</label>
                                    <input 
                                        type="text" 
                                        value={editFormData.name}
                                        onChange={(e) => setEditFormData({...editFormData, name: e.target.value})}
                                        required
                                        style={{ width: '95%', padding: '8px', marginBottom: '10px' }}
                                    />
                                </div>
                                <div>
                                    <label style={{ fontWeight: 'bold' }}>Summary:</label>
                                    <textarea
                                        value={editFormData.summary}
                                        onChange={(e) => setEditFormData({...editFormData, summary: e.target.value})}
                                        style={{ width: '95%', minHeight: '80px', padding: '8px', marginBottom: '10px' }}
                                    />
                                </div>
                                <button type="submit" style={{ backgroundColor: '#28a745', color: 'white', marginRight: '10px' }}>
                                    Save Changes
                                </button>
                                <button type="button" onClick={handleCancelEditHeader} style={{ backgroundColor: '#6c757d', color: 'white' }}>
                                    Cancel
                                </button>
                            </form>
                        )}

                        {/* --- 2. CV BODY (DASHBOARD OR DETAIL) --- */}
                        {activeSection === null ? (
                            <CVSectionDashboard 
                                cv={detailedCV} 
                                onSelectSection={setActiveSection} 
                            />
                        ) : (
                            renderSectionDetail()
                        )}

                        {/* --- 3. FOOTER (DELETE BUTTON) --- */}
                        <div style={{ borderTop: '2px solid #eee', marginTop: '30px', paddingTop: '20px', textAlign: 'right' }}>
                            <button onClick={() => handleDeleteCV(detailedCV.id)} style={{ backgroundColor: '#dc3545', color: 'white', padding: '10px' }}>
                                Delete This Entire CV
                            </button>
                        </div>
                    </>
                ) : (
                    <p style={{ textAlign: 'center', color: '#777' }}>Select a CV above to begin editing its components.</p>
                )}
            </div>
        </div>
    );
};

export default CVManagerPage;