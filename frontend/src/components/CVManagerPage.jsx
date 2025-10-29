// frontend/src/components/CVManagerPage.jsx
import React, { useState, useEffect } from 'react';
import {
    deleteBaseCV,
    fetchCVDetails,
    updateBaseCV, // <-- Was missing
    addExperience,
    updateExperience,
    addSkill,
    addAchievement,
    addEducation,
    // Add updateEducation if/when created
    addProject,
    // Add updateProject if/when created
    addHobby,
    // Add updateHobby if/when created
    deleteNestedItem
    // Import updateSkill, updateAchievement etc. when backend is ready
} from '../api/cvClient';

// --- Component Imports ---
import CVSelector from './cv/CVList';
import NestedList from './cv/NestedList';
import ExperienceForm from './cv/ExperienceForm';
import EducationForm from './cv/EducationForm';
import ProjectForm from './cv/ProjectForm';
import SkillForm from './cv/SkillForm';
import AchievementForm from './cv/AchievementForm';
import HobbyForm from './cv/HobbyForm';
// Assuming Achievement components exist from previous steps
import AchievementManagerModal from './cv/AchievementManagerModal';
import AchievementDisplayGrid from './cv/AchievementDisplayGrid';


// ... (SectionButton, CVSectionDashboard components remain the same) ...
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

const CVSectionDashboard = ({ cv, onSelectSection }) => (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'center', padding: '20px 0' }}>
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
    // 燥燥燥 THESE WERE MISSING 燥燥燥
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    // ^^^ THESE WERE MISSING ^^^

    const [activeSection, setActiveSection] = useState(null);
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({ name: '', summary: '' });
    const [editingItem, setEditingItem] = useState(null);

    // --- Data Fetching Logic ---
     const fetchAndSetDetails = async (id) => {
        setLoadingDetails(true); // Now declared
        try {
            const response = await fetchCVDetails(id);
            setDetailedCV(response); // Now declared
        } catch (error) {
            console.error("Failed to load CV details:", error);
            setDetailedCV(null); // Now declared
        } finally {
            setLoadingDetails(false); // Now declared
        }
    };

    // --- useEffect Hook ---
     useEffect(() => {
        setActiveSection(null);
        setIsEditingHeader(false);
        setEditingItem(null);

        // This block should now work correctly
        if (selectedCVId) {
            fetchAndSetDetails(selectedCVId);
        } else if (cvs.length > 0) {
            setSelectedCVId(cvs[0].id);
        } else {
             setDetailedCV(null);
        }
    }, [selectedCVId, cvs.length]); // selectedCVId is correctly declared


    // --- CRUD Handlers ---
    // (Rest of the handlers: handleStartEditHeader, handleCancelEdit, handleAddOrUpdateNestedItem, etc.)
    // (These should be okay, assuming imports are correct now)
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
            const updatedCV = await updateBaseCV(detailedCV.id, editFormData); // Needs updateBaseCV import
            setDetailedCV(updatedCV);
            setIsEditingHeader(false);
            await reloadData();
        } catch (error) {
            alert('Failed to update CV.');
            console.error(error);
        }
    };

    const handleStartEditItem = (item, sectionName) => {
        setActiveSection(sectionName);
        setEditingItem(item);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const handleCancelEdit = () => {
        setEditingItem(null);
    };

    const handleAddOrUpdateNestedItem = async (cvId, data, itemType) => {
        const isUpdating = Boolean(data.id);
        const itemId = data.id;
        let finalSkillIds = data.existing_skill_ids || [];
        let finalAchievementIds = data.existing_achievement_ids || [];

        // Placeholder for missing update functions - replace comments when ready
        const apiFunctions = {
            'Experience': { add: addExperience, update: updateExperience },
            'Education': { add: addEducation, update: /* updateEducation */ addEducation },
            'Project': { add: addProject, update: /* updateProject */ addProject },
            'Hobby': { add: addHobby, update: /* updateHobby */ addHobby },
            'Achievement': { add: addAchievement, update: /* updateAchievement */ addAchievement },
            'Skill': { add: addSkill, update: /* updateSkill */ addSkill },
        };

        const { add: addFn, update: updateFn } = apiFunctions[itemType] || {}; // Default to empty object if not found
        if (!addFn || !updateFn) { // Check if functions exist
            alert(`API functions not fully configured for ${itemType}`);
            return;
        }


        try {
            // Step 1: Create pending skills/achievements
            if (data.new_skills && data.new_skills.length > 0) {
                 console.log("Creating new skills:", data.new_skills);
                 const creationPromises = data.new_skills.map(skillData => addSkill(cvId, skillData));
                 const results = await Promise.all(creationPromises);
                 const newSkillIds = results.map(response => response.data.id);
                 finalSkillIds = [...finalSkillIds, ...newSkillIds];
                 const newSkills = results.map(response => response.data);
                 setDetailedCV(prevCV => ({...prevCV, skills: [...(prevCV.skills || []), ...newSkills]})); // Ensure prevCV.skills exists
                 await reloadData();
            }
             // Add similar logic for data.new_achievements here if implementing

            // Step 2: Prepare payload
            const itemDataPayload = { ...data };
            delete itemDataPayload.id;
            delete itemDataPayload.existing_skill_ids;
            delete itemDataPayload.new_skills;
            delete itemDataPayload.existing_achievement_ids;
            delete itemDataPayload.new_achievements;
            itemDataPayload.skill_ids = finalSkillIds;
            itemDataPayload.achievement_ids = finalAchievementIds;

            // Step 3: Call API
            if (isUpdating) {
                console.log(`Updating ${itemType} (${itemId}) with data:`, itemDataPayload);
                await updateFn(cvId, itemId, itemDataPayload);
                alert(`${itemType} updated successfully!`);
                setEditingItem(null);
            } else {
                console.log(`Creating ${itemType} with data:`, itemDataPayload);
                await addFn(cvId, itemDataPayload);
                alert(`${itemType} added successfully!`);
            }

            // Step 4: Refresh
            fetchAndSetDetails(cvId);

        } catch (error) {
            alert(`Failed to ${isUpdating ? 'update' : 'add'} ${itemType}. Check console.`);
            console.error(error);
        }
    };


    const handleDeleteCV = async (cvIdToDelete) => { // Use different variable name
        if (window.confirm("Delete this master CV?")) {
            try {
                await deleteBaseCV(cvIdToDelete); // Use the function arg
                alert("CV deleted!");
                await reloadData();
                setSelectedCVId(null);
                setDetailedCV(null);
            } catch (error) {
                alert("Failed to delete CV."); console.error(error);
            }
        }
    };

     const handleDeleteNested = async (cvIdToDeleteFrom, itemId, listName) => { // Use different variable name
        if (window.confirm(`Delete this item from ${listName}?`)) {
            try {
                await deleteNestedItem(cvIdToDeleteFrom, itemId, listName); // Use the function arg
                fetchAndSetDetails(cvIdToDeleteFrom); // Use the function arg
            } catch (error) {
                alert(`Error deleting item.`); console.error(error);
            }
        }
    };

    // --- RENDER LOGIC ---

    // Safely access nested properties
    const masterSkills = detailedCV?.skills || [];
    const masterAchievements = detailedCV?.achievements || [];

    if (cvs.length === 0) {
        return (
             <div style={{ textAlign: 'center', padding: '50px', border: '2px dashed #007bff', borderRadius: '8px' }}>
                <h3 style={{ color: '#007bff' }}>No CVs Found</h3>
                <p>Please use the form on the Dashboard Home to create your first base CV.</p>
                <button onClick={() => setActiveView('Dashboard')}>Go to Dashboard</button>
            </div>
        );
    }

    // --- Render Section Detail Helper ---
    const renderSectionDetail = () => {
        // ... (renderSectionDetail logic remains the same as previous correct version) ...
         if (!activeSection) return null;

        // Map section names to components and data
        const sectionMap = {
            'Experiences': { Form: ExperienceForm, items: detailedCV?.experiences || [], listName: 'experiences', formProps: { allSkills: masterSkills, allAchievements: masterAchievements } },
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

        return (
            <div>
                 <button onClick={() => { setActiveSection(null); handleCancelEdit(); }} style={{ marginBottom: '20px', backgroundColor: '#6c757d', color: 'white' }}>
                    &larr; Back to CV Dashboard
                </button>

                <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap', marginTop: '10px' }}>
                    <div style={{ flex: '1 1 350px', minWidth: '350px' }}>
                        <h3 style={{ borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
                            {currentEditingItem ? `Edit ${listName.slice(0, -1)}` : `Add New ${listName.slice(0, -1)}`}
                        </h3>
                        <Form
                            key={currentEditingItem?.id || 'new-item-form'}
                            onSubmit={handleAddOrUpdateNestedItem}
                            cvId={detailedCV.id} // detailedCV should exist if we are here
                            initialData={currentEditingItem}
                            onCancelEdit={handleCancelEdit}
                            {...formProps}
                        />
                    </div>
                    <div style={{ flex: '2 1 600px' }}>
                        {!currentEditingItem && <h3>Existing {listName}</h3>}
                        
                        {/* 💡💡💡 THIS IS THE FIX 💡💡💡 */}
                        <NestedList
                            cvId={detailedCV.id}
                            items={items}
                            listName={listName}
                            onDelete={handleDeleteNested}
                            onEdit={(item) => handleStartEditItem(item, activeSection)}
                            allSkills={masterSkills} // 👈 **FIX: Pass master skills list**
                            allAchievements={masterAchievements} // 👈 **FIX: Pass master achievements list**
                        />
                    </div>
                </div>
            </div>
        );
    };


    // --- Final Return Block ---
    return (
        <div style={{ textAlign: 'left' }}>
            <h2>CV Manager & Editor</h2>
            <CVSelector cvs={cvs} onSelect={setSelectedCVId} selectedCVId={selectedCVId} />
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px', backgroundColor: '#fff' }}>
                {loadingDetails ? (
                    <p style={{ textAlign: 'center', color: '#007bff' }}>Loading CV details...</p>
                ) : detailedCV ? ( // Check detailedCV before accessing its properties
                    <>
                        {/* CV Header */}
                        {!isEditingHeader ? (
                             <div style={{ marginBottom: '20px', paddingBottom: '20px', borderBottom: '2px solid #007bff', position: 'relative' }}>
                                 <h3 style={{ margin: 0, color: '#007bff' }}>{detailedCV.name}</h3>
                                 <p style={{ margin: '5px 0 0 0', fontStyle: 'italic', color: '#555', whiteSpace: 'pre-wrap' }}>
                                     {detailedCV.summary || "No summary."}
                                 </p>
                                 <button onClick={handleStartEditHeader} style={{ position: 'absolute', top: '10px', right: '10px', fontSize: '0.8em' }}>
                                     Edit Details
                                 </button>
                            </div>
                        ) : (
                            <form onSubmit={handleUpdateCVHeader} style={{ marginBottom: '20px', padding: '15px', border: '2px dashed #007bff', borderRadius: '5px' }}>
                                <div><label>Name:</label><input type="text" value={editFormData.name} onChange={(e) => setEditFormData({...editFormData, name: e.target.value})} required style={{width: '95%'}} /></div>
                                <div><label>Summary:</label><textarea value={editFormData.summary} onChange={(e) => setEditFormData({...editFormData, summary: e.target.value})} style={{width: '95%', minHeight: '80px'}}/></div>
                                <button type="submit">Save</button> <button type="button" onClick={handleCancelEditHeader}>Cancel</button>
                            </form>
                        )}

                        {/* CV Body */}
                        {activeSection === null ? (
                            <CVSectionDashboard cv={detailedCV} onSelectSection={setActiveSection} />
                        ) : (
                            renderSectionDetail()
                        )}

                        {/* Footer */}
                         <div style={{ borderTop: '2px solid #eee', marginTop: '30px', paddingTop: '20px', textAlign: 'right' }}>
                            <button onClick={() => handleDeleteCV(detailedCV.id)} style={{ backgroundColor: '#dc3545', color: 'white' }}>
                                Delete This Entire CV
                            </button>
                        </div>
                    </>
                ) : (
                     // Handle case where detailedCV is still null (e.g., initial load or error)
                    selectedCVId ? <p>Loading CV details...</p> : <p style={{ textAlign: 'center', color: '#777' }}>Select a CV to manage.</p>
                )}
            </div>
        </div>
    );
};

export default CVManagerPage;