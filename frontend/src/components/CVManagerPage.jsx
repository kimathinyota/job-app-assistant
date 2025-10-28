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
    addHobby, 
    deleteNestedItem
} from '../api/cvClient'; 

import CVSelector from './cv/CVList'; 
import NestedList from './cv/NestedList'; 
import ExperienceForm from './cv/ExperienceForm';
import EducationForm from './cv/EducationForm'; 
import ProjectForm from './cv/ProjectForm';     
import SkillForm from './cv/SkillForm'; 
import AchievementForm from './cv/AchievementForm';
import HobbyForm from './cv/HobbyForm'; 

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
        <SectionButton title="Experiences" count={cv.experiences.length} onClick={() => onSelectSection('Experiences')} />
        <SectionButton title="Education" count={cv.education.length} onClick={() => onSelectSection('Education')} />
        <SectionButton title="Projects" count={cv.projects.length} onClick={() => onSelectSection('Projects')} />
        <SectionButton title="Master Skills" count={cv.skills.length} onClick={() => onSelectSection('Skills')} />
        <SectionButton title="Master Achievements" count={cv.achievements.length} onClick={() => onSelectSection('Achievements')} />
        <SectionButton title="Hobbies" count={cv.hobbies.length} onClick={() => onSelectSection('Hobbies')} />
    </div>
);

const CVManagerPage = ({ cvs, setActiveView, reloadData }) => {
    const [selectedCVId, setSelectedCVId] = useState(cvs[0]?.id || null);
    const [detailedCV, setDetailedCV] = useState(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [activeSection, setActiveSection] = useState(null);
    const [isEditingHeader, setIsEditingHeader] = useState(false);
    const [editFormData, setEditFormData] = useState({ name: '', summary: '' });
    const [editingExperience, setEditingExperience] = useState(null);

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
        setActiveSection(null);
        setIsEditingHeader(false);
        if (selectedCVId) fetchAndSetDetails(selectedCVId);
    }, [selectedCVId, cvs.length]);

    const handleStartEditHeader = () => {
        setEditFormData({ name: detailedCV.name, summary: detailedCV.summary || '' });
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

    const handleCreateSkillAndLink = async (cvId, skillData) => {
        try {
            const response = await addSkill(cvId, skillData);
            const newSkill = response.data;
            alert(`New Skill "${newSkill.name}" created!`);
            setDetailedCV(prev => ({ ...prev, skills: [...prev.skills, newSkill] }));
            return newSkill; 
        } catch (error) {
            alert('Failed to create skill.');
            console.error(error);
            return null;
        }
    };

    const handleAddNestedItem = async (cvId, data, addFunction, itemType) => {
        try {
            await addFunction(cvId, data);
            alert(`${itemType} added successfully!`);
            fetchAndSetDetails(cvId);
        } catch (error) {
            alert(`Failed to add ${itemType}.`);
            console.error(error);
        }
    };

    const handleDeleteNested = async (cvId, itemId, listName) => {
        if (window.confirm(`Delete this item from ${listName}?`)) {
            try {
                await deleteNestedItem(cvId, itemId, listName);
                fetchAndSetDetails(cvId);
            } catch (error) {
                alert(`Error deleting item.`);
                console.error(error);
            }
        }
    };

    const handleEditExperience = (experience) => {
        setActiveSection('Experiences');
        setEditingExperience(experience);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    const handleUpdateExperience = async (cvId, expId, data) => {
        try {
            await updateBaseCV(cvId, {
                experiences: detailedCV.experiences.map(e => e.id === expId ? { ...e, ...data } : e)
            });
            alert("Experience updated successfully!");
            setEditingExperience(null);
            fetchAndSetDetails(cvId);
        } catch (error) {
            alert("Failed to update experience.");
            console.error(error);
        }
    };

    const masterSkills = detailedCV ? detailedCV.skills : [];

    const renderSectionDetail = () => {
        if (!activeSection) return null;
        const sectionMap = {
            'Experiences': { Form: ExperienceForm, items: detailedCV.experiences, listName: 'experiences', addFn: addExperience },
            'Education': { Form: EducationForm, items: detailedCV.education, listName: 'education', addFn: addEducation },
            'Projects': { Form: ProjectForm, items: detailedCV.projects, listName: 'projects', addFn: addProject },
            'Skills': { Form: SkillForm, items: detailedCV.skills, listName: 'skills', addFn: addSkill },
            'Achievements': { Form: AchievementForm, items: detailedCV.achievements, listName: 'achievements', addFn: addAchievement },
            'Hobbies': { Form: HobbyForm, items: detailedCV.hobbies, listName: 'hobbies', addFn: addHobby },
        };
        const current = sectionMap[activeSection];
        if (!current) return <p>Section not found.</p>;
        const { Form, items, listName, addFn } = current;

        return (
            <div>
                <button onClick={() => { setActiveSection(null); setEditingExperience(null); }} style={{ marginBottom: '20px', backgroundColor: '#6c757d', color: 'white' }}>
                    &larr; Back
                </button>
                <div style={{ display: 'flex', gap: '30px', flexWrap: 'wrap', marginTop: '10px' }}>
                    <div style={{ flex: '1 1 350px', minWidth: '350px' }}>
                        <h3 style={{ borderBottom: '1px solid #eee', paddingBottom: '10px' }}>
                            {editingExperience ? "Edit Experience" : `Add New ${listName.slice(0, -1)}`}
                        </h3>
                        <Form
                            cvId={detailedCV.id}
                            onSubmit={(cvId, data) =>
                                editingExperience
                                    ? handleUpdateExperience(cvId, editingExperience.id, data)
                                    : handleAddNestedItem(cvId, data, addFn, activeSection.slice(0, -1))
                            }
                            allSkills={masterSkills}
                            onSkillCreate={handleCreateSkillAndLink}
                            existingData={editingExperience}
                            onCancelEdit={() => setEditingExperience(null)}
                        />
                    </div>
                    <div style={{ flex: '2 1 600px' }}>
                        <NestedList
                            cvId={detailedCV.id}
                            items={items}
                            title={`Existing ${listName}`}
                            listName={listName}
                            onRefresh={fetchAndSetDetails}
                            onDelete={handleDeleteNested}
                            onEdit={listName === 'experiences' ? handleEditExperience : undefined}
                        />
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div style={{ textAlign: 'left' }}>
            <h2>CV Manager & Editor</h2>
            <CVSelector cvs={cvs} onSelect={setSelectedCVId} selectedCVId={selectedCVId} />
            <div style={{ marginTop: '20px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px', backgroundColor: '#fff' }}>
                {loadingDetails ? (
                    <p>Loading...</p>
                ) : detailedCV ? (
                    <>
                        {!isEditingHeader ? (
                            <div style={{ marginBottom: '20px', borderBottom: '2px solid #007bff', position: 'relative' }}>
                                <h3 style={{ margin: 0, color: '#007bff' }}>{detailedCV.name}</h3>
                                <p style={{ margin: '5px 0 0 0', fontStyle: 'italic', color: '#555', whiteSpace: 'pre-wrap' }}>
                                    {detailedCV.summary || "No summary provided."}
                                </p>
                                <button onClick={handleStartEditHeader} style={{ position: 'absolute', top: '10px', right: '10px' }}>
                                    Edit Details
                                </button>
                            </div>
                        ) : (
                            <form onSubmit={handleUpdateCVHeader}>
                                <label>CV Name:</label>
                                <input
                                    type="text"
                                    value={editFormData.name}
                                    onChange={(e) => setEditFormData({ ...editFormData, name: e.target.value })}
                                    required
                                />
                                <label>Summary:</label>
                                <textarea
                                    value={editFormData.summary}
                                    onChange={(e) => setEditFormData({ ...editFormData, summary: e.target.value })}
                                />
                                <button type="submit">Save</button>
                                <button type="button" onClick={handleCancelEditHeader}>Cancel</button>
                            </form>
                        )}
                        {activeSection === null ? (
                            <CVSectionDashboard cv={detailedCV} onSelectSection={setActiveSection} />
                        ) : (
                            renderSectionDetail()
                        )}
                    </>
                ) : (
                    <p>Select a CV above to begin editing.</p>
                )}
            </div>
        </div>
    );
};

export default CVManagerPage;
