import React from 'react';
import { useOutletContext } from 'react-router-dom';
import { 
    addExperienceComplex, updateExperienceComplex,
    addEducationComplex, updateEducationComplex,
    addHobbyComplex, updateHobbyComplex,
    addProjectComplex, updateProjectComplex,
    addSkill, deleteNestedItem 
} from '../../api/cvClient';

const CVSectionWrapper = ({ component: Component, section }) => {
    const { cv, refreshCV } = useOutletContext();

    // Map section names to API functions
    const apiStrategies = {
        experiences: { add: addExperienceComplex, update: updateExperienceComplex },
        education:   { add: addEducationComplex, update: updateEducationComplex },
        projects:    { add: addProjectComplex, update: updateProjectComplex },
        hobbies:     { add: addHobbyComplex, update: updateHobbyComplex },
        skills:      { add: addSkill, update: addSkill }, // Simple updates usually handled differently, but mapping for consistency
        achievements:{ add: null, update: null } // AchievementHub handles its own API calls internally usually, or we pass a generic
    };

    const handleSave = async (cvId, data, itemType) => {
        // itemType string from the form (e.g. 'Experience', 'Project')
        // We map it to the API strategy
        // Note: The managers often pass 'Experience' as itemType string. 
        // We need to map that to our strategies object keys.
        
        const typeKeyMap = {
            'Experience': 'experiences',
            'Education': 'education',
            'Project': 'projects',
            'Hobby': 'hobbies',
            'Skill': 'skills'
        };

        const key = typeKeyMap[itemType] || section;
        const strategies = apiStrategies[key];

        if (!strategies) {
            console.error("No strategy found for", itemType);
            return;
        }

        try {
            if (data.id) {
                // Update
                if (key === 'skills') await strategies.update(cvId, data); 
                else await strategies.update(cvId, data.id, data);
            } else {
                // Create
                await strategies.add(cvId, data);
            }
            alert(`${itemType} saved successfully!`);
            await refreshCV();
        } catch (error) {
            console.error(error);
            alert(`Failed to save ${itemType}`);
        }
    };

    const handleDelete = async (cvId, itemId, listName) => {
        if (window.confirm("Are you sure you want to delete this item?")) {
            try {
                await deleteNestedItem(cvId, itemId, listName);
                await refreshCV();
            } catch (error) {
                alert("Failed to delete item.");
            }
        }
    };

    // Props required by most Managers
    const commonProps = {
        cvId: cv.id,
        allSkills: cv.skills || [],
        allAchievements: cv.achievements || [],
        // Specific Lists
        experiences: cv.experiences || [],
        education: cv.education || [],
        projects: cv.projects || [],
        hobbies: cv.hobbies || [],
        // Context Lists (for linkers)
        allExperiences: cv.experiences || [],
        allEducation: cv.education || [],
        allProjects: cv.projects || [],
        allHobbies: cv.hobbies || [],
        
        // Actions
        onSubmit: handleSave,
        onDelete: handleDelete,
        // Helper to force a refresh if component does internal logic
        onRefresh: refreshCV 
    };

    return (
        <div className="bg-white rounded-xl border shadow-sm p-4">
            <Component {...commonProps} />
        </div>
    );
};

export default CVSectionWrapper;