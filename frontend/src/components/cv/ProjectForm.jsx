import React, { useState } from 'react';
import SkillLinker from './SkillLinker';

const ProjectForm = ({ onSubmit, cvId, allSkills, onSkillCreate }) => {
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    const [selectedSkillIds, setSelectedSkillIds] = useState([]); // NEW STATE

    const handleSubmit = (e) => {
        e.preventDefault();
        if (title.trim() && description.trim()) {
            onSubmit(cvId, { 
                title, 
                description,
                skill_ids: selectedSkillIds // PASS THE LINKED SKILL IDs
            }, 'Project');
            
            setTitle('');
            setDescription('');
            setSelectedSkillIds([]); // Reset selection
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #ffc107', borderRadius: '5px', backgroundColor: '#fff8e1', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#ffc107' }}>+ Add New Project</h4>
            
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Project Title" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Project Description (Required)" required style={{ width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />
            
            <SkillLinker 
                cvId={cvId}
                allSkills={allSkills} 
                selectedSkillIds={selectedSkillIds} 
                setSelectedSkillIds={setSelectedSkillIds}
                onCreateNewSkill={onSkillCreate} 
            />
            
            <button type="submit" style={{ backgroundColor: '#ffc107', color: '#333', padding: '8px 15px', border: 'none', borderRadius: '4px', marginTop: '10px' }}>
                Create Project
            </button>
        </form>
    );
};

export default ProjectForm;