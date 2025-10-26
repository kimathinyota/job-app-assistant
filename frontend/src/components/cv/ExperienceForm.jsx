import React, { useState } from 'react';
import SkillLinker from './SkillLinker';

const ExperienceForm = ({ onSubmit, cvId, allSkills, onSkillCreate }) => {
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [description, setDescription] = useState('');
    const [selectedSkillIds, setSelectedSkillIds] = useState([]); 

    const handleSubmit = (e) => {
        e.preventDefault();
        if (title.trim() && company.trim()) {
            onSubmit(cvId, { 
                title, 
                company, 
                description: description || 'No description provided.',
                skill_ids: selectedSkillIds // INCLUDES SKILL IDS
            }, 'Experience');
            
            setTitle('');
            setCompany('');
            setDescription('');
            setSelectedSkillIds([]); // Reset selection
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #007bff', borderRadius: '5px', backgroundColor: '#e6f7ff', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#007bff' }}>+ Add New Experience</h4>
            
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Job Title" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <input type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Company" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description (Optional)" style={{ width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />
            
            <SkillLinker 
                cvId={cvId}
                allSkills={allSkills} 
                selectedSkillIds={selectedSkillIds} 
                setSelectedSkillIds={setSelectedSkillIds}
                onCreateNewSkill={onSkillCreate} 
            />
            
            <button type="submit" style={{ backgroundColor: '#007bff', color: 'white', padding: '8px 15px', border: 'none', borderRadius: '4px', marginTop: '10px' }}>
                Create Experience
            </button>
        </form>
    );
};

export default ExperienceForm;