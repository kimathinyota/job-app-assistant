// frontend/src/components/cv/HobbyForm.jsx
import React, { useState } from 'react';
import SkillLinker from './SkillLinker';

const HobbyForm = ({ onSubmit, cvId, allSkills, onSkillCreate }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [selectedSkillIds, setSelectedSkillIds] = useState([]); 

    const handleSubmit = (e) => {
        e.preventDefault();
        if (name.trim()) {
            onSubmit(cvId, { 
                name, 
                description: description || 'No description provided.',
                skill_ids: selectedSkillIds 
            }, 'Hobby');
            
            setName('');
            setDescription('');
            setSelectedSkillIds([]); // Reset selection
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #6f42c1', borderRadius: '5px', backgroundColor: '#f5f0ff', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#6f42c1' }}>+ Add New Hobby</h4>
            
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Hobby Name (e.g., Hiking, Chess)" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description (Optional)" style={{ width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />
            
            <SkillLinker 
                cvId={cvId}
                allSkills={allSkills} 
                selectedSkillIds={selectedSkillIds} 
                setSelectedSkillIds={setSelectedSkillIds}
                onCreateNewSkill={onSkillCreate} 
            />
            
            <button type="submit" style={{ backgroundColor: '#6f42c1', color: 'white', padding: '8px 15px', border: 'none', borderRadius: '4px', marginTop: '10px' }}>
                Create Hobby
            </button>
        </form>
    );
};

export default HobbyForm;