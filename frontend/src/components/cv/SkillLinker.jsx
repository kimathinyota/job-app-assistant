import React, { useState } from 'react';

const SkillLinker = ({ allSkills, selectedSkillIds, setSelectedSkillIds, onCreateNewSkill, cvId }) => {
    
    // Local state for the new skill creation field
    const [newSkillName, setNewSkillName] = useState('');
    const [newSkillCategory, setNewSkillCategory] = useState('technical');
    
    // Handler for existing skills: Toggles selection status
    const handleToggleSkill = (skillId) => {
        if (selectedSkillIds.includes(skillId)) {
            setSelectedSkillIds(selectedSkillIds.filter(id => id !== skillId));
        } else {
            setSelectedSkillIds([...selectedSkillIds, skillId]);
        }
    };
    
    // Handler for creating and linking a new skill
    const handleCreateAndLink = (e) => {
        e.preventDefault();
        if (newSkillName.trim()) {
            // Call the parent handler (CVManagerPage) to create the skill
            onCreateNewSkill(cvId, { 
                name: newSkillName, 
                category: newSkillCategory 
            });
            
            // Clear the form fields (the selector will update upon data refresh)
            setNewSkillName('');
            setNewSkillCategory('technical');
        }
    };

    return (
        <div style={{ marginTop: '15px', padding: '15px', border: '2px solid #28a745', borderRadius: '8px', backgroundColor: '#f0fff0' }}>
            <strong style={{ display: 'block', marginBottom: '10px', fontSize: '1.1em', color: '#28a745' }}>Skill Management</strong>
            
            {/* 1. SECTION FOR CREATING NEW SKILL */}
            <form onSubmit={handleCreateAndLink} style={{ paddingBottom: '10px', borderBottom: '1px dotted #ccc', marginBottom: '10px' }}>
                <p style={{ margin: '0 0 5px 0', fontWeight: 'bold' }}>Create & Link New Skill:</p>
                <input
                    type="text"
                    value={newSkillName}
                    onChange={(e) => setNewSkillName(e.target.value)}
                    placeholder="e.g., Python, Team Leadership"
                    style={{ width: '60%', padding: '8px', marginRight: '5px' }}
                />
                <select value={newSkillCategory} onChange={(e) => setNewSkillCategory(e.target.value)} style={{ width: '20%', padding: '8px', marginRight: '5px' }}>
                    <option value="technical">Tech</option>
                    <option value="soft">Soft</option>
                    <option value="language">Lang</option>
                    <option value="other">Other</option>
                </select>
                <button type="submit" disabled={!newSkillName.trim()} style={{ padding: '8px 10px', backgroundColor: '#28a745', color: 'white' }}>
                    Create
                </button>
            </form>

            {/* 2. SECTION FOR LINKING EXISTING SKILLS */}
            <strong style={{ display: 'block', marginBottom: '8px', marginTop: '15px' }}>Link Existing Skills:</strong>
            <p style={{ fontSize: '0.8em', color: '#666', marginTop: 0 }}>
                {allSkills.length} master skills available. Select skills used here.
            </p>
            
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', maxHeight: '150px', overflowY: 'auto', border: '1px solid #eee', padding: '5px' }}>
                {allSkills.map(skill => (
                    <label key={skill.id} style={{ display: 'flex', alignItems: 'center', fontSize: '0.9em', padding: '4px 8px', border: '1px solid #ddd', borderRadius: '20px', cursor: 'pointer', backgroundColor: selectedSkillIds.includes(skill.id) ? '#d4edda' : 'white', transition: 'background-color 0.2s' }}>
                        <input
                            type="checkbox"
                            checked={selectedSkillIds.includes(skill.id)}
                            onChange={() => handleToggleSkill(skill.id)}
                            style={{ marginRight: '5px' }}
                        />
                        {skill.name}
                    </label>
                ))}
            </div>
        </div>
    );
};

export default SkillLinker;