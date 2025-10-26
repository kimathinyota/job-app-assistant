import React, { useState } from 'react';

const SkillForm = ({ onSubmit, cvId }) => {
    const [name, setName] = useState('');
    const [category, setCategory] = useState('technical');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (name.trim()) {
            // Note: This form's onSubmit handler is passed the addSkill client function in CVManagerPage
            onSubmit(cvId, { name, category }, 'Skill');
            setName('');
            setCategory('technical');
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #28a745', borderRadius: '5px', backgroundColor: '#e8f9e8', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#28a745' }}>+ Add Master Skill (Standalone)</h4>
            
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="Skill Name (e.g., Python, Leadership)" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            
            <select value={category} onChange={(e) => setCategory(e.target.value)} style={{ width: '98%', padding: '8px', marginBottom: '10px' }}>
                <option value="technical">Technical</option>
                <option value="soft">Soft Skill</option>
                <option value="language">Language</option>
                <option value="other">Other</option>
            </select>
            
            <button type="submit" style={{ backgroundColor: '#28a745', color: 'white', padding: '8px 15px', border: 'none', borderRadius: '4px' }}>
                Create Master Skill
            </button>
        </form>
    );
};

export default SkillForm;