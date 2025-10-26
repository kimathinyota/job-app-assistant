import React, { useState } from 'react';

const ProjectForm = ({ onSubmit, cvId }) => {
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (title.trim() && description.trim()) {
            onSubmit(cvId, { title, description }, 'Project');
            setTitle('');
            setDescription('');
        }
    };

    return (
        <form onSubmit={handleSubmit} style={{ margin: '10px 0', padding: '15px', border: '1px solid #ffc107', borderRadius: '5px', backgroundColor: '#fff8e1', textAlign: 'left' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#ffc107' }}>+ Add New Project</h4>
            
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Project Title" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Project Description (Required)" required style={{ width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />
            
            <button type="submit" style={{ backgroundColor: '#ffc107', color: '#333', padding: '8px 15px', border: 'none', borderRadius: '4px' }}>
                Create Project
            </button>
        </form>
    );
};

export default ProjectForm;
