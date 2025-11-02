// frontend/src/components/cv/SkillForm.jsx
import React, { useState, useEffect } from 'react';
import { addSkill } from '../../api/cvClient'; // Keep addSkill for potential non-manager use

const SkillForm = ({ 
    onSubmit, 
    cvId, 
    initialData, 
    onCancelEdit // <-- ADD THIS PROP
}) => {
    const [name, setName] = useState('');
    const [category, setCategory] = useState('technical');
    const [level, setLevel] = useState('');
    const [importance, setImportance] = useState('');
    const [description, setDescription] = useState('');

    const isEditing = Boolean(initialData);

    useEffect(() => {
        if (isEditing) {
            setName(initialData.name || '');
            setCategory(initialData.category || 'technical');
            setLevel(initialData.level || '');
            setImportance(initialData.importance || '');
            setDescription(initialData.description || '');
        } else {
            // Reset for "create new"
            setName('');
            setCategory('technical');
            setLevel('');
            setImportance('');
            setDescription('');
        }
    }, [initialData, isEditing]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!name.trim()) return;

        const skillData = {
            name,
            category,
            level: level || null,
            importance: importance ? parseInt(importance, 10) : null,
            description: description || null,
        };
        
        if(isEditing) {
            skillData.id = initialData.id;
        }

        // Use the manager's onSubmit handler
        onSubmit(cvId, skillData, 'Skill');
        
        // Clear form only if we are in "create" mode
        if (!isEditing) {
            setName('');
            setCategory('technical');
            setLevel('');
            setImportance('');
            setDescription('');
        }
    };

    return (
        <form 
            onSubmit={handleSubmit} 
            className="card p-3" 
            style={{ borderTop: `4px solid #0dcaf0` }} // Skill theme color
        >
            <h4 className="mt-0 mb-3" style={{ color: '#0dcaf0' }}>
                {isEditing ? 'Edit Master Skill' : 'Add New Master Skill'}
            </h4>
            
            {/* Form Row 1: Name and Category */}
            <div className="row g-2 mb-3">
                <div className="col-md-6">
                    <label htmlFor="skill-name" className="form-label fw-medium">Skill Name</label>
                    <input
                        id="skill-name"
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder="e.g., Python, Teamwork"
                        required
                        className="form-control"
                    />
                </div>
                <div className="col-md-6">
                    <label htmlFor="skill-category" className="form-label fw-medium">Category</label>
                    <select
                        id="skill-category"
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                        className="form-select"
                    >
                        <option value="technical">Technical</option>
                        <option value="soft">Soft</option>
                        <option value="language">Language</option>
                        <option value="other">Other</option>
                    </select>
                </div>
            </div>

            {/* Form Row 2: Level and Importance */}
            <div className="row g-2 mb-3">
                <div className="col-md-6">
                    <label htmlFor="skill-level" className="form-label fw-medium">Level (Optional)</label>
                    <input
                        id="skill-level"
                        type="text"
                        value={level}
                        onChange={(e) => setLevel(e.target.value)}
                        placeholder="e.g., Expert, Proficient"
                        className="form-control"
                    />
                </div>
                <div className="col-md-6">
                    <label htmlFor="skill-importance" className="form-label fw-medium">Importance (1-5, Optional)</label>
                    <input
                        id="skill-importance"
                        type="number"
                        value={importance}
                        onChange={(e) => setImportance(e.target.value)}
                        min="1"
                        max="5"
                        className="form-control"
                    />
                </div>
            </div>
            
            {/* Form Row 3: Description */}
            <div className="mb-3">
                <label htmlFor="skill-description" className="form-label fw-medium">Description (Optional)</label>
                <textarea
                    id="skill-description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    rows="2"
                    placeholder="e.g., Used for data analysis and web backends"
                    className="form-control"
                ></textarea>
            </div>
            
            {/* Action Buttons */}
            <div className="mt-3 border-top pt-3">
                <button type="submit" className="btn btn-primary me-2">
                    {isEditing ? 'Save Changes' : 'Add Master Skill'}
                </button>
                
                {/* ADDED THIS CANCEL BUTTON */}
                {onCancelEdit && (
                    <button 
                        type="button" 
                        onClick={onCancelEdit} 
                        className="btn btn-outline-secondary"
                    >
                        Cancel
                    </button>
                )}
            </div>
        </form>
    );
};

export default SkillForm;