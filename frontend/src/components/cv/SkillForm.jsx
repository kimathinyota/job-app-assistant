// frontend/src/components/cv/SkillForm.jsx
import React, { useState, useEffect } from 'react';
import { Layers } from 'lucide-react'; 

const SkillForm = ({ 
    onSubmit, 
    cvId, 
    initialData, 
    onCancelEdit,
    existingCategories = ['technical', 'soft', 'language', 'other'] // Default fallback
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

        onSubmit(cvId, skillData, 'Skill');
        
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
            className="card border-0 shadow-sm p-4" 
        >
            {/* Header Title */}
            <div className="d-flex align-items-center gap-2 mb-4 border-bottom pb-2">
                <Layers className="text-emerald-600" size={20}/>
                <h5 className="mb-0 fw-bold text-dark">
                    {isEditing ? 'Edit Master Skill' : 'Add New Master Skill'}
                </h5>
            </div>
            
            {/* Form Row 1: Name and Category */}
            <div className="row g-3 mb-3">
                <div className="col-md-6">
                    <label htmlFor="skill-name" className="form-label fw-bold small text-uppercase text-muted">Skill Name</label>
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
                    <label htmlFor="skill-category" className="form-label fw-bold small text-uppercase text-muted">Category</label>
                    
                    <input
                        list="category-options"
                        id="skill-category"
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                        className="form-control"
                        placeholder="Select or Type..."
                    />
                    <datalist id="category-options">
                        {existingCategories.map((cat, index) => (
                             <option key={`${cat}-${index}`} value={cat} />
                        ))}
                    </datalist>
                    
                    {/* Visual Helper: Quick Select Chips */}
                    <div className="mt-2 d-flex flex-wrap gap-1">
                        <small className="text-muted me-1">Suggested:</small>
                        {existingCategories.slice(0, 5).map(cat => (
                            <span 
                                key={cat}
                                onClick={() => setCategory(cat)}
                                className="badge bg-light text-secondary border cursor-pointer hover:bg-gray-200"
                            >
                                {cat}
                            </span>
                        ))}
                    </div>

                </div>
            </div>

            {/* Form Row 2: Level and Importance */}
            <div className="row g-3 mb-3">
                <div className="col-md-6">
                    <label htmlFor="skill-level" className="form-label fw-bold small text-uppercase text-muted">Level (Optional)</label>
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
                    <label htmlFor="skill-importance" className="form-label fw-bold small text-uppercase text-muted">Importance (1-5)</label>
                    <input
                        id="skill-importance"
                        type="number"
                        value={importance}
                        onChange={(e) => setImportance(e.target.value)}
                        min="1"
                        max="5"
                        placeholder="Rank 1-5"
                        className="form-control"
                    />
                </div>
            </div>
            
            {/* Form Row 3: Description */}
            <div className="mb-3">
                <label htmlFor="skill-description" className="form-label fw-bold small text-uppercase text-muted">Description (Optional)</label>
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
            <div className="d-flex gap-2 justify-content-end mt-4 pt-3 border-top">
                {onCancelEdit && (
                    <button 
                        type="button" 
                        onClick={onCancelEdit} 
                        className="btn btn-light border"
                    >
                        Cancel
                    </button>
                )}
                <button type="submit" className="btn btn-primary px-4">
                    {isEditing ? 'Save Changes' : 'Add Master Skill'}
                </button>
            </div>
        </form>
    );
};

export default SkillForm;