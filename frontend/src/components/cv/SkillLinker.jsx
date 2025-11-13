// frontend/src/components/cv/SkillLinker.jsx
import React, { useState } from 'react';
import { Plus, Search, Check, X } from 'lucide-react';

const SkillLinker = ({
  allSkills,
  selectedSkillIds,
  setSelectedSkillIds,
  pendingSkills,
  setPendingSkills
}) => {
  const [newSkillName, setNewSkillName] = useState('');
  const [newSkillCategory, setNewSkillCategory] = useState('technical');
  const [searchTerm, setSearchTerm] = useState('');

  const handleToggleSkill = (skillId) => {
    if (selectedSkillIds.includes(skillId)) {
      setSelectedSkillIds(selectedSkillIds.filter(id => id !== skillId));
    } else {
      setSelectedSkillIds([...selectedSkillIds, skillId]);
    }
  };

  const handleAddPendingSkill = (e) => {
    e.preventDefault();
    if (newSkillName.trim()) {
      const isPending = pendingSkills.some(
        skill => skill.name.toLowerCase() === newSkillName.trim().toLowerCase()
      );
      const exists = allSkills.some(
        skill => skill.name.toLowerCase() === newSkillName.trim().toLowerCase()
      );

      if (isPending || exists) {
        alert(`Skill "${newSkillName.trim()}" already exists or is pending creation.`);
        return;
      }

      setPendingSkills(prev => [
        ...prev,
        { name: newSkillName.trim(), category: newSkillCategory }
      ]);

      setNewSkillName('');
      setNewSkillCategory('technical');
    }
  };

  const handleRemovePendingSkill = (nameToRemove) => {
    setPendingSkills(prev => prev.filter(skill => skill.name !== nameToRemove));
  };

  // Filter existing skills
  const filteredSkills = allSkills.filter(skill => 
    skill.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="bg-white rounded-xl border shadow-sm p-4">
      
      {/* 1. Add Pending Skill Section */}
      <div className="pb-4 border-bottom mb-4">
        <label className="form-label fw-bold small text-uppercase text-muted mb-3">
          Create New Skill (Local to this item)
        </label>
        <div className="input-group">
            <input
                type="text"
                value={newSkillName}
                onChange={(e) => setNewSkillName(e.target.value)}
                placeholder="e.g., React Native, Crisis Management"
                className="form-control"
            />
            <select
                value={newSkillCategory}
                onChange={(e) => setNewSkillCategory(e.target.value)}
                className="form-select bg-light"
                style={{ maxWidth: '130px' }}
            >
                <option value="technical">Technical</option>
                <option value="soft">Soft</option>
                <option value="language">Language</option>
                <option value="other">Other</option>
            </select>
            <button
                type="button"
                onClick={handleAddPendingSkill}
                disabled={!newSkillName.trim()}
                className="btn btn-success d-flex align-items-center gap-2"
            >
                <Plus size={18} /> Add
            </button>
        </div>

        {/* Pending Skills Display (FIXED: Removed 'badge' class) */}
        {pendingSkills.length > 0 && (
            <div className="mt-3 d-flex flex-wrap gap-2">
                {pendingSkills.map((skill, index) => (
                <span 
                    key={index} 
                    className="px-3 py-2 rounded-pill bg-emerald-50 text-emerald-700 border border-emerald-200 d-flex align-items-center gap-2 small fw-medium"
                >
                    {skill.name} 
                    <span className="opacity-50 ms-1" style={{fontSize: '0.75em'}}>({skill.category})</span>
                    <button
                        type="button"
                        onClick={() => handleRemovePendingSkill(skill.name)}
                        className="btn-close btn-close-sm" // This works because parent isn't forcing white text
                        aria-label="Remove"
                        style={{ fontSize: '0.6em' }} 
                    ></button>
                </span>
                ))}
            </div>
        )}
      </div>

      {/* 2. Link Existing Skills */}
      <div>
        <div className="d-flex justify-content-between align-items-center mb-2">
            <label className="form-label fw-bold small text-uppercase text-muted mb-0">
                Link Master Skills
            </label>
            <span className="badge bg-light text-muted border rounded-pill px-3">{selectedSkillIds.length} Selected</span>
        </div>
        
        {/* Search Bar */}
        <div className="position-relative mb-3">
            <Search className="position-absolute text-muted" size={16} style={{ top: '10px', left: '12px' }} />
            <input 
                type="text" 
                className="form-control ps-5 form-control-sm rounded-pill"
                placeholder="Search master skills..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
            />
        </div>

        <div 
            className="d-flex flex-wrap gap-2 p-2 bg-light rounded-xl border inner-shadow"
            style={{ maxHeight: '200px', overflowY: 'auto' }}
        >
            {filteredSkills.length === 0 ? (
                <div className="w-100 text-center py-4 text-muted fst-italic small">
                    {searchTerm ? "No matching skills found." : "No master skills available."}
                </div>
            ) : (
                filteredSkills.map(skill => {
                    const isSelected = selectedSkillIds.includes(skill.id);
                    return (
                        <div
                            key={skill.id}
                            onClick={() => handleToggleSkill(skill.id)}
                            className={`
                                px-3 py-1 rounded-pill border cursor-pointer user-select-none d-flex align-items-center gap-2 transition-all small fw-medium
                                ${isSelected 
                                    ? 'bg-primary text-white border-primary shadow-sm' 
                                    : 'bg-white text-secondary border-light-subtle hover:bg-gray-100'}
                            `}
                            style={{ cursor: 'pointer' }}
                        >
                            {isSelected && <Check size={14} />}
                            {skill.name}
                        </div>
                    );
                })
            )}
        </div>
      </div>
    </div>
  );
};

export default SkillLinker;