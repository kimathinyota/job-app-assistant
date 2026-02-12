import React, { useState, useMemo } from 'react';
import { Plus, Search, Check } from 'lucide-react';

const SkillLinker = ({
  allSkills,
  selectedSkillIds,
  setSelectedSkillIds,
  pendingSkills,
  setPendingSkills,
  sessionSkills = [] 
}) => {
  const [newSkillName, setNewSkillName] = useState('');
  const [newSkillCategory, setNewSkillCategory] = useState('technical');
  const [searchTerm, setSearchTerm] = useState('');

  // --- NEW: Dynamic Categories Logic ---
  const availableCategories = useMemo(() => {
    const defaultCats = ['technical', 'soft', 'language', 'other'];
    // Gather categories from existing skills
    const usedCats = allSkills.map(s => s.category).filter(Boolean);
    // Combine, deduplicate, and sort
    return [...new Set([...defaultCats, ...usedCats])].sort();
  }, [allSkills]);

  // Toggle Master ID
  const handleToggleSkill = (skillId) => {
    if (selectedSkillIds.includes(skillId)) {
      setSelectedSkillIds(selectedSkillIds.filter(id => id !== skillId));
    } else {
      setSelectedSkillIds([...selectedSkillIds, skillId]);
    }
  };

  // Toggle Session Skill (Add/Remove from local pendingSkills)
  const handleToggleSessionSkill = (skill) => {
    const isAlreadyAdded = pendingSkills.some(s => s.name === skill.name);
    if (isAlreadyAdded) {
        setPendingSkills(prev => prev.filter(s => s.name !== skill.name));
    } else {
        // Add a copy of the session skill to this item's pending list
        setPendingSkills(prev => [...prev, { ...skill }]);
    }
  };

  const handleAddPendingSkill = (e) => {
    e.preventDefault();
    if (newSkillName.trim()) {
      const name = newSkillName.trim();
      
      // Check duplicates in Pending, Master, AND Session
      const isPending = pendingSkills.some(s => s.name.toLowerCase() === name.toLowerCase());
      const isMaster = allSkills.some(s => s.name.toLowerCase() === name.toLowerCase());
      const isSession = sessionSkills.some(s => s.name.toLowerCase() === name.toLowerCase());

      if (isPending || isMaster || isSession) {
        alert(`Skill "${name}" already exists.`);
        return;
      }

      setPendingSkills(prev => [...prev, { name, category: newSkillCategory }]);
      setNewSkillName('');
      setNewSkillCategory('technical');
    }
  };

  const handleRemovePendingSkill = (nameToRemove) => {
    setPendingSkills(prev => prev.filter(skill => skill.name !== nameToRemove));
  };

  // Filter Master Skills
  const filteredMasterSkills = allSkills.filter(skill => 
    skill.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Filter Session Skills (exclude ones that match master to avoid dupes visual)
  const filteredSessionSkills = sessionSkills.filter(skill => 
    skill.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
    !allSkills.some(ms => ms.name.toLowerCase() === skill.name.toLowerCase())
  );

  return (
    <div className="bg-white rounded-xl border shadow-sm p-4">
      
      {/* 1. Add New */}
      <div className="pb-4 border-bottom mb-4">
        <label className="form-label fw-bold small text-uppercase text-muted mb-3">
          Create New Skill (Local)
        </label>
        
        <div className="d-flex flex-column flex-sm-row gap-2">
            <input
                type="text"
                value={newSkillName}
                onChange={(e) => setNewSkillName(e.target.value)}
                placeholder="e.g., React Native"
                className="form-control"
            />
            
            {/* Dynamic Category Input */}
            <input
                list="linker-category-options"
                value={newSkillCategory}
                onChange={(e) => setNewSkillCategory(e.target.value)}
                className="form-control bg-light"
                placeholder="Category"
                style={{ maxWidth: '130px' }}
            />
             <datalist id="linker-category-options">
                {availableCategories.map((cat, index) => (
                    <option key={`${cat}-${index}`} value={cat} />
                ))}
            </datalist>

            <button
                type="button"
                onClick={handleAddPendingSkill}
                disabled={!newSkillName.trim()}
                className="btn btn-success d-flex align-items-center gap-2"
            >
                <Plus size={18} /> Add
            </button>
        </div>

        {/* Display Currently Added (Pending) */}
        {pendingSkills.length > 0 && (
            <div className="mt-3 d-flex flex-wrap gap-2">
                {pendingSkills.map((skill, index) => (
                <span 
                    key={index} 
                    className="px-3 py-2 rounded-pill bg-emerald-50 text-emerald-700 border border-emerald-200 d-flex align-items-center gap-2 small fw-medium"
                >
                    {skill.name} 
                    <button
                        type="button"
                        onClick={() => handleRemovePendingSkill(skill.name)}
                        className="btn-close btn-close-sm"
                        style={{ fontSize: '0.6em' }} 
                    ></button>
                </span>
                ))}
            </div>
        )}
      </div>

      {/* 2. Link Existing (Master & Session) */}
      <div>
        <div className="d-flex justify-content-between align-items-center mb-2">
            <label className="form-label fw-bold small text-uppercase text-muted mb-0">
                Link Available Skills
            </label>
        </div>
        
        <div className="position-relative mb-3">
            <Search className="position-absolute text-muted" size={16} style={{ top: '10px', left: '12px' }} />
            <input 
                type="text" 
                className="form-control ps-5 form-control-sm rounded-pill"
                placeholder="Search skills..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
            />
        </div>

        <div className="d-flex flex-wrap gap-2 p-2 bg-light rounded-xl border inner-shadow" style={{ maxHeight: '200px', overflowY: 'auto' }}>
            
            {/* A. SESSION SKILLS (Reusable) */}
            {filteredSessionSkills.map((skill, idx) => {
                const isAdded = pendingSkills.some(s => s.name === skill.name);
                return (
                    <div
                        key={`session-${idx}`}
                        onClick={() => handleToggleSessionSkill(skill)}
                        className={`
                            px-3 py-1 rounded-pill border cursor-pointer user-select-none d-flex align-items-center gap-2 transition-all small fw-medium
                            ${isAdded 
                                ? 'bg-emerald-600 text-white border-emerald-600 shadow-sm' 
                                : 'bg-white text-emerald-700 border-emerald-200 border-dashed hover:bg-emerald-50'}
                        `}
                    >
                        {isAdded ? <Check size={14} /> : <Plus size={14} />}
                        {skill.name} <span className="opacity-75 text-xs">(New)</span>
                    </div>
                );
            })}

            {/* B. MASTER SKILLS */}
            {filteredMasterSkills.map(skill => {
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
                    >
                        {isSelected && <Check size={14} />}
                        {skill.name}
                    </div>
                );
            })}

            {filteredMasterSkills.length === 0 && filteredSessionSkills.length === 0 && (
                <div className="w-100 text-center py-4 text-muted fst-italic small">
                    No matching skills available.
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default SkillLinker;