// frontend/src/components/cv/SkillLinker.jsx
import React, { useState } from 'react';

const SkillLinker = ({
  allSkills,
  selectedSkillIds,
  setSelectedSkillIds,
  pendingSkills,
  setPendingSkills
}) => {
  const [newSkillName, setNewSkillName] = useState('');
  const [newSkillCategory, setNewSkillCategory] = useState('technical');

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

  return (
    // Use Bootstrap card and colors
    <div className="card bg-light-subtle border-success p-3">
      <strong className="d-block mb-2 fs-6 text-success">
        Skill Management
      </strong>

      {/* 1️⃣ Add Pending Skill Section */}
      <div className="pb-3 border-bottom mb-3">
        <p className="mb-1 fw-bold">
          Add New Skill (will be created on save):
        </p>
        <div className="input-group">
            <input
                type="text"
                value={newSkillName}
                onChange={(e) => setNewSkillName(e.target.value)}
                placeholder="e.g., Python, Team Leadership"
                className="form-control"
            />
            <select
                value={newSkillCategory}
                onChange={(e) => setNewSkillCategory(e.target.value)}
                className="form-select"
                style={{ flex: '0 0 120px' }} // Give select a fixed width
            >
                <option value="technical">Tech</option>
                <option value="soft">Soft</option>
                <option value="language">Lang</option>
                <option value="other">Other</option>
            </select>
            <button
                type="button"
                onClick={handleAddPendingSkill}
                disabled={!newSkillName.trim()}
                className="btn btn-success"
            >
                Add Pending
            </button>
        </div>
      </div>

      {/* 2️⃣ Show Pending Skills */}
      {pendingSkills.length > 0 && (
        <div className="mt-2 mb-3">
          <strong className="form-label small">Skills to Create:</strong>
          <div className="d-flex flex-wrap gap-2 mt-1">
            {pendingSkills.map((skill, index) => (
              <span 
                key={index} 
                className="badge rounded-pill text-bg-success-subtle border border-success-subtle fw-medium d-flex align-items-center p-2"
              >
                {skill.name} ({skill.category.substring(0, 4)})
                <button
                  type="button"
                  onClick={() => handleRemovePendingSkill(skill.name)}
                  className="btn-close btn-close-sm ms-1"
                  aria-label="Remove pending skill"
                ></button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 3️⃣ Link Existing Skills */}
      <strong className="d-block mb-1 mt-3 border-top pt-3 fw-bold">
        Link Existing Skills:
      </strong>
      <p className="form-text mt-0 mb-2">
        {allSkills.length} master skills available. Select skills used here.
      </p>

      <div 
        className="d-flex flex-wrap gap-2 border p-2 rounded"
        style={{ maxHeight: '150px', overflowY: 'auto' }}
      >
        {allSkills.map(skill => (
            <React.Fragment key={skill.id}>
                <input
                    type="checkbox"
                    className="btn-check"
                    id={`skill-check-${skill.id}`}
                    checked={selectedSkillIds.includes(skill.id)}
                    onChange={() => handleToggleSkill(skill.id)}
                    autoComplete="off"
                />
                <label 
                    className={`btn btn-sm ${selectedSkillIds.includes(skill.id) ? 'btn-success' : 'btn-outline-secondary'}`}
                    htmlFor={`skill-check-${skill.id}`}
                >
                    {skill.name}
                </label>
            </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default SkillLinker;