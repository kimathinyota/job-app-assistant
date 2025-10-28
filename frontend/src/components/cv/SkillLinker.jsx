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
    e.preventDefault(); // just in case
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
    <div
      style={{
        marginTop: '15px',
        padding: '15px',
        border: '2px solid #28a745',
        borderRadius: '8px',
        backgroundColor: '#f0fff0'
      }}
    >
      <strong
        style={{
          display: 'block',
          marginBottom: '10px',
          fontSize: '1.1em',
          color: '#28a745'
        }}
      >
        Skill Management
      </strong>

      {/* 1️⃣ Add Pending Skill Section */}
      <div
        style={{
          paddingBottom: '10px',
          borderBottom: '1px dotted #ccc',
          marginBottom: '10px'
        }}
      >
        <p style={{ margin: '0 0 5px 0', fontWeight: 'bold' }}>
          Add New Skill (will be created on save):
        </p>
        <input
          type="text"
          value={newSkillName}
          onChange={(e) => setNewSkillName(e.target.value)}
          placeholder="e.g., Python, Team Leadership"
          style={{ width: '60%', padding: '8px', marginRight: '5px' }}
        />
        <select
          value={newSkillCategory}
          onChange={(e) => setNewSkillCategory(e.target.value)}
          style={{ width: '20%', padding: '8px', marginRight: '5px' }}
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
          style={{
            padding: '8px 10px',
            backgroundColor: '#28a745',
            color: 'white',
            cursor: 'pointer',
            border: 'none',
            borderRadius: '5px'
          }}
        >
          Add Pending
        </button>
      </div>

      {/* 2️⃣ Show Pending Skills */}
      {pendingSkills.length > 0 && (
        <div style={{ marginTop: '10px', marginBottom: '10px' }}>
          <strong style={{ fontSize: '0.9em' }}>Skills to Create:</strong>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '5px',
              marginTop: '5px'
            }}
          >
            {pendingSkills.map((skill, index) => (
              <span
                key={index}
                style={{
                  backgroundColor: '#e2e3e5',
                  padding: '3px 8px',
                  borderRadius: '15px',
                  fontSize: '0.85em',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                {skill.name} ({skill.category.substring(0, 4)})
                <button
                  type="button"
                  onClick={() => handleRemovePendingSkill(skill.name)}
                  style={{
                    marginLeft: '5px',
                    padding: '0 5px',
                    fontSize: '0.7em',
                    color: 'red',
                    backgroundColor: 'transparent',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                  title="Remove pending skill"
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 3️⃣ Link Existing Skills */}
      <strong
        style={{
          display: 'block',
          marginBottom: '8px',
          marginTop: '15px',
          borderTop: '1px dotted #ccc',
          paddingTop: '10px'
        }}
      >
        Link Existing Skills:
      </strong>
      <p style={{ fontSize: '0.8em', color: '#666', marginTop: 0 }}>
        {allSkills.length} master skills available. Select skills used here.
      </p>

      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '8px',
          maxHeight: '150px',
          overflowY: 'auto',
          border: '1px solid #eee',
          padding: '5px'
        }}
      >
        {allSkills.map(skill => (
          <label
            key={skill.id}
            style={{
              display: 'flex',
              alignItems: 'center',
              fontSize: '0.9em',
              padding: '4px 8px',
              border: '1px solid #ddd',
              borderRadius: '20px',
              cursor: 'pointer',
              backgroundColor: selectedSkillIds.includes(skill.id)
                ? '#d4edda'
                : 'white',
              transition: 'background-color 0.2s'
            }}
          >
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
