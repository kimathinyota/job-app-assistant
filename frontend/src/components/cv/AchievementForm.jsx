import React, { useState, useEffect } from 'react';
import SkillManagerModal from './SkillManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

const AchievementForm = ({ onSubmit, cvId, allSkills, initialData, onCancelEdit }) => {
  const [text, setText] = useState('');
  const [selectedSkillIds, setSelectedSkillIds] = useState([]);
  const [pendingSkills, setPendingSkills] = useState([]);
  const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);

  useEffect(() => {
    if (initialData) {
      setText(initialData.text || '');
      setSelectedSkillIds(initialData.skill_ids || []);
      setPendingSkills(initialData.new_skills || []);
    } else {
      setText('');
      setSelectedSkillIds([]);
      setPendingSkills([]);
    }
  }, [initialData]);

  const handleAddAchievement = () => {
    if (!text.trim()) return;
    onSubmit(cvId, {
      text,
      context: 'Global',
      skill_ids: selectedSkillIds,
      new_skills: pendingSkills
    });
  };

  return (
    <div
      style={{
        margin: '10px 0',
        padding: '15px',
        border: '1px solid #6c757d',
        borderRadius: '5px',
        backgroundColor: '#e9ecef',
        textAlign: 'left'
      }}
    >
      <h4 style={{ margin: '0 0 10px 0', color: '#6c757d' }}>
        {initialData ? 'Edit Achievement' : '+ Add Temporary Achievement'}
      </h4>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Achievement text (e.g., Optimized X by Y%)"
        style={{
          width: '95%',
          padding: '8px',
          minHeight: '60px',
          marginBottom: '10px'
        }}
      />

      <div style={{ marginTop: '10px' }}>
        <strong>Related Skills:</strong>
        <button
          type="button"
          onClick={() => setIsSkillModalOpen(true)}
          style={{
            marginLeft: '10px',
            backgroundColor: '#6c757d',
            color: 'white',
            padding: '5px 10px'
          }}
        >
          Manage Skills
        </button>

        <SelectedSkillsDisplay
          allSkills={allSkills}
          selectedSkillIds={selectedSkillIds}
          pendingSkills={pendingSkills}
        />
      </div>

      <SkillManagerModal
        isOpen={isSkillModalOpen}
        onClose={() => setIsSkillModalOpen(false)}
        allSkills={allSkills}
        selectedSkillIds={selectedSkillIds}
        setSelectedSkillIds={setSelectedSkillIds}
        pendingSkills={pendingSkills}
        setPendingSkills={setPendingSkills}
      />

      <div style={{ marginTop: '10px' }}>
        <button
          type="button"
          onClick={handleAddAchievement}
          style={{
            backgroundColor: initialData ? '#ffc107' : '#6c757d',
            color: 'white',
            padding: '6px 12px',
            borderRadius: '5px',
            marginRight: '10px'
          }}
        >
          {initialData ? 'Save Changes' : 'Add Achievement'}
        </button>
        {initialData && (
          <button
            type="button"
            onClick={onCancelEdit}
            style={{
              backgroundColor: '#6c757d',
              color: 'white',
              padding: '6px 12px',
              borderRadius: '5px'
            }}
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  );
};

export default AchievementForm;
