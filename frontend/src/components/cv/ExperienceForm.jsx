import React, { useState, useEffect } from 'react';
import SkillManagerModal from './SkillManagerModal';
import AchievementManagerModal from './AchievementManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const ExperienceForm = ({
  onSubmit,
  cvId,
  allSkills,
  allAchievements,
  initialData,
  onCancelEdit
}) => {
  const [title, setTitle] = useState('');
  const [company, setCompany] = useState('');
  const [description, setDescription] = useState('');
  const [selectedSkillIds, setSelectedSkillIds] = useState([]);
  const [pendingSkills, setPendingSkills] = useState([]);
  const [selectedAchievementIds, setSelectedAchievementIds] = useState([]);
  const [pendingAchievements, setPendingAchievements] = useState([]);
  const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
  const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

  // Populate form when editing
  useEffect(() => {
    if (initialData) {
      setTitle(initialData.title || '');
      setCompany(initialData.company || '');
      setDescription(initialData.description || '');
      setSelectedSkillIds(initialData.existing_skill_ids || initialData.skill_ids || []);
      setPendingSkills(initialData.new_skills || []);
      setSelectedAchievementIds(
        initialData.existing_achievement_ids || initialData.achievement_ids || []
      );
      setPendingAchievements(initialData.new_achievements || []);
    } else {
      setTitle('');
      setCompany('');
      setDescription('');
      setSelectedSkillIds([]);
      setPendingSkills([]);
      setSelectedAchievementIds([]);
      setPendingAchievements([]);
    }
  }, [initialData]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !company.trim()) return;

    onSubmit(
      cvId,
      {
        title,
        company,
        description,
        existing_skill_ids: selectedSkillIds,
        new_skills: pendingSkills,
        existing_achievement_ids: selectedAchievementIds,
        new_achievements: pendingAchievements,
      },
      'Experience'
    );
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        margin: '15px 0',
        padding: '15px',
        border: '1px solid #007bff',
        borderRadius: '8px',
        backgroundColor: '#f0f8ff',
      }}
    >
      <h3 style={{ color: '#007bff', marginBottom: '10px' }}>
        {initialData ? 'Edit Experience' : '+ Add New Experience'}
      </h3>

      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Job Title"
        required
        style={{ display: 'block', width: '95%', marginBottom: '10px', padding: '8px' }}
      />

      <input
        type="text"
        value={company}
        onChange={(e) => setCompany(e.target.value)}
        placeholder="Company"
        required
        style={{ display: 'block', width: '95%', marginBottom: '10px', padding: '8px' }}
      />

      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Describe your role..."
        style={{ width: '95%', padding: '8px', minHeight: '60px' }}
      />

      {/* SKILLS */}
      <div style={{ marginTop: '15px' }}>
        <strong>Skills:</strong>
        <button
          type="button"
          onClick={() => setIsSkillModalOpen(true)}
          style={{
            marginLeft: '10px',
            backgroundColor: '#6c757d',
            color: 'white',
            padding: '5px 10px',
            borderRadius: '5px',
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

      {/* ACHIEVEMENTS */}
      <div style={{ marginTop: '15px' }}>
        <strong>Achievements:</strong>
        <button
          type="button"
          onClick={() => setIsAchievementModalOpen(true)}
          style={{
            marginLeft: '10px',
            backgroundColor: '#6c757d',
            color: 'white',
            padding: '5px 10px',
            borderRadius: '5px',
          }}
        >
          Manage Achievements
        </button>
        <AchievementDisplayGrid
          allAchievements={allAchievements}
          selectedAchievementIds={selectedAchievementIds}
          pendingAchievements={pendingAchievements}
          onRemoveAchievement={(id) => {
            setSelectedAchievementIds((prev) => prev.filter((a) => a !== id));
            setPendingAchievements((prev) => prev.filter((a) => a.id !== id));
          }}
        />
      </div>

      {/* SKILL MODAL */}
      <SkillManagerModal
        isOpen={isSkillModalOpen}
        onClose={() => setIsSkillModalOpen(false)}
        allSkills={allSkills}
        selectedSkillIds={selectedSkillIds}
        setSelectedSkillIds={setSelectedSkillIds}
        pendingSkills={pendingSkills}
        setPendingSkills={setPendingSkills}
      />

      {/* ACHIEVEMENT MODAL */}
      <AchievementManagerModal
        isOpen={isAchievementModalOpen}
        onClose={() => setIsAchievementModalOpen(false)}
        allAchievements={allAchievements}
        selectedAchievementIds={selectedAchievementIds}
        setSelectedAchievementIds={setSelectedAchievementIds}
        pendingAchievements={pendingAchievements}
        setPendingAchievements={setPendingAchievements}
        allSkills={allSkills}
      />

      {/* ACTIONS */}
      <div style={{ marginTop: '15px' }}>
        <button
          type="submit"
          style={{
            backgroundColor: '#007bff',
            color: 'white',
            padding: '8px 15px',
            borderRadius: '5px',
          }}
        >
          {initialData ? 'Save Changes' : 'Add Experience'}
        </button>
        {initialData && (
          <button
            type="button"
            onClick={onCancelEdit}
            style={{
              marginLeft: '10px',
              backgroundColor: '#6c757d',
              color: 'white',
              padding: '8px 15px',
              borderRadius: '5px',
            }}
          >
            Cancel
          </button>
        )}
      </div>
    </form>
  );
};

export default ExperienceForm;
