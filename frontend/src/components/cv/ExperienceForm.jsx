import React, { useState, useEffect } from 'react';
import SkillManagerModal from './SkillManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementManagerModal from './AchievementManagerModal';
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
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [description, setDescription] = useState('');
  const [selectedSkillIds, setSelectedSkillIds] = useState([]);
  const [pendingSkills, setPendingSkills] = useState([]);
  const [selectedAchievementIds, setSelectedAchievementIds] = useState([]);
  const [pendingAchievements, setPendingAchievements] = useState([]);
  const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
  const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

  const isEditing = Boolean(initialData);

  useEffect(() => {
    if (isEditing) {
      setTitle(initialData.title || '');
      setCompany(initialData.company || '');
      setStartDate(initialData.start_date || '');
      setEndDate(initialData.end_date || '');
      setDescription(initialData.description || '');
      setSelectedSkillIds(initialData.skill_ids || []);
      setSelectedAchievementIds(initialData.achievement_ids || []);
      setPendingSkills([]);
      setPendingAchievements([]);
    } else {
      setTitle('');
      setCompany('');
      setStartDate('');
      setEndDate('');
      setDescription('');
      setSelectedSkillIds([]);
      setPendingSkills([]);
      setSelectedAchievementIds([]);
      setPendingAchievements([]);
    }
  }, [initialData, isEditing]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !company.trim()) return;

    const dataToSend = {
      title,
      company,
      start_date: startDate || null,
      end_date: endDate || null,
      description: description || null,
      existing_skill_ids: selectedSkillIds,
      new_skills: pendingSkills,
      existing_achievement_ids: selectedAchievementIds,
      new_achievements: pendingAchievements
    };
    if (isEditing) dataToSend.id = initialData.id;

    onSubmit(cvId, dataToSend, 'Experience');

    if (!isEditing) {
      setTitle('');
      setCompany('');
      setStartDate('');
      setEndDate('');
      setDescription('');
      setSelectedSkillIds([]);
      setPendingSkills([]);
      setSelectedAchievementIds([]);
      setPendingAchievements([]);
    }
  };

  return (
    <form
      key={initialData?.id || 'new'}
      onSubmit={handleSubmit}
      style={{
        margin: '15px 0',
        padding: '15px',
        border: '1px solid #0d6efd',
        borderRadius: '8px',
        backgroundColor: '#f8f9fa', // light neutral background for contrast
        color: '#212529', // default dark text
        textAlign: 'left',
        boxShadow: '0 2px 6px rgba(0,0,0,0.1)'
      }}
    >
      <h3
        style={{
          marginBottom: '15px',
          marginTop: 0,
          color: '#0d6efd', // accessible blue
          fontWeight: '600'
        }}
      >
        {isEditing ? 'Edit Experience' : '+ Add New Experience'}
      </h3>

      {/* Inputs */}
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Job Title"
        required
        style={{
          display: 'block',
          width: '95%',
          marginBottom: '10px',
          padding: '8px',
          border: '1px solid #adb5bd',
          borderRadius: '4px',
          backgroundColor: '#ffffff',
          color: '#212529'
        }}
      />

      <input
        type="text"
        value={company}
        onChange={(e) => setCompany(e.target.value)}
        placeholder="Company"
        required
        style={{
          display: 'block',
          width: '95%',
          marginBottom: '10px',
          padding: '8px',
          border: '1px solid #adb5bd',
          borderRadius: '4px',
          backgroundColor: '#ffffff',
          color: '#212529'
        }}
      />

      <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
        <input
          type="text"
          value={startDate}
          onChange={(e) => setStartDate(e.target.value)}
          placeholder="Start Date"
          style={{
            flex: 1,
            padding: '8px',
            border: '1px solid #adb5bd',
            borderRadius: '4px',
            backgroundColor: '#ffffff',
            color: '#212529'
          }}
        />
        <input
          type="text"
          value={endDate}
          onChange={(e) => setEndDate(e.target.value)}
          placeholder="End Date"
          style={{
            flex: 1,
            padding: '8px',
            border: '1px solid #adb5bd',
            borderRadius: '4px',
            backgroundColor: '#ffffff',
            color: '#212529'
          }}
        />
      </div>

      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Description..."
        style={{
          display: 'block',
          width: '95%',
          padding: '8px',
          minHeight: '60px',
          marginBottom: '10px',
          border: '1px solid #adb5bd',
          borderRadius: '4px',
          backgroundColor: '#ffffff',
          color: '#212529'
        }}
      />

      {/* Skills Section */}
      <div style={{ marginTop: '15px' }}>
        <strong style={{ display: 'block', marginBottom: '5px', color: '#343a40' }}>
          Skills:
        </strong>
        <button
          type="button"
          onClick={() => setIsSkillModalOpen(true)}
          style={{
            marginLeft: '10px',
            backgroundColor: '#495057',
            color: '#ffffff',
            padding: '6px 12px',
            borderRadius: '5px',
            border: 'none',
            cursor: 'pointer'
          }}
          onMouseOver={(e) => (e.target.style.backgroundColor = '#343a40')}
          onMouseOut={(e) => (e.target.style.backgroundColor = '#495057')}
        >
          Manage Skills
        </button>
        <SelectedSkillsDisplay
          allSkills={allSkills}
          selectedSkillIds={selectedSkillIds}
          pendingSkills={pendingSkills}
        />
      </div>

      {/* Achievements Section */}
      <div style={{ marginTop: '15px' }}>
        <strong style={{ display: 'block', marginBottom: '5px', color: '#343a40' }}>
          Achievements:
        </strong>
        <button
          type="button"
          onClick={() => setIsAchievementModalOpen(true)}
          style={{
            marginLeft: '10px',
            backgroundColor: '#495057',
            color: '#ffffff',
            padding: '6px 12px',
            borderRadius: '5px',
            border: 'none',
            cursor: 'pointer'
          }}
          onMouseOver={(e) => (e.target.style.backgroundColor = '#343a40')}
          onMouseOut={(e) => (e.target.style.backgroundColor = '#495057')}
        >
          Manage Achievements
        </button>

        <AchievementDisplayGrid
          allSkills={allSkills}
          allAchievements={allAchievements}
          selectedAchievementIds={selectedAchievementIds}
          pendingAchievements={pendingAchievements}
          isDisplayOnly={true}
        />
      </div>

      {/* Modals */}
      <SkillManagerModal
        isOpen={isSkillModalOpen}
        onClose={() => setIsSkillModalOpen(false)}
        allSkills={allSkills}
        selectedSkillIds={selectedSkillIds}
        setSelectedSkillIds={setSelectedSkillIds}
        pendingSkills={pendingSkills}
        setPendingSkills={setPendingSkills}
      />

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

      {/* Action Buttons */}
      <div style={{ marginTop: '20px' }}>
        <button
          type="submit"
          style={{
            backgroundColor: '#0d6efd',
            color: '#ffffff',
            padding: '8px 15px',
            borderRadius: '5px',
            border: 'none',
            cursor: 'pointer'
          }}
          onMouseOver={(e) => (e.target.style.backgroundColor = '#0b5ed7')}
          onMouseOut={(e) => (e.target.style.backgroundColor = '#0d6efd')}
        >
          {isEditing ? 'Save Changes' : 'Add Experience'}
        </button>

        {isEditing && (
          <button
            type="button"
            onClick={onCancelEdit}
            style={{
              marginLeft: '10px',
              backgroundColor: '#6c757d',
              color: '#ffffff',
              padding: '8px 15px',
              borderRadius: '5px',
              border: 'none',
              cursor: 'pointer'
            }}
            onMouseOver={(e) => (e.target.style.backgroundColor = '#5a6268')}
            onMouseOut={(e) => (e.target.style.backgroundColor = '#6c757d')}
          >
            Cancel Edit
          </button>
        )}
      </div>
    </form>
  );
};

export default ExperienceForm;
