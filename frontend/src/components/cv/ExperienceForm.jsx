import React, { useState } from 'react';
import SkillManagerModal from './SkillManagerModal';
import AchievementManagerModal from './AchievementManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

const ExperienceForm = ({ onSubmit, cvId, allSkills, allAchievements }) => {
  const [title, setTitle] = useState('');
  const [company, setCompany] = useState('');
  const [description, setDescription] = useState('');

  const [selectedSkillIds, setSelectedSkillIds] = useState([]);
  const [pendingSkills, setPendingSkills] = useState([]);
  const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);

  const [selectedAchievementIds, setSelectedAchievementIds] = useState([]);
  const [pendingAchievements, setPendingAchievements] = useState([]);
  const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!title.trim() || !company.trim()) return;

    onSubmit(cvId, {
      title,
      company,
      description: description || 'No description provided.',
      existing_skill_ids: selectedSkillIds,
      new_skills: pendingSkills,
      existing_achievement_ids: selectedAchievementIds,
      new_achievements: pendingAchievements
    }, 'Experience');

    // Reset
    setTitle('');
    setCompany('');
    setDescription('');
    setSelectedSkillIds([]);
    setPendingSkills([]);
    setSelectedAchievementIds([]);
    setPendingAchievements([]);
  };

  return (
    <form onSubmit={handleSubmit} style={{
      margin: '10px 0', padding: '15px', border: '1px solid #007bff',
      borderRadius: '5px', backgroundColor: '#e6f7ff', textAlign: 'left'
    }}>
      <h4 style={{ margin: '0 0 10px 0', color: '#007bff' }}>+ Add New Experience</h4>

      <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Job Title" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
      <input type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Company" required style={{ width: '95%', padding: '8px', marginBottom: '8px' }} />
      <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description (Optional)" style={{ width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />

      <div style={{ marginTop: '15px' }}>
        <strong>Related Skills:</strong>
        <button
          type="button"
          onClick={() => setIsSkillModalOpen(true)}
          style={{ padding: '6px 12px', backgroundColor: '#6c757d', color: 'white', marginLeft: '10px' }}
        >
          Manage Skills
        </button>
        <SelectedSkillsDisplay allSkills={allSkills} selectedSkillIds={selectedSkillIds} pendingSkills={pendingSkills} />
      </div>

      <div style={{ marginTop: '15px' }}>
        <strong>Linked Achievements:</strong>
        <button
          type="button"
          onClick={() => setIsAchievementModalOpen(true)}
          style={{ padding: '6px 12px', backgroundColor: '#6c757d', color: 'white', marginLeft: '10px' }}
        >
          Manage Achievements
        </button>

        {/* Show linked and pending achievements */}
        <ul style={{ marginTop: '10px' }}>
          {selectedAchievementIds.map(id => {
            const ach = allAchievements.find(a => a.id === id);
            return <li key={id}>{ach?.text || 'Unknown Achievement'}</li>;
          })}
          {pendingAchievements.map((a, idx) => (
            <li key={`pending-${idx}`}><em>{a.text} (pending)</em></li>
          ))}
        </ul>
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

      <AchievementManagerModal
        isOpen={isAchievementModalOpen}
        onClose={() => setIsAchievementModalOpen(false)}
        allAchievements={allAchievements}
        allSkills={allSkills}
        selectedAchievementIds={selectedAchievementIds}
        setSelectedAchievementIds={setSelectedAchievementIds}
        pendingAchievements={pendingAchievements}
        setPendingAchievements={setPendingAchievements}
      />

      <button type="submit" style={{
        backgroundColor: '#007bff', color: 'white',
        padding: '8px 15px', border: 'none', borderRadius: '4px',
        marginTop: '20px'
      }}>
        Create Experience
      </button>
    </form>
  );
};

export default ExperienceForm;
