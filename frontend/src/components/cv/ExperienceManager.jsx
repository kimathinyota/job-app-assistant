// frontend/src/components/cv/ExperienceManager.jsx
import React, { useState } from 'react';
import { Plus, Edit2, Trash2, Briefcase } from 'lucide-react';
import ExperienceForm from './ExperienceForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
// --- Import the helper ---
import { formatDateRange } from '../../utils/cvHelpers';

const ExperienceManager = ({
  cvId,
  experiences = [],
  allSkills = [],
  allAchievements = [],
  onSubmit,
  onDelete,
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);

  const getAchievements = (achievementIds = []) => {
    if (!achievementIds || achievementIds.length === 0) return [];
    return achievementIds
      .map(id => allAchievements.find(a => a.id === id))
      .filter(Boolean); // Ensure no nulls if achievement was deleted
  };

  const handleAddNewClick = () => { setIsCreating(true); setEditingId(null); };
  const handleEditClick = (itemId) => { setEditingId(itemId); setIsCreating(false); };
  const handleCancel = () => { setIsCreating(false); setEditingId(null); };

  const handleSubmitCreate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType);
    handleCancel();
  };

  const handleSubmitUpdate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType);
    handleCancel();
  };

  const handleDeleteClick = (itemId) => {
    onDelete(cvId, itemId, 'experiences');
  };

  return (
    <div className="animate-fade-in">
      {/* --- Header Section --- */}
      <div className="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom">
        <h4 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2">
            <Briefcase className="text-primary" size={20} /> 
            Professional Experience
        </h4>
        {!isCreating && !editingId && (
          <button 
            className="btn btn-primary btn-sm d-flex align-items-center gap-2" 
            onClick={handleAddNewClick}
          >
            <Plus size={16} /> Add Experience
          </button>
        )}
      </div>

      {/* --- Create Form --- */}
      {isCreating && (
        <div className="mb-4 p-4 bg-light rounded-xl border">
            <ExperienceForm
                key="new-experience-form"
                onSubmit={handleSubmitCreate}
                cvId={cvId}
                allSkills={allSkills}
                allAchievements={allAchievements}
                initialData={null}
                onCancelEdit={handleCancel}
            />
        </div>
      )}

      {/* --- Experience List --- */}
      <div className="d-flex flex-column gap-3">
        {experiences.length === 0 && !isCreating && (
            <div className="text-center py-5 text-muted bg-light rounded-xl border border-dashed">
                No experience entries yet. Click "Add Experience" to start.
            </div>
        )}
        
        {experiences.map(item => {
          if (item.id === editingId) {
            // --- EDIT MODE (In-Place Card) ---
            return (
              <div key={item.id} className="p-4 bg-light rounded-xl border shadow-sm">
                  <ExperienceForm
                    onSubmit={handleSubmitUpdate}
                    cvId={cvId}
                    allSkills={allSkills}
                    allAchievements={allAchievements}
                    initialData={item}
                    onCancelEdit={handleCancel}
                  />
              </div>
            );
          } else {
            // --- VIEW MODE (Professional Card) ---
            const linkedAchievements = getAchievements(item.achievement_ids);
            
            // Calculate aggregated skills
            const allIds = new Set(item.skill_ids || []);
            linkedAchievements.forEach(ach => (ach.skill_ids || []).forEach(id => allIds.add(id)));
            const aggregatedSkillIds = Array.from(allIds);

            return (
              <div key={item.id} className="bg-white p-4 rounded-xl border shadow-sm hover-lift transition-all">
                
                {/* Card Header: Title, Company, Location, Dates, Actions */}
                <div className="d-flex justify-content-between align-items-start mb-3">
                    <div>
                        <h5 className="fw-bold text-dark mb-1">{item.title || 'Untitled Experience'}</h5>
                        <div className="text-muted fw-medium d-flex align-items-center flex-wrap gap-2">
                            {item.company && <span>@{item.company}</span>}
                            {item.location && (
                                <span className="small text-secondary border-start ps-2">
                                    {item.location}
                                </span>
                            )}
                            {(item.start_date || item.end_date) && (
                                <span className="small text-secondary border-start ps-2">
                                     {/* Use helper here */}
                                    {formatDateRange(item.start_date, item.end_date)}
                                </span>
                            )}
                        </div>
                    </div>
                    <div className="d-flex gap-2">
                        <button 
                            onClick={() => handleEditClick(item.id)} 
                            className="btn btn-light btn-sm text-primary" 
                            title="Edit"
                        >
                            <Edit2 size={16} />
                        </button>
                        <button 
                            onClick={() => handleDeleteClick(item.id)} 
                            className="btn btn-light btn-sm text-danger" 
                            title="Delete"
                        >
                            <Trash2 size={16} />
                        </button>
                    </div>
                </div>

                {/* Description */}
                {item.description && (
                  <p className="text-secondary mb-3 small" style={{ whiteSpace: 'pre-wrap' }}>
                    {item.description}
                  </p>
                )}

                {/* Content Sections: Achievements & Skills */}
                <div className="d-flex flex-column gap-3">
                    {linkedAchievements.length > 0 && (
                        <div className="bg-light p-3 rounded-lg">
                            <h6 className="small fw-bold text-uppercase text-muted mb-2">Key Achievements</h6>
                            <AchievementDisplayGrid
                                achievementsToDisplay={linkedAchievements}
                                allSkills={allSkills}
                                isDisplayOnly={true}
                            />
                        </div>
                    )}

                    {aggregatedSkillIds.length > 0 && (
                        <div>
                            <SelectedSkillsDisplay
                                allSkills={allSkills}
                                selectedSkillIds={aggregatedSkillIds}
                                pendingSkills={[]}
                            />
                        </div>
                    )}
                </div>
              </div>
            );
          }
        })}
      </div>
    </div>
  );
};

export default ExperienceManager;