// frontend/src/components/cv/ProjectManager.jsx
import React, { useState } from 'react';
import { Plus, Edit2, Trash2, Cpu, Link } from 'lucide-react';
import ProjectForm from './ProjectForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

const ProjectManager = ({
  cvId,
  projects = [],
  allExperiences = [],
  allEducation = [],
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
      .filter(Boolean);
  };
  
  const getRelatedExperienceName = (id) => {
    if (!id) return null;
    const exp = allExperiences.find(e => e.id === id);
    return exp ? `${exp.title} @ ${exp.company}` : 'Unknown Experience';
  };
  
  const getRelatedEducationName = (id) => {
    if (!id) return null;
    const edu = allEducation.find(e => e.id === id);
    return edu ? `${edu.degree} @ ${edu.institution}` : 'Unknown Education';
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
    onDelete(cvId, itemId, 'projects');
  };

  return (
    <div className="animate-fade-in">
      {/* Header Section */}
      <div className="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom">
        <h4 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2">
            <Cpu className="text-purple-600" size={20} />
            Projects
        </h4>
        {!isCreating && !editingId && (
          <button className="btn btn-primary btn-sm d-flex align-items-center gap-2" onClick={handleAddNewClick}>
            <Plus size={16} /> Add Project
          </button>
        )}
      </div>

      {/* Create Form */}
      {isCreating && (
        <div className="mb-4 p-4 bg-light rounded-xl border">
            <ProjectForm
                key="new-project-form"
                onSubmit={handleSubmitCreate}
                cvId={cvId}
                allExperiences={allExperiences}
                allEducation={allEducation}
                allSkills={allSkills}
                allAchievements={allAchievements}
                initialData={null}
                onCancelEdit={handleCancel}
            />
        </div>
      )}

      {/* Projects List */}
      <div className="d-flex flex-column gap-3">
        {projects.length === 0 && !isCreating && (
            <div className="text-center py-5 text-muted bg-light rounded-xl border border-dashed">
                No projects added yet. Click "Add Project" to start.
            </div>
        )}

        {projects.map(item => {
          if (item.id === editingId) {
            // --- EDIT MODE ---
            return (
              <div key={item.id} className="p-4 bg-light rounded-xl border shadow-sm">
                  <ProjectForm
                    key={item.id}
                    onSubmit={handleSubmitUpdate}
                    cvId={cvId}
                    allExperiences={allExperiences}
                    allEducation={allEducation}
                    allSkills={allSkills}
                    allAchievements={allAchievements}
                    initialData={item}
                    onCancelEdit={handleCancel}
                  />
              </div>
            );
          } else {
            // --- DISPLAY MODE ---
            const linkedAchievements = getAchievements(item.achievement_ids);
            const allIds = new Set(item.skill_ids || []);
            linkedAchievements.forEach(ach => (ach.skill_ids || []).forEach(id => allIds.add(id)));
            const aggregatedSkillIds = Array.from(allIds);
            
            const relatedExpName = getRelatedExperienceName(item.related_experience_id);
            const relatedEduName = getRelatedEducationName(item.related_education_id);

            return (
              <div key={item.id} className="bg-white p-4 rounded-xl border shadow-sm hover-lift transition-all">
                
                {/* Card Header */}
                <div className="d-flex justify-content-between align-items-start mb-2">
                    <div>
                        <h5 className="fw-bold text-dark mb-1">{item.title || 'Untitled Project'}</h5>
                        
                        {/* Relationship Badges */}
                        <div className="d-flex flex-wrap gap-2 mt-2">
                            {relatedExpName && (
                                <span className="badge bg-secondary bg-opacity-10 text-secondary border d-flex align-items-center gap-1 fw-normal">
                                    <Link size={12}/> {relatedExpName}
                                </span>
                            )}
                            {relatedEduName && (
                                <span className="badge bg-info bg-opacity-10 text-info border d-flex align-items-center gap-1 fw-normal">
                                    <Link size={12}/> {relatedEduName}
                                </span>
                            )}
                        </div>
                    </div>
                    <div className="d-flex gap-2">
                        <button onClick={() => handleEditClick(item.id)} className="btn btn-light btn-sm text-primary" title="Edit">
                            <Edit2 size={16} />
                        </button>
                        <button onClick={() => handleDeleteClick(item.id)} className="btn btn-light btn-sm text-danger" title="Delete">
                            <Trash2 size={16} />
                        </button>
                    </div>
                </div>

                {/* Description */}
                {item.description && (
                  <p className="text-secondary mb-3 small mt-3" style={{ whiteSpace: 'pre-wrap' }}>
                    {item.description}
                  </p>
                )}

                {/* Content Sections */}
                <div className="d-flex flex-column gap-3">
                    {linkedAchievements.length > 0 && (
                        <div className="bg-light p-3 rounded-lg">
                            <h6 className="small fw-bold text-uppercase text-muted mb-2">Achievements</h6>
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

export default ProjectManager;