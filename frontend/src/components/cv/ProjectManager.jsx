// frontend/src/components/cv/ProjectManager.jsx
import React, { useState } from 'react';
import { Plus, Edit2, Trash2, Cpu, Briefcase, BookOpen, Smile } from 'lucide-react';
import ProjectForm from './ProjectForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

const ProjectManager = ({
  cvId,
  projects = [],
  allExperiences = [],
  allEducation = [],
  allHobbies = [], 
  allSkills = [],
  allAchievements = [],
  onSubmit,
  onDelete,
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);

  // --- Helper to resolve mixed (singular/plural) references ---
  const resolveReferences = (itemIds, singularId, sourceList) => {
    const uniqueIds = new Set(itemIds || []);
    // Backward compatibility: if no list items, check the old singular ID
    if (singularId && uniqueIds.size === 0) {
        uniqueIds.add(singularId); 
    }
    return Array.from(uniqueIds).map(id => sourceList.find(i => i.id === id)).filter(Boolean);
  };

  const getAchievements = (achievementIds = []) => {
    if (!achievementIds || achievementIds.length === 0) return [];
    return achievementIds
      .map(id => allAchievements.find(a => a.id === id))
      .filter(Boolean);
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
                allHobbies={allHobbies}
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
                    allHobbies={allHobbies}
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
            
            // Calculate Aggregate Skills
            const allIds = new Set(item.skill_ids || []);
            linkedAchievements.forEach(ach => (ach.skill_ids || []).forEach(id => allIds.add(id)));
            const aggregatedSkillIds = Array.from(allIds);
            
            // Resolve Contexts
            const linkedExperiences = resolveReferences(item.related_experience_ids, item.related_experience_id, allExperiences);
            const linkedEducation = resolveReferences(item.related_education_ids, item.related_education_id, allEducation);
            const linkedHobbies = resolveReferences(item.related_hobby_ids, null, allHobbies);

            return (
              <div key={item.id} className="bg-white p-4 rounded-xl border shadow-sm hover-lift transition-all">
                
                {/* Card Header */}
                <div className="d-flex justify-content-between align-items-start mb-2">
                    <div>
                        <h5 className="fw-bold text-dark mb-1">{item.title || 'Untitled Project'}</h5>
                        
                        {/* Context Badges */}
                        <div className="d-flex flex-wrap gap-2 mt-2">
                            {linkedExperiences.map(exp => (
                                <span key={exp.id} className="rounded-pill bg-blue-50 text-blue-700 border border-blue-200 small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                    <Briefcase size={12} /> 
                                    {exp.title}{exp.company ? ` @ ${exp.company}` : ''}
                                </span>
                            ))}
                            {linkedEducation.map(edu => (
                                <span key={edu.id} className="rounded-pill bg-indigo-50 text-indigo-700 border border-indigo-200 small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                    <BookOpen size={12} /> 
                                    {edu.degree}{edu.institution ? ` @ ${edu.institution}` : ''}
                                </span>
                            ))}
                            {linkedHobbies.map(hobby => (
                                <span key={hobby.id} className="rounded-pill bg-pink-50 text-pink-600 border border-pink-200 small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                    <Smile size={12} /> {hobby.name}
                                </span>
                            ))}
                        </div>
                    </div>
                    
                    {/* Action Buttons */}
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