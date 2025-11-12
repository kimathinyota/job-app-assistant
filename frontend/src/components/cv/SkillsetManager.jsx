// frontend/src/components/cv/SkillsetManager.jsx
import React, { useState, useMemo } from 'react';
import { Plus, Search, Layers } from 'lucide-react';
import SkillForm from './SkillForm';
import SkillCard from './SkillCard';
import DeleteConfirmationModal from '../common/DeleteConfirmationModal';

const SkillsetManager = ({
  cvId,
  allSkills = [],
  allAchievements = [],
  allExperiences = [],
  allEducation = [],
  allProjects = [],
  allHobbies = [],
  onSubmit,
  onDelete,
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null);

  const skillRelationshipMap = useMemo(() => {
    const newMap = new Map();
    if (!allSkills.length) return newMap;
    allSkills.forEach(skill => {
      newMap.set(skill.id, {
        experiences: new Map(),
        education: new Map(),
        projects: new Map(),
        hobbies: new Map(),
        achievements: new Map(),
      });
    });
    const achievementMap = new Map(allAchievements.map(ach => [ach.id, ach]));
    const achievementSkillMap = new Map();
    allAchievements.forEach(ach => {
      achievementSkillMap.set(ach.id, ach.skill_ids || []);
    });

    const processItem = (item, type, title) => {
      (item.skill_ids || []).forEach(skillId => {
        if (newMap.has(skillId)) {
          const rels = newMap.get(skillId)[type];
          if (!rels.has(item.id)) rels.set(item.id, { item, title, via: [] });
          rels.get(item.id).direct = true;
        }
      });
      (item.achievement_ids || []).forEach(achId => {
        const linkedAch = achievementMap.get(achId);
        if (!linkedAch) return;
        const skillsFromAch = achievementSkillMap.get(achId) || [];
        skillsFromAch.forEach(skillId => {
          if (newMap.has(skillId)) {
            const rels = newMap.get(skillId)[type];
            if (!rels.has(item.id)) rels.set(item.id, { item, title, via: [] });
            rels.get(item.id).via.push(linkedAch);
          }
        });
      });
    };

    allExperiences.forEach(item => processItem(item, 'experiences', `${item.title} @ ${item.company}`));
    allEducation.forEach(item => processItem(item, 'education', `${item.degree} @ ${item.institution}`));
    allProjects.forEach(item => processItem(item, 'projects', item.title));
    allHobbies.forEach(item => processItem(item, 'hobbies', item.name));
    allAchievements.forEach(item => {
      (item.skill_ids || []).forEach(skillId => {
         if (newMap.has(skillId)) {
             const rels = newMap.get(skillId).achievements;
             if (!rels.has(item.id)) rels.set(item.id, { item, title: item.text, via: [], direct: true });
         }
      });
    });
    return newMap;
  }, [allSkills, allExperiences, allEducation, allProjects, allHobbies, allAchievements]);
  
  const handleAddNewClick = () => { setIsCreating(true); setEditingId(null); };
  const handleEditClick = (itemId) => { setEditingId(itemId); setIsCreating(false); };
  const handleCancel = () => { setIsCreating(false); setEditingId(null); };
  const handleSubmitCreate = async (cvId, data, itemType) => { await onSubmit(cvId, data, itemType); handleCancel(); };
  const handleSubmitUpdate = async (cvId, data, itemType) => { await onSubmit(cvId, data, itemType); handleCancel(); };

  const handleDeleteClick = (itemId) => {
    const skill = allSkills.find((s) => s.id === itemId);
    if (skill) {
      setItemToDelete(skill);
      setIsDeleteModalOpen(true);
    }
  };
  const handleCloseDeleteModal = () => { setItemToDelete(null); setIsDeleteModalOpen(false); };
  const handleConfirmDelete = () => {
    if (!itemToDelete) return;
    onDelete(cvId, itemToDelete.id, 'skills');
    handleCloseDeleteModal();
  };
  
  const filteredSkills = useMemo(() => {
      return allSkills.filter(skill => skill.name.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [allSkills, searchTerm]);
  
  const groupedSkills = useMemo(() => {
      const groups = { technical: [], soft: [], language: [], other: [] };
      filteredSkills.forEach(skill => {
          if (groups[skill.category]) groups[skill.category].push(skill);
          else groups.other.push(skill);
      });
      return groups;
  }, [filteredSkills]);
  
  const categoryOrder = ['technical', 'soft', 'language', 'other'];

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom">
        <h4 className="fw-bold text-dark mb-0 d-flex align-items-center gap-2">
            <Layers className="text-emerald-600" size={20} />
            Master Skillset
        </h4>
        {!isCreating && !editingId && (
          <button className="btn btn-primary btn-sm d-flex align-items-center gap-2" onClick={handleAddNewClick}>
            <Plus size={16} /> Add Skill
          </button>
        )}
      </div>

      {/* Create Form */}
      {isCreating && (
        <div className="mb-4 p-4 bg-light rounded-xl border">
            <SkillForm key="new-skill-form" onSubmit={handleSubmitCreate} cvId={cvId} initialData={null} onCancelEdit={handleCancel} />
        </div>
      )}
      
      {/* Search Bar */}
      {!isCreating && !editingId && (
          <div className="my-3 position-relative">
              <Search className="position-absolute text-muted" size={16} style={{ top: '10px', left: '12px' }} />
              <input 
                  type="text"
                  className="form-control ps-5 rounded-pill"
                  placeholder="Search skills by name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
              />
          </div>
      )}

      {/* Skills Display Grid */}
      <div className="mt-3">
          {categoryOrder.map(category => (
              groupedSkills[category].length > 0 && (
                  <div key={category} className="mb-4">
                      <h5 className="text-capitalize fw-bold text-secondary border-bottom pb-2 mb-3">
                          {category} Skills
                      </h5>
                      <div className="row g-3">
                          {groupedSkills[category].map(item => {
                              if (item.id === editingId) {
                                  // --- EDIT MODE ---
                                  return (
                                    <div className="col-12" key={item.id}>
                                      <div className="p-4 bg-light rounded-xl border">
                                          <SkillForm onSubmit={handleSubmitUpdate} cvId={cvId} initialData={item} onCancelEdit={handleCancel} />
                                      </div>
                                    </div>
                                  );
                              } else {
                                  // --- DISPLAY MODE ---
                                  return (
                                    <div className="col-12 col-lg-6" key={item.id}>
                                      <SkillCard
                                        skill={item}
                                        relationships={skillRelationshipMap.get(item.id)}
                                        onEdit={() => handleEditClick(item.id)}
                                        onDelete={() => handleDeleteClick(item.id)}
                                      />
                                    </div>
                                  );
                              }
                          })}
                      </div>
                  </div>
              )
          ))}
      </div>

      {/* Delete Modal */}
      {itemToDelete && (
        <DeleteConfirmationModal
          isOpen={isDeleteModalOpen}
          onClose={handleCloseDeleteModal}
          onConfirm={handleConfirmDelete}
          itemName={itemToDelete.name}
          itemType="skill"
        />
      )}
    </div>
  );
};

export default SkillsetManager;