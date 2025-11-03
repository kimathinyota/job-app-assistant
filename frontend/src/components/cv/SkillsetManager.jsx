// frontend/src/components/cv/SkillsetManager.jsx
import React, { useState, useMemo } from 'react';
import SkillForm from './SkillForm';
import SkillCard from './SkillCard'; // We will create this
import DeleteConfirmationModal from '../common/DeleteConfirmationModal'; // <-- 1. Import modal
// This is the new component that encapsulates all logic for Master Skills
const SkillsetManager = ({
  cvId,
  // We need ALL CV data to build the relationship map
  allSkills = [],
  allAchievements = [],
  allExperiences = [],
  allEducation = [],
  allProjects = [],
  allHobbies = [],
  onSubmit, // This will be `handleAddOrUpdateNestedItem` from the parent
  onDelete, // This will be `handleDeleteNested` from the parent
  onBack    // This will be `() => setActiveSection(null)`
}) => {
  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  // --- 2. Add state for delete modal ---
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [itemToDelete, setItemToDelete] = useState(null);

  // --- Relationship Map Logic ---
  // This powerful 'useMemo' hook builds a map of all skill relationships
  // whenever the CV data changes.
  const skillRelationshipMap = useMemo(() => {
    const newMap = new Map();
    if (!allSkills.length) return newMap;

    // 1. Initialize map for all skills
    allSkills.forEach(skill => {
      newMap.set(skill.id, {
        experiences: new Map(), // Use a Map for de-duplication <itemId, { item, title, via: [ach] }>
        education: new Map(),
        projects: new Map(),
        hobbies: new Map(),
        achievements: new Map(), // For skills linked *directly* to an achievement
      });
    });

    // 2. Create helper maps for fast lookups
    const achievementMap = new Map(allAchievements.map(ach => [ach.id, ach]));
    const achievementSkillMap = new Map(); // Map<achId, skillId[]>
    allAchievements.forEach(ach => {
      achievementSkillMap.set(ach.id, ach.skill_ids || []);
    });

    // 3. Generic function to process any item (Experience, Project, etc.)
    const processItem = (item, type, title) => {
      // A. Check for direct skill links
      (item.skill_ids || []).forEach(skillId => {
        if (newMap.has(skillId)) {
          const rels = newMap.get(skillId)[type]; // e.g., newMap.get(skillId).experiences
          if (!rels.has(item.id)) {
            rels.set(item.id, { item, title, via: [] });
          }
          rels.get(item.id).direct = true;
        }
      });
      
      // B. Check for indirect links via achievements
      (item.achievement_ids || []).forEach(achId => {
        const linkedAch = achievementMap.get(achId);
        if (!linkedAch) return;

        const skillsFromAch = achievementSkillMap.get(achId) || [];
        skillsFromAch.forEach(skillId => {
          if (newMap.has(skillId)) {
            const rels = newMap.get(skillId)[type];
            if (!rels.has(item.id)) {
              rels.set(item.id, { item, title, via: [] });
            }
            rels.get(item.id).via.push(linkedAch);
          }
        });
      });
    };

    // 4. Run all items through the processor
    allExperiences.forEach(item => processItem(item, 'experiences', `${item.title} @ ${item.company}`));
    allEducation.forEach(item => processItem(item, 'education', `${item.degree} @ ${item.institution}`));
    allProjects.forEach(item => processItem(item, 'projects', item.title));
    allHobbies.forEach(item => processItem(item, 'hobbies', item.name));
    
    // 5. Process direct links from skills to achievements
    allAchievements.forEach(item => {
      (item.skill_ids || []).forEach(skillId => {
         if (newMap.has(skillId)) {
             const rels = newMap.get(skillId).achievements;
             if (!rels.has(item.id)) {
                  rels.set(item.id, { item, title: item.text, via: [], direct: true });
             }
         }
      });
    });

    return newMap;

  }, [allSkills, allExperiences, allEducation, allProjects, allHobbies, allAchievements]);
  
  // --- Handlers ---
  const handleAddNewClick = () => {
    setIsCreating(true);
    setEditingId(null);
  };

  const handleEditClick = (itemId) => {
    setEditingId(itemId);
    setIsCreating(false);
  };

  const handleCancel = () => {
    setIsCreating(false);
    setEditingId(null);
  };

  const handleSubmitCreate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType);
    handleCancel();
  };

  const handleSubmitUpdate = async (cvId, data, itemType) => {
    await onSubmit(cvId, data, itemType);
    handleCancel();
  };

  // --- 3. Modify handleDeleteClick ---
  const handleDeleteClick = (itemId) => {
    const skill = allSkills.find((s) => s.id === itemId);
    if (skill) {
      setItemToDelete(skill);
      setIsDeleteModalOpen(true);
    }
    // The original logic is moved to handleConfirmDelete
    // onDelete(cvId, itemId, 'skills');
  };

  // --- 4. Add new handlers for modal ---
  const handleCloseDeleteModal = () => {
    setItemToDelete(null);
    setIsDeleteModalOpen(false);
  };

  const handleConfirmDelete = () => {
    if (!itemToDelete) return;
    // This calls the original parent function
    onDelete(cvId, itemToDelete.id, 'skills');
    handleCloseDeleteModal();
  };
  
  // --- Filtering and Grouping ---
  const filteredSkills = useMemo(() => {
      return allSkills.filter(skill => 
          skill.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
  }, [allSkills, searchTerm]);
  
  const groupedSkills = useMemo(() => {
      const groups = {
          technical: [],
          soft: [],
          language: [],
          other: []
      };
      filteredSkills.forEach(skill => {
          if (groups[skill.category]) {
              groups[skill.category].push(skill);
          } else {
              groups.other.push(skill);
          }
      });
      return groups;
  }, [filteredSkills]);
  
  const categoryOrder = ['technical', 'soft', 'language', 'other'];

  return (
    <div>
      <button onClick={onBack} className="btn btn-secondary mb-3">
        &larr; Back to CV Dashboard
      </button>

      <h3 className="h4 border-bottom pb-2 text-capitalize">
        Master Skillset
      </h3>

      {/* "Add New" Button */}
      {!isCreating && !editingId && (
        <button
          className="btn btn-primary my-3"
          onClick={handleAddNewClick}
        >
          + Add New Master Skill
        </button>
      )}

      {/* "Create New" Form */}
      {isCreating && (
        <SkillForm
          key="new-skill-form"
          onSubmit={handleSubmitCreate}
          cvId={cvId}
          initialData={null}
          onCancelEdit={handleCancel}
        />
      )}
      
      {/* Search Bar */}
      {!isCreating && !editingId && (
          <div className="my-3">
              <input 
                  type="text"
                  className="form-control"
                  placeholder="Search skills by name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
              />
          </div>
      )}

      {/* Skills Display */}
      <div className="mt-3">
          {categoryOrder.map(category => (
              groupedSkills[category].length > 0 && (
                  <div key={category} className="mb-4">
                      <h4 className="text-capitalize text-muted border-bottom pb-2">
                          {category} Skills
                      </h4>
                      <div className="row g-3">
                          {groupedSkills[category].map(item => {
                              if (item.id === editingId) {
                                  // --- RENDER EDIT FORM ---
                                  return (
                                    <div className="col-12" key={item.id}>
                                      <SkillForm
                                        onSubmit={handleSubmitUpdate}
                                        cvId={cvId}
                                        initialData={item}
                                        onCancelEdit={handleCancel}
                                      />
                                    </div>
                                  );
                              } else {
                                  // --- RENDER DISPLAY CARD ---
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

      {/* --- 5. Render the delete modal --- */}
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