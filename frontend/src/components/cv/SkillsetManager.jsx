import React, { useState, useMemo } from 'react';
import { Plus, Search, Layers, ChevronDown, ChevronRight, Hash } from 'lucide-react';
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

  // State for collapsible categories. Default: all open
  const [collapsedCategories, setCollapsedCategories] = useState({});

  // 1. Build Relationship Map (Memoized)
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
  
  // Handlers
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

  const toggleCategory = (category) => {
    setCollapsedCategories(prev => ({
        ...prev,
        [category]: !prev[category]
    }));
  };
  
  const filteredSkills = useMemo(() => {
      return allSkills.filter(skill => skill.name.toLowerCase().includes(searchTerm.toLowerCase()));
  }, [allSkills, searchTerm]);
  
  // 2. Dynamic Grouping Logic
  const groupedSkills = useMemo(() => {
      const groups = {};
      
      filteredSkills.forEach(skill => {
          // Normalize category: Lowercase for sorting, but keep display friendly if needed.
          // For now, we trust the input, but fallback to 'Uncategorized' if empty.
          const cat = skill.category ? skill.category.trim() : 'Uncategorized';
          
          if (!groups[cat]) {
              groups[cat] = [];
          }
          groups[cat].push(skill);
      });
      return groups;
  }, [filteredSkills]);
  
  // 3. Smart Sorting of Categories
  const sortedCategories = useMemo(() => {
      const cats = Object.keys(groupedSkills);
      
      // Define priority order for standard categories
      const priority = ['technical', 'soft', 'language'];
      
      return cats.sort((a, b) => {
          const aLower = a.toLowerCase();
          const bLower = b.toLowerCase();
          
          const aIdx = priority.indexOf(aLower);
          const bIdx = priority.indexOf(bLower);
          
          // If both are priority, sort by priority index
          if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
          
          // If one is priority, it goes first
          if (aIdx !== -1) return -1;
          if (bIdx !== -1) return 1;
          
          // Push "other" or "uncategorized" to the very bottom
          if (aLower === 'other' || aLower === 'uncategorized') return 1;
          if (bLower === 'other' || bLower === 'uncategorized') return -1;
          
          // Otherwise, alphabetical
          return aLower.localeCompare(bLower);
      });
  }, [groupedSkills]);

  // --- NEW: Calculate All Available Categories (Defaults + Custom) ---
  // This aggregates every category currently used in 'allSkills' plus the defaults.
  const availableCategories = useMemo(() => {
      const defaultCats = ['technical', 'soft', 'language', 'other'];
      const usedCats = allSkills.map(s => s.category).filter(Boolean);
      // Combine and deduplicate
      return [...new Set([...defaultCats, ...usedCats])].sort();
  }, [allSkills]);

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
            <SkillForm 
                key="new-skill-form" 
                onSubmit={handleSubmitCreate} 
                cvId={cvId} 
                initialData={null} 
                onCancelEdit={handleCancel}
                // --- PASS AVAILABLE CATEGORIES HERE ---
                existingCategories={availableCategories} 
            />
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

      {/* Skills Display Grid (Dynamic Categories) */}
      <div className="mt-3">
          {sortedCategories.map(category => {
              const skills = groupedSkills[category];
              const isCollapsed = collapsedCategories[category];
              const count = skills.length;

              return (
                  <div key={category} className="mb-3">
                      {/* Category Header */}
                      <div 
                        className="d-flex align-items-center justify-content-between p-3 bg-white rounded border shadow-sm cursor-pointer hover:bg-slate-50 transition-all"
                        onClick={() => toggleCategory(category)}
                      >
                          <div className="d-flex align-items-center gap-2">
                              {/* Icon logic: Hash for custom, Layers for general */}
                              <div className={`p-1 rounded ${['technical', 'soft', 'language'].includes(category.toLowerCase()) ? 'bg-emerald-100 text-emerald-600' : 'bg-gray-100 text-gray-500'}`}>
                                  <Hash size={16} />
                              </div>
                              <h5 className="mb-0 text-capitalize fw-bold text-dark">
                                  {category}
                              </h5>
                              <span className="badge bg-light text-secondary border rounded-pill ms-2">
                                  {count}
                              </span>
                          </div>
                          
                          <div className="text-muted">
                              {isCollapsed ? <ChevronRight size={20} /> : <ChevronDown size={20} />}
                          </div>
                      </div>

                      {/* Skills Grid (Collapsible) */}
                      {!isCollapsed && (
                          <div className="row g-3 mt-1 ps-2 animate-fade-in">
                              {skills.map(item => {
                                  if (item.id === editingId) {
                                      // --- EDIT MODE ---
                                      return (
                                        <div className="col-12" key={item.id}>
                                          <div className="p-4 bg-light rounded-xl border">
                                              <SkillForm 
                                                  onSubmit={handleSubmitUpdate} 
                                                  cvId={cvId} 
                                                  initialData={item} 
                                                  onCancelEdit={handleCancel}
                                                  // --- PASS AVAILABLE CATEGORIES HERE TOO ---
                                                  existingCategories={availableCategories} 
                                              />
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
                      )}
                  </div>
              );
          })}
          
          {sortedCategories.length === 0 && (
             <div className="text-center py-5 text-muted">
                 <p>No skills found. Click "Add Skill" to create one.</p>
             </div>
          )}
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