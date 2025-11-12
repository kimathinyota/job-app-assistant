// frontend/src/components/cv/AchievementHub.jsx
import React, { useState, useMemo } from 'react';
import { Search, Award } from 'lucide-react';
import AchievementCard from './AchievementCard'; 

const AchievementHub = ({
  cvId,
  allSkills = [],
  allAchievements = [],
  allExperiences = [],
  allEducation = [],
  allProjects = [],
  allHobbies = [],
  onDelete,
  // onBack removed
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  // --- Relationship Map Logic (Unchanged) ---
  const achievementRelationshipMap = useMemo(() => {
    const newMap = new Map();
    if (!allAchievements.length) return newMap;

    allAchievements.forEach(ach => {
      newMap.set(ach.id, {
        experiences: [],
        education: [],
        projects: [],
        hobbies: [],
      });
    });

    const processItem = (item, type) => {
      (item.achievement_ids || []).forEach(achId => {
        if (newMap.has(achId)) {
          newMap.get(achId)[type].push(item);
        }
      });
    };

    allExperiences.forEach(item => processItem(item, 'experiences'));
    allEducation.forEach(item => processItem(item, 'education'));
    allProjects.forEach(item => processItem(item, 'projects'));
    allHobbies.forEach(item => processItem(item, 'hobbies'));

    return newMap;

  }, [allAchievements, allExperiences, allEducation, allProjects, allHobbies]);
  
  // --- Handlers ---
  const handleDeleteClick = (itemId) => {
    if (window.confirm("Are you sure you want to delete this master achievement? This cannot be undone.")) {
        onDelete(cvId, itemId, 'achievements');
    }
  };
  
  // --- Filtering ---
  const filteredAchievements = useMemo(() => {
      return allAchievements.filter(ach => 
          ach.text.toLowerCase().includes(searchTerm.toLowerCase())
      );
  }, [allAchievements, searchTerm]);

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-4 pb-2 border-bottom">
        <h4 className="fw-bold text-dark mb-1 d-flex align-items-center gap-2">
            <Award className="text-amber-500" size={20} />
            Master Achievement Hub
        </h4>
        <p className="text-muted small mb-0">
            A read-only overview of all achievements and their usage. Create achievements within Experiences or Projects.
        </p>
      </div>

      {/* Search Bar */}
      <div className="my-3 position-relative">
          <Search className="position-absolute text-muted" size={16} style={{ top: '10px', left: '12px' }} />
          <input 
              type="text"
              className="form-control ps-5 rounded-pill"
              placeholder="Search achievements by text..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
          />
      </div>

      {/* Achievements Display Grid */}
      <div className="mt-3">
          {filteredAchievements.length === 0 ? (
              <p className="text-muted fst-italic text-center py-5">No achievements found matching your search.</p>
          ) : (
              <div className="row g-3">
                  {filteredAchievements.map(item => (
                    <div className="col-12 col-lg-6" key={item.id}>
                      <AchievementCard
                        achievement={item}
                        allSkills={allSkills}
                        relationships={achievementRelationshipMap.get(item.id)}
                        onDelete={() => handleDeleteClick(item.id)}
                      />
                    </div>
                  ))}
              </div>
          )}
      </div>
    </div>
  );
};

export default AchievementHub;