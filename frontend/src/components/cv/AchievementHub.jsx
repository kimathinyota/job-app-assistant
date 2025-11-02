// frontend/src/components/cv/AchievementHub.jsx
import React, { useState, useMemo } from 'react';
import AchievementCard from './AchievementCard'; // We will create this

// This is the new component that encapsulates all logic for Master Achievements
const AchievementHub = ({
  cvId,
  // We need ALL CV data to build the relationship map
  allSkills = [],
  allAchievements = [],
  allExperiences = [],
  allEducation = [],
  allProjects = [],
  allHobbies = [],
  onDelete, // This will be `handleDeleteNested` from the parent
  onBack    // This will be `() => setActiveSection(null)`
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  // --- Relationship Map Logic ---
  // This builds a map of all achievement relationships
  const achievementRelationshipMap = useMemo(() => {
    const newMap = new Map();
    if (!allAchievements.length) return newMap;

    // 1. Initialize map for all achievements
    allAchievements.forEach(ach => {
      newMap.set(ach.id, {
        experiences: [],
        education: [],
        projects: [],
        hobbies: [],
      });
    });

    // 2. Helper function to process items
    const processItem = (item, type) => {
      (item.achievement_ids || []).forEach(achId => {
        if (newMap.has(achId)) {
          newMap.get(achId)[type].push(item);
        }
      });
    };

    // 3. Run all items through the processor
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
  
  // --- Filtering and Grouping ---
  const filteredAchievements = useMemo(() => {
      return allAchievements.filter(ach => 
          ach.text.toLowerCase().includes(searchTerm.toLowerCase())
      );
  }, [allAchievements, searchTerm]);

  return (
    <div>
      <button onClick={onBack} className="btn btn-secondary mb-3">
        &larr; Back to CV Dashboard
      </button>

      <h3 className="h4 border-bottom pb-2 text-capitalize">
        Master Achievement Hub
      </h3>
      <p className="text-muted">
        A read-only overview of all achievements and where they are used in your CV.
      </p>

      {/* Search Bar */}
      <div className="my-3">
          <input 
              type="text"
              className="form-control"
              placeholder="Search achievements by text..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
          />
      </div>

      {/* Achievements Display */}
      <div className="mt-3">
          {filteredAchievements.length === 0 ? (
              <p className="text-muted fst-italic">No achievements found matching your search.</p>
          ) : (
              <div className="row g-3">
                  {filteredAchievements.map(item => (
                    <div className="col-12 col-lg-6" key={item.id}>
                      <AchievementCard
                        achievement={item}
                        allSkills={allSkills} // Pass skills to display on the card
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