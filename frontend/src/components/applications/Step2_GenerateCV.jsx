// frontend/src/components/applications/Step2_GenerateCV.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { generateCvPrompt } from '../../api/applicationClient';
import PromptModal from './PromptModal';
import CVItemDisplayCard from './CVItemDisplayCard'; // <-- 1. Import the display card
import PromoteItemModal from './PromoteItemModal'; // <-- 1. IMPORT NEW MODAL

// --- 2. CREATE THE "GHOST" COMPONENT ---
const GhostItemCard = ({ item, itemType, onPromote }) => {
  // Get the simple title for the item
  let itemTitle = item.title || item.name || item.degree || 'Unknown Item';
  if (itemType === 'experiences') itemTitle = `${item.title} @ ${item.company}`;
  if (itemType === 'education') itemTitle = `${item.degree} @ ${item.institution}`;

  return (
    <div 
        className="card card-body p-2 mb-2 border-dashed" 
        style={{ opacity: 0.6 }}
    >
        <div className="d-flex justify-content-between align-items-center">
            <span className="fst-italic">{itemTitle}</span>
            <button 
                type="button" 
                className="btn btn-sm btn-outline-primary"
                onClick={() => onPromote(item, itemType)}
            >
                + Promote to CV
            </button>
        </div>
    </div>
  );
};

const Step2_GenerateCV = ({ job, cv, mapping, onPrev, onNext, onMappingChanged }) => {
    const [cvPromptJson, setCvPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

// --- 4. ADD STATE FOR THE PROMOTE MODAL ---
    const [isPromoteModalOpen, setIsPromoteModalOpen] = useState(false);
    const [itemToPromote, setItemToPromote] = useState(null); // { item: {...}, type: '...' }

    // --- 5. UPDATE useMemo TO INCLUDE *UNMAPPED* ITEMS ---
    const {
        mappedExperiences,
        mappedProjects,
        mappedEducation,
        mappedHobbies,
        unmappedExperiences,
        unmappedProjects,
        unmappedEducation,
        unmappedHobbies,
        aggregatedSkillIds, 
        groupedSkills
    } = useMemo(() => {
        const mappedItemIds = new Set(
            mapping.pairs.map(p => p.context_item_id || p.experience_id)
        );

        // Mapped items
        const mappedExperiences = cv.experiences.filter(exp => mappedItemIds.has(exp.id));
        const mappedProjects = cv.projects.filter(proj => mappedItemIds.has(proj.id));
        const mappedEducation = cv.education.filter(edu => mappedItemIds.has(edu.id));
        const mappedHobbies = cv.hobbies.filter(hob => mappedItemIds.has(hob.id));
        
        // Unmapped items
        const unmappedExperiences = cv.experiences.filter(exp => !mappedItemIds.has(exp.id));
        const unmappedProjects = cv.projects.filter(proj => !mappedItemIds.has(proj.id));
        const unmappedEducation = cv.education.filter(edu => !mappedItemIds.has(edu.id));
        const unmappedHobbies = cv.hobbies.filter(hob => !mappedItemIds.has(hob.id));

        // Skill logic (unchanged from before)
        let skillIds = new Set();
        const allMappedItems = [...mappedExperiences, ...mappedProjects, ...mappedEducation, ...mappedHobbies];
        
        allMappedItems.forEach(item => {
            (item.skill_ids || []).forEach(id => skillIds.add(id));
            (item.achievement_ids || []).forEach(achId => {
                const ach = cv.achievements.find(a => a.id === achId);
                if (ach) {
                    (ach.skill_ids || []).forEach(id => skillIds.add(id));
                }
            });
        });

        const groupedSkills = { technical: [], soft: [], language: [], other: [] };
        cv.skills.forEach(skill => {
            (groupedSkills[skill.category] || groupedSkills.other).push(skill);
        });
        
        return { 
            mappedExperiences, mappedProjects, mappedEducation, mappedHobbies, 
            unmappedExperiences, unmappedProjects, unmappedEducation, unmappedHobbies,
            aggregatedSkillIds: skillIds,
            groupedSkills
        };
    }, [cv, mapping]); // <-- DEPEND ON MAPPING (so it recalculates on change)

    // --- 3. STATE TO MANAGE SKILL SELECTION ---
    // This holds the set of skill IDs that will be sent to the AI.
    // It's initialized by the aggregatedSkillIds from the mapping.
    const [selectedSkillIds, setSelectedSkillIds] = useState(new Set(aggregatedSkillIds));

    // Update state if the mapping changes
    useEffect(() => {
        setSelectedSkillIds(new Set(aggregatedSkillIds));
    }, [aggregatedSkillIds]);

    const handleToggleSkill = (skillId) => {
        setSelectedSkillIds(prev => {
            const newSet = new Set(prev);
            if (newSet.has(skillId)) {
                newSet.delete(skillId);
            } else {
                newSet.add(skillId);
            }
            return newSet;
        });
    };
    
    // --- 4. UPDATE API CALL ---
    const handleGeneratePrompt = async () => {
        setIsLoading(true);
        try {
            // Pass the array of selected skill IDs to the API
            const skillIdArray = Array.from(selectedSkillIds);
            const res = await generateCvPrompt(cv.id, job.id, skillIdArray); // <-- Pass skills
            
            setCvPromptJson(JSON.stringify(res.data, null, 2));
            setIsModalOpen(true);
        } catch (err) {
            alert("Failed to generate prompt.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    // --- 6. ADD HANDLERS FOR THE PROMOTE MODAL ---
    const handleOpenPromoteModal = (item, itemType) => {
        setItemToPromote({ item: item, type: itemType });
        setIsPromoteModalOpen(true);
    };

    const handleClosePromoteModal = () => {
        setItemToPromote(null);
        setIsPromoteModalOpen(false);
    };

    // --- 5. RENDER THE FULL CV PREVIEW ---
    return (
        <div>
            <h4 className="h5">Step 2: Review Your Tailored CV</h4>
            <p className="text-muted">
                This is a preview of the content that will be sent to the AI.
                You can "Promote" unmapped items to include them.
            </p>
            
            <div className="border p-3 p-md-4 rounded bg-light" style={{ maxHeight: '60vh', overflowY: 'auto' }}>
                
                {/* Header */}
                <div className="text-center border-bottom pb-3 mb-3">
                    <h3 className="h4 m-0">{cv.name}</h3>
                    {cv.summary && (
                        <p className="m-0 mt-1 fst-italic text-muted" style={{whiteSpace: 'pre-wrap'}}>
                            {cv.summary}
                        </p>
                    )}
                </div>

                {/* Mapped Experiences */}
                {(mappedExperiences.length > 0 || unmappedExperiences.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Professional Experience</h5>
                        {mappedExperiences.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="experiences" allSkills={cv.skills} allAchievements={cv.achievements} />
                        ))}
                        {/* --- GHOST EXPERIENCES --- */}
                        {unmappedExperiences.length > 0 && (
                            <div className="mt-3">
                                <h6 className="text-muted small">Available (Not Mapped):</h6>
                                {unmappedExperiences.map(item => (
                                    <GhostItemCard key={item.id} item={item} itemType="experiences" onPromote={handleOpenPromoteModal} />
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Mapped Education */}
                {(mappedEducation.length > 0 || unmappedEducation.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Education</h5>
                        {mappedEducation.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="education" allSkills={cv.skills} allAchievements={cv.achievements} />
                        ))}
                        {/* --- GHOST EDUCATION --- */}
                        {unmappedEducation.length > 0 && (
                            <div className="mt-3">
                                <h6 className="text-muted small">Available (Not Mapped):</h6>
                                {unmappedEducation.map(item => (
                                    <GhostItemCard key={item.id} item={item} itemType="education" onPromote={handleOpenPromoteModal} />
                                ))}
                            </div>
                        )}
                    </div>
                )}
                
                {/* Mapped Projects */}
                {(mappedProjects.length > 0 || unmappedProjects.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Projects</h5>
                        {mappedProjects.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="projects" allSkills={cv.skills} allAchievements={cv.achievements} allExperiences={cv.experiences} allEducation={cv.education} />
                        ))}
                        {/* --- GHOST PROJECTS --- */}
                        {unmappedProjects.length > 0 && (
                            <div className="mt-3">
                                <h6 className="text-muted small">Available (Not Mapped):</h6>
                                {unmappedProjects.map(item => (
                                    <GhostItemCard key={item.id} item={item} itemType="projects" onPromote={handleOpenPromoteModal} />
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Mapped Hobbies */}
                {(mappedHobbies.length > 0 || unmappedHobbies.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Hobbies & Interests</h5>
                        {mappedHobbies.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="hobbies" allSkills={cv.skills} allAchievements={cv.achievements} />
                        ))}
                        {/* --- GHOST HOBBIES --- */}
                        {unmappedHobbies.length > 0 && (
                            <div className="mt-3">
                                <h6 className="text-muted small">Available (Not Mapped):</h6>
                                {unmappedHobbies.map(item => (
                                    <GhostItemCard key={item.id} item={item} itemType="hobbies" onPromote={handleOpenPromoteModal} />
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* --- Interactive Skills Section --- */}
                <div className="mb-4">
                    <h5 className="text-primary border-bottom pb-1">Skills</h5>
                    <p className="form-text mt-0 mb-2">
                        Skills highlighted in green are automatically selected based on your mapping.
                        Click any skill to include or exclude it from the prompt.
                    </p>
                    
                    {['technical', 'soft', 'language', 'other'].map(category => (
                        groupedSkills[category].length > 0 && (
                            <div key={category} className="mb-3">
                                <strong className="text-capitalize d-block mb-2">{category}</strong>
                                <div className="d-flex flex-wrap gap-2">
                                    {groupedSkills[category].map(skill => {
                                        const isSelected = selectedSkillIds.has(skill.id);
                                        const isAutoSelected = aggregatedSkillIds.has(skill.id);
                                        
                                        return (
                                            <React.Fragment key={skill.id}>
                                                <input
                                                    type="checkbox"
                                                    className="btn-check"
                                                    id={`skill-check-${skill.id}`}
                                                    checked={isSelected}
                                                    onChange={() => handleToggleSkill(skill.id)}
                                                    autoComplete="off"
                                                />
                                                <label 
                                                    className={`btn btn-sm ${isSelected ? (isAutoSelected ? 'btn-success' : 'btn-primary') : 'btn-outline-secondary'}`}
                                                    htmlFor={`skill-check-${skill.id}`}
                                                >
                                                    {skill.name}
                                                </label>
                                            </React.Fragment>
                                        );
                                    })}
                                </div>
                            </div>
                        )
                    ))}
                </div>
            </div>

            <button 
                className="btn btn-info mt-3" 
                onClick={handleGeneratePrompt}
                disabled={isLoading}
            >
                {isLoading ? "Generating..." : "Generate CV Prompt"}
            </button>

            <PromptModal
                isOpen={isModalOpen}
                jsonString={cvPromptJson}
                onClose={() => setIsModalOpen(false)}
            />

            {/* --- 8. RENDER THE PROMOTE MODAL --- */}
            <PromoteItemModal
                isOpen={isPromoteModalOpen}
                onClose={handleClosePromoteModal}
                job={job}
                mapping={mapping}
                itemToPromote={itemToPromote}
                onMappingChanged={onMappingChanged}
            />
            
            <div className="d-flex justify-content-between mt-4">
                <button className="btn btn-secondary" onClick={onPrev}>
                    &lt; Back: Mapping
                </button>
                <button className="btn btn-primary" onClick={onNext}>
                    Next: Build Cover Letter &gt;
                </button>
            </div>
        </div>
    );
};

export default Step2_GenerateCV;