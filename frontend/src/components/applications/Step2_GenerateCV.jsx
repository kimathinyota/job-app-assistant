// frontend/src/components/applications/Step2_GenerateCV.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { generateCvPrompt } from '../../api/applicationClient';
import PromptModal from './PromptModal';
import CVItemDisplayCard from './CVItemDisplayCard'; // <-- 1. Import the display card

const Step2_GenerateCV = ({ job, cv, mapping, onPrev, onNext }) => {
    const [cvPromptJson, setCvPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(false);

    // --- 2. LOGIC TO FIND MAPPED ITEMS & SKILLS ---
    const {
        mappedExperiences,
        mappedProjects,
        mappedEducation,
        mappedHobbies,
        aggregatedSkillIds, // This is the set of *all* skills from mapped items
        groupedSkills // This is *all* skills, grouped by category
    } = useMemo(() => {
        const mappedExpIds = new Set();
        const mappedProjIds = new Set();
        const mappedEduIds = new Set();
        const mappedHobbyIds = new Set();

        mapping.pairs.forEach(p => {
            const id = p.context_item_id || p.experience_id; 
            if (!id) return;
            const type = p.context_item_type; 
            
            if (type === 'experiences') mappedExpIds.add(id);
            else if (type === 'projects') mappedProjIds.add(id);
            else if (type === 'education') mappedEduIds.add(id);
            else if (type === 'hobbies') mappedHobbyIds.add(id);
        });

        const mappedExperiences = cv.experiences.filter(exp => mappedExpIds.has(exp.id));
        const mappedProjects = cv.projects.filter(proj => mappedProjIds.has(proj.id));
        const mappedEducation = cv.education.filter(edu => mappedEduIds.has(edu.id));
        const mappedHobbies = cv.hobbies.filter(hob => mappedHobbyIds.has(hob.id));
        
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

        // Group *all* skills from the base CV
        const groupedSkills = { technical: [], soft: [], language: [], other: [] };
        cv.skills.forEach(skill => {
            if (groupedSkills[skill.category]) {
                groupedSkills[skill.category].push(skill);
            } else {
                groupedSkills.other.push(skill);
            }
        });
        
        return { 
            mappedExperiences, 
            mappedProjects, 
            mappedEducation, 
            mappedHobbies, 
            aggregatedSkillIds: skillIds, // The auto-selected skills
            groupedSkills // All skills for the toggle section
        };
    }, [cv, mapping]);

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

    // --- 5. RENDER THE FULL CV PREVIEW ---
    return (
        <div>
            <h4 className="h5">Step 2: Review Your Tailored CV</h4>
            <p className="text-muted">
                This is a preview of the content that will be sent to the AI, based on your mapping.
                Deselect any skills you wish to exclude from the final prompt.
            </p>
            
            {/* The main CV preview area */}
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
                {mappedExperiences.length > 0 && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Professional Experience</h5>
                        {mappedExperiences.map(item => (
                            <CVItemDisplayCard
                                key={item.id}
                                item={item}
                                itemType="experiences"
                                allSkills={cv.skills}
                                allAchievements={cv.achievements}
                            />
                        ))}
                    </div>
                )}

                {/* Mapped Education */}
                {mappedEducation.length > 0 && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Education</h5>
                        {mappedEducation.map(item => (
                            <CVItemDisplayCard
                                key={item.id}
                                item={item}
                                itemType="education"
                                allSkills={cv.skills}
                                allAchievements={cv.achievements}
                            />
                        ))}
                    </div>
                )}
                
                {/* Mapped Projects */}
                {mappedProjects.length > 0 && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Projects</h5>
                        {mappedProjects.map(item => (
                            <CVItemDisplayCard
                                key={item.id}
                                item={item}
                                itemType="projects"
                                allSkills={cv.skills}
                                allAchievements={cv.achievements}
                                allExperiences={cv.experiences}
                                allEducation={cv.education}
                            />
                        ))}
                    </div>
                )}

                {/* Mapped Hobbies */}
                {mappedHobbies.length > 0 && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Hobbies & Interests</h5>
                        {mappedHobbies.map(item => (
                            <CVItemDisplayCard
                                key={item.id}
                                item={item}
                                itemType="hobbies"
                                allSkills={cv.skills}
                                allAchievements={cv.achievements}
                            />
                        ))}
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