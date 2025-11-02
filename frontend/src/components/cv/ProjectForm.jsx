// frontend/src/components/cv/ProjectForm.jsx
import React, { useState, useEffect } from 'react';
import SkillManagerModal from './SkillManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementManagerModal from './AchievementManagerModal';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const ProjectForm = ({
    onSubmit,
    cvId,
    // NEW: We need these for the dropdowns
    allExperiences = [],
    allEducation = [],
    allSkills,
    allAchievements,
    initialData,
    onCancelEdit
}) => {
    // Form fields for Project
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    
    // NEW: State for the related item dropdowns
    const [relatedExperienceId, setRelatedExperienceId] = useState('');
    const [relatedEducationId, setRelatedEducationId] = useState('');
    
    // State for *direct* skills
    const [directSkillIds, setDirectSkillIds] = useState([]);
    const [directPendingSkills, setDirectPendingSkills] = useState([]);

    // State for achievements
    const [linkedExistingAchievements, setLinkedExistingAchievements] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    
    // Modal state
    const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
    const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

    // State for "rolled-up" display
    const [aggregatedSkillIds, setAggregatedSkillIds] = useState([]);
    const [aggregatedPendingSkills, setAggregatedPendingSkills] = useState([]);

    const [disabledSkillsForModal, setDisabledSkillsForModal] = useState([]);

    const isEditing = Boolean(initialData);

    // This effect populates the form on load
    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setDescription(initialData.description || '');
            // NEW
            setRelatedExperienceId(initialData.related_experience_id || '');
            setRelatedEducationId(initialData.related_education_id || '');
            
            setDirectSkillIds(initialData.skill_ids || []);
            setDirectPendingSkills([]); 

            const initialAchievements = (initialData.achievement_ids || [])
                .map(id => allAchievements.find(a => a.id === id))
                .filter(Boolean)
                .map(ach => ({ ...ach })); 
            setLinkedExistingAchievements(initialAchievements);
            setPendingAchievements([]);
        } else {
            // Reset form for "create new"
            setTitle('');
            setDescription('');
            // NEW
            setRelatedExperienceId('');
            setRelatedEducationId('');
            
            setDirectSkillIds([]); 
            setDirectPendingSkills([]); 
            setLinkedExistingAchievements([]);
            setPendingAchievements([]);
        }
    }, [initialData, isEditing, cvId, allAchievements]); 

    
    // This effect calculates the aggregated lists (unchanged)
    useEffect(() => {
        const allIds = new Set(directSkillIds);
        const achIds = new Set(); 

        linkedExistingAchievements.forEach(ach => {
            (ach.skill_ids || []).forEach(id => achIds.add(id));
        });
        pendingAchievements.forEach(ach => {
            (ach.skill_ids || []).forEach(id => achIds.add(id));
        });

        achIds.forEach(id => allIds.add(id));
        setAggregatedSkillIds(Array.from(allIds));

        const directSet = new Set(directSkillIds);
        const disabledIds = Array.from(achIds).filter(id => !directSet.has(id));
        setDisabledSkillsForModal(disabledIds);

        const allPending = [...directPendingSkills];
        const pendingNames = new Set(directPendingSkills.map(s => s.name));
        
        pendingAchievements.forEach(ach => {
            (ach.new_skills || []).forEach(ps => {
                if (!pendingNames.has(ps.name)) {
                    allPending.push(ps);
                    pendingNames.add(ps.name);
                }
            });
        });
        setAggregatedPendingSkills(allPending);

    }, [directSkillIds, directPendingSkills, linkedExistingAchievements, pendingAchievements]);
    
    
    // handleExistingAchievementSelection handler (unchanged)
    const handleExistingAchievementSelection = (newIdList) => {
        const newIds = newIdList.filter(id => !linkedExistingAchievements.some(a => a.id === id));
        const removedIds = linkedExistingAchievements.map(a => a.id).filter(id => !newIdList.includes(id));
        let newList = [...linkedExistingAchievements];
        newList = newList.filter(a => !removedIds.includes(a.id));
        newIds.forEach(id => {
            const ach = allAchievements.find(a => a.id === id);
            if (ach) {
                newList.push({ ...ach }); 
            }
        });
        setLinkedExistingAchievements(newList);
    };

    // handleSkillSelectionChange handler (unchanged)
    const handleSkillSelectionChange = (newAggregatedList) => {
        const oldAggregatedList = aggregatedSkillIds; 

        const removedSkillIds = oldAggregatedList.filter(id => !newAggregatedList.includes(id));
        const addedSkillIds = newAggregatedList.filter(id => !oldAggregatedList.includes(id));

        if (addedSkillIds.length > 0) {
            setDirectSkillIds(prev => [...new Set([...prev, ...addedSkillIds])]);
        }
        
        if (removedSkillIds.length > 0) {
             setDirectSkillIds(prev => prev.filter(id => !removedSkillIds.includes(id)));

            setPendingAchievements(prevPending => 
                prevPending.map(ach => ({
                    ...ach,
                    skill_ids: (ach.skill_ids || []).filter(id => !removedSkillIds.includes(id))
                }))
            );
            
            const newPendingFromMaster = [];
            const stillLinkedMasterAchievements = []; 

            linkedExistingAchievements.forEach(ach => {
                const hasRemovedSkill = (ach.skill_ids || []).some(skillId => removedSkillIds.includes(skillId));
                if (hasRemovedSkill) {
                    newPendingFromMaster.push({
                        ...ach,
                        skill_ids: (ach.skill_ids || []).filter(id => !removedSkillIds.includes(id)),
                        original_id: ach.id, 
                        id: `pending-mod-${ach.id}-${Date.now()}` 
                    });
                } else {
                    stillLinkedMasterAchievements.push(ach);
                }
            });

            setLinkedExistingAchievements(stillLinkedMasterAchievements);
            setPendingAchievements(prevPending => [
                ...prevPending, 
                ...newPendingFromMaster
            ]);
        }
    };

    // "Smart" handler for pending skills (unchanged)
    const smartSetAggregatedPendingSkills = (updaterFn) => {
        const currentAggregated = aggregatedPendingSkills;
        const newAggregated = updaterFn(currentAggregated);
        const currentNames = new Set(currentAggregated.map(s => s.name));
        const newNames = new Set(newAggregated.map(s => s.name));
        const addedSkills = newAggregated.filter(s => !currentNames.has(s.name));
        const removedSkillNames = currentAggregated
            .filter(s => !newNames.has(s.name))
            .map(s => s.name);
        
        if (addedSkills.length > 0) {
            setDirectPendingSkills(prev => [
                ...prev,
                ...addedSkills.filter(added => !prev.some(p => p.name === added.name))
            ]);
        }

        if (removedSkillNames.length > 0) {
            setDirectPendingSkills(prev => 
                prev.filter(s => !removedSkillNames.includes(s.name))
            );
            setPendingAchievements(prev => 
                prev.map(ach => ({
                    ...ach,
                    new_skills: (ach.new_skills || []).filter(s => !removedSkillNames.includes(s.name))
                }))
            );
        }
    };

    // handleSubmit (Modified for Project)
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim()) return;

        const dataToSend = {
            title,
            description: description || null,
            
            // NEW: Add the related IDs
            related_experience_id: relatedExperienceId || null,
            related_education_id: relatedEducationId || null,
            
            existing_skill_ids: directSkillIds, 
            new_skills: directPendingSkills,

            existing_achievement_ids: linkedExistingAchievements.map(a => a.id),
            new_achievements: pendingAchievements,
        };

        if (isEditing) {
            dataToSend.id = initialData.id;
        }

        // Submit as 'Project'
        onSubmit(cvId, dataToSend, 'Project');

        if (!isEditing) {
            setTitle('');
            setDescription('');
            setRelatedExperienceId('');
            setRelatedEducationId('');
            setDirectSkillIds([]);
            setDirectPendingSkills([]);
            setLinkedExistingAchievements([]); 
            setPendingAchievements([]);
        }
    };

    const allAchievementsToShow = [...linkedExistingAchievements, ...pendingAchievements];

    return (
        <form 
            key={initialData?.id || 'new'} 
            onSubmit={handleSubmit} 
            className="card p-3"
            style={{ borderTop: `4px solid #fd7e14` }} // Project theme color
        >
            <h4 className="mt-0 mb-3" style={{ color: '#fd7e14' }}>
                {isEditing ? 'Edit Project' : 'Add New Project'}
            </h4>

            {/* Form fields (Modified for Project) */}
            <div className="mb-3">
                <label htmlFor="project-title" className="form-label fw-medium">Project Title</label>
                <input id="project-title" type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., My Portfolio Website" required className="form-control" />
            </div>
             <div className="mb-3">
                <label htmlFor="project-desc" className="form-label fw-medium">Description (Optional)</label>
                <textarea id="project-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="e.g., Built with React and FastAPI..." className="form-control" rows="3"/>
            </div>
            
            {/* --- NEW: Related Item Linking --- */}
            <div className="row g-2 mb-3">
                <div className="col-md-6">
                    <label htmlFor="project-rel-exp" className="form-label fw-medium">Related Experience (Optional)</label>
                    <select
                        id="project-rel-exp"
                        className="form-select"
                        value={relatedExperienceId}
                        onChange={(e) => setRelatedExperienceId(e.target.value)}
                    >
                        <option value="">-- None --</option>
                        {allExperiences.map(exp => (
                            <option key={exp.id} value={exp.id}>
                                {exp.title} @ {exp.company}
                            </option>
                        ))}
                    </select>
                </div>
                <div className="col-md-6">
                    <label htmlFor="project-rel-edu" className="form-label fw-medium">Related Education (Optional)</label>
                    <select
                        id="project-rel-edu"
                        className="form-select"
                        value={relatedEducationId}
                        onChange={(e) => setRelatedEducationId(e.target.value)}
                    >
                        <option value="">-- None --</option>
                        {allEducation.map(edu => (
                            <option key={edu.id} value={edu.id}>
                                {edu.degree} @ {edu.institution}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            {/* --- SKILLS Section (unchanged) --- */}
            <div className="mb-3">
                <strong className="form-label">Skills:</strong>
                <button 
                    type="button" 
                    onClick={() => setIsSkillModalOpen(true)} 
                    className="btn btn-secondary btn-sm d-block mb-2"
                >
                    Manage Skills
                </button>
                <SelectedSkillsDisplay
                    allSkills={allSkills}
                    selectedSkillIds={aggregatedSkillIds}
                    pendingSkills={aggregatedPendingSkills}
                />
            </div>

            {/* --- ACHIEVEMENTS Section (unchanged) --- */}
            <div className="mb-3">
                 <strong className="form-label">Achievements:</strong>
                 <button 
                    type="button" 
                    onClick={() => setIsAchievementModalOpen(true)} 
                    className="btn btn-secondary btn-sm d-block mb-2"
                 >
                     Manage Achievements
                 </button>
                 <AchievementDisplayGrid
                     achievementsToDisplay={allAchievementsToShow}
                     allSkills={allSkills}
                     isDisplayOnly={true}
                 />
            </div>

            {/* --- Modals (unchanged) --- */}
            <SkillManagerModal
                isOpen={isSkillModalOpen}
                onClose={() => setIsSkillModalOpen(false)}
                allSkills={allSkills}
                selectedSkillIds={aggregatedSkillIds}
                setSelectedSkillIds={handleSkillSelectionChange}
                pendingSkills={aggregatedPendingSkills}
                setPendingSkills={smartSetAggregatedPendingSkills}
                disabledSkillIds={disabledSkillsForModal}
            />
             <AchievementManagerModal
                 isOpen={isAchievementModalOpen}
                 onClose={() => setIsAchievementModalOpen(false)}
                 allAchievements={allAchievements}
                 selectedAchievementIds={linkedExistingAchievements.map(a => a.id)}
                 setSelectedAchievementIds={handleExistingAchievementSelection}
                 pendingAchievements={pendingAchievements}
                 setPendingAchievements={setPendingAchievements}
                 allSkills={allSkills}
             />

            {/* --- ACTION BUTTONS (Modified) --- */}
            <div className="mt-3 border-top pt-3">
                <button type="submit" className="btn btn-primary me-2">
                    {isEditing ? 'Save Changes' : 'Add Project'}
                </button>
                
                {onCancelEdit && (
                    <button 
                        type="button" 
                        onClick={onCancelEdit} 
                        className="btn btn-outline-secondary"
                    >
                        Cancel
                    </button>
                )}
            </div>
        </form>
    );
};
export default ProjectForm;