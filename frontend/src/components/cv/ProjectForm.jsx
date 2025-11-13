// frontend/src/components/cv/ProjectForm.jsx
import React, { useState, useEffect } from 'react';
import { Cpu, Layers, Award, Link, ChevronDown, ChevronUp } from 'lucide-react';
import SkillLinker from './SkillLinker'; // <-- Import new linker
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementManagerModal from './AchievementManagerModal';
import AchievementDisplayGrid from './AchievementDisplayGrid';
// import SkillManagerModal from './SkillManagerModal'; // <-- Removed modal

const ProjectForm = ({
    onSubmit,
    cvId,
    allExperiences = [],
    allEducation = [],
    allSkills,
    allAchievements,
    initialData,
    onCancelEdit
}) => {
    // Form fields
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    
    // Related item dropdowns
    const [relatedExperienceId, setRelatedExperienceId] = useState('');
    const [relatedEducationId, setRelatedEducationId] = useState('');
    
    // State for *direct* skills
    const [directSkillIds, setDirectSkillIds] = useState([]);
    const [directPendingSkills, setDirectPendingSkills] = useState([]);

    // State for achievements
    const [linkedExistingAchievements, setLinkedExistingAchievements] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    
    // State
    const [showSkillLinker, setShowSkillLinker] = useState(false); // <-- ADDED
    // const [isSkillModalOpen, setIsSkillModalOpen] = useState(false); // <-- REMOVED
    const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

    // State for "rolled-up" display
    const [aggregatedSkillIds, setAggregatedSkillIds] = useState([]);
    const [aggregatedPendingSkills, setAggregatedPendingSkills] = useState([]);

    // This state is no longer needed by the new SkillLinker
    // const [disabledSkillsForModal, setDisabledSkillsForModal] = useState([]);

    const isEditing = Boolean(initialData);

    // Populate form on load
    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setDescription(initialData.description || '');
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
            setTitle('');
            setDescription('');
            setRelatedExperienceId('');
            setRelatedEducationId('');
            setDirectSkillIds([]); 
            setDirectPendingSkills([]); 
            setLinkedExistingAchievements([]);
            setPendingAchievements([]);
            setShowSkillLinker(false); // Reset toggle
        }
    }, [initialData, isEditing, cvId, allAchievements]); 

    
    // Calculate aggregated lists
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

        // This is no longer needed by SkillLinker
        // const directSet = new Set(directSkillIds);
        // const disabledIds = Array.from(achIds).filter(id => !directSet.has(id));
        // setDisabledSkillsForModal(disabledIds);

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
    
    
    // Handlers
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

    // --- LOGIC PRESERVED ---
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

    // --- LOGIC PRESERVED ---
    const smartSetAggregatedPendingSkills = (updaterFn) => {
        const currentAggregated = aggregatedPendingSkills;
        const newAggregated = typeof updaterFn === 'function' ? updaterFn(currentAggregated) : updaterFn;
        
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

    // --- LOGIC PRESERVED ---
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim()) return;

        const dataToSend = {
            title,
            description: description || null,
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
            setShowSkillLinker(false);
        }
    };

    const allAchievementsToShow = [...linkedExistingAchievements, ...pendingAchievements];

    return (
        <form 
            key={initialData?.id || 'new'} 
            onSubmit={handleSubmit} 
            className="card border-0 shadow-sm p-4 bg-white"
        >
            {/* Header */}
            <div className="d-flex align-items-center gap-2 mb-4 border-bottom pb-2">
                <Cpu className="text-purple-600" size={20}/>
                <h5 className="mb-0 fw-bold text-dark">
                    {isEditing ? 'Edit Project' : 'Add New Project'}
                </h5>
            </div>

            {/* Core Fields */}
            <div className="mb-3">
                <label htmlFor="project-title" className="form-label fw-bold small text-uppercase text-muted">Project Title</label>
                <input id="project-title" type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., My Portfolio Website" required className="form-control" />
            </div>
             <div className="mb-3">
                <label htmlFor="project-desc" className="form-label fw-bold small text-uppercase text-muted">Description</label>
                <textarea id="project-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="e.g., Built with React and FastAPI..." className="form-control" rows="3"/>
            </div>
            
            <hr className="my-4 opacity-10" />

            {/* Relationships Section */}
            <div className="row g-2 mb-4">
                <label className="form-label fw-bold small text-uppercase text-muted d-flex align-items-center gap-2">
                    <Link size={14}/> Linked Context (Optional)
                </label>
                <div className="col-md-6">
                    <div className="form-floating">
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
                        <label htmlFor="project-rel-exp">Related Experience</label>
                    </div>
                </div>
                <div className="col-md-6">
                    <div className="form-floating">
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
                        <label htmlFor="project-rel-edu">Related Education</label>
                    </div>
                </div>
            </div>

            {/* --- SKILLS Section (Integrated Linker) --- */}
            <div className="mb-4">
                <div 
                    className="d-flex justify-content-between align-items-center mb-2 cursor-pointer"
                    onClick={() => setShowSkillLinker(!showSkillLinker)}
                >
                    <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0 cursor-pointer">
                        <Layers size={16} className="text-emerald-600"/> 
                        Skills Used
                        <span className="text-muted fw-normal small">
                            ({aggregatedSkillIds.length + aggregatedPendingSkills.length})
                        </span>
                    </label>
                    <button 
                        type="button" 
                        className="btn btn-light btn-sm text-secondary"
                    >
                        {showSkillLinker ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
                    </button>
                </div>

                {showSkillLinker ? (
                    <div className="animate-fade-in mt-2 p-3 bg-light rounded border">
                        <SkillLinker
                            allSkills={allSkills}
                            // --- BIND TO THE CORE LOGIC ---
                            selectedSkillIds={aggregatedSkillIds}
                            setSelectedSkillIds={handleSkillSelectionChange}
                            pendingSkills={aggregatedPendingSkills}
                            setPendingSkills={smartSetAggregatedPendingSkills}
                            // ---
                            sessionSkills={aggregatedPendingSkills} 
                        />
                    </div>
                ) : (
                    (aggregatedSkillIds.length > 0 || aggregatedPendingSkills.length > 0) ? (
                        <div 
                            className="bg-light p-3 rounded border cursor-pointer hover:bg-slate-100 transition-all"
                            onClick={() => setShowSkillLinker(true)}
                        >
                            <SelectedSkillsDisplay
                                allSkills={allSkills}
                                selectedSkillIds={aggregatedSkillIds}
                                pendingSkills={aggregatedPendingSkills}
                            />
                        </div>
                    ) : (
                        <div 
                            className="text-muted small fst-italic border border-dashed rounded p-2 text-center cursor-pointer hover:bg-light"
                            onClick={() => setShowSkillLinker(true)}
                        >
                            Click to link skills...
                        </div>
                    )
                )}
            </div>

            {/* --- ACHIEVEMENTS Section --- */}
            <div className="mb-4">
                 <div className="d-flex justify-content-between align-items-center mb-2">
                     <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0">
                        <Award size={16} className="text-amber-500"/> Achievements
                     </label>
                     <button 
                        type="button" 
                        onClick={() => setIsAchievementModalOpen(true)} 
                        className="btn btn-outline-secondary btn-sm py-0 px-2"
                        style={{fontSize: '0.8rem'}}
                     >
                         + Manage
                     </button>
                 </div>
                 {allAchievementsToShow.length > 0 ? (
                     <div className="bg-light p-3 rounded border">
                        <AchievementDisplayGrid
                            achievementsToDisplay={allAchievementsToShow}
                            allSkills={allSkills}
                            isDisplayOnly={true}
                        />
                     </div>
                 ) : (
                     <div className="bg-light p-3 rounded border text-center">
                        <span className="text-muted small fst-italic">No achievements added.</span>
                     </div>
                 )}
            </div>

            {/* Achievement Modal */}
             <AchievementManagerModal
                 isOpen={isAchievementModalOpen}
                 onClose={() => setIsAchievementModalOpen(false)}
                 allAchievements={allAchievements}
                 selectedAchievementIds={linkedExistingAchievements.map(a => a.id)}
                 setSelectedAchievementIds={handleExistingAchievementSelection}
                 pendingAchievements={pendingAchievements}
                 setPendingAchievements={setPendingAchievements}
                 allSkills={allSkills}
                 sessionSkills={aggregatedPendingSkills}
             />

            {/* ACTION BUTTONS */}
            <div className="d-flex gap-2 justify-content-end mt-4 pt-3 border-top">
                {onCancelEdit && (
                    <button 
                        type="button" 
                        onClick={onCancelEdit} 
                        className="btn btn-light border"
                    >
                        Cancel
                    </button>
                )}
                <button type="submit" className="btn btn-primary px-4">
                    {isEditing ? 'Save Changes' : 'Add Project'}
                </button>
            </div>
        </form>
    );
};
export default ProjectForm;