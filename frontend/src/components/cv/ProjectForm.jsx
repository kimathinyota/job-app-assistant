import React, { useState, useEffect } from 'react';
import { Cpu, Layers, Award, Link, ChevronDown, ChevronUp, Briefcase } from 'lucide-react';
import SkillLinker from './SkillLinker';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementManagerPanel from './AchievementManagerPanel'; // 1. IMPORT PANEL
import AchievementDisplayGrid from './AchievementDisplayGrid';
import ContextLinker from './ContextLinker';
import SkillManagerPanel from './SkillManagerPanel'; // 2. IMPORT PANEL
import { useWindowSize } from '../../hooks/useWindowSize'; // 3. IMPORT HOOK

const ProjectForm = ({
    onSubmit,
    cvId,
    allExperiences = [],
    allEducation = [],
    allHobbies = [],
    allSkills,
    allAchievements,
    initialData,
    onCancelEdit
}) => {
    // 4. HOOK FOR RESPONSIVENESS
    const { width } = useWindowSize();
    const isMobile = width <= 768; // Bootstrap 'md' breakpoint

    // Form fields
    const [title, setTitle] = useState('');
    const [description, setDescription] = useState('');
    
    // Related item Lists
    const [relatedExpIds, setRelatedExpIds] = useState([]);
    const [relatedEduIds, setRelatedEduIds] = useState([]);
    const [relatedHobbyIds, setRelatedHobbyIds] = useState([]);
    
    // State for *direct* skills
    const [directSkillIds, setDirectSkillIds] = useState([]);
    const [directPendingSkills, setDirectPendingSkills] = useState([]);

    // State for achievements
    const [linkedExistingAchievements, setLinkedExistingAchievements] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    
    // Toggles
    const [showSkillLinker, setShowSkillLinker] = useState(false);
    const [showContextLinker, setShowContextLinker] = useState(false);
    const [isAchievementPanelOpen, setIsAchievementPanelOpen] = useState(false); // 5. STATE RENAMED
    const [isSkillPanelOpen, setIsSkillPanelOpen] = useState(false); // 6. NEW STATE

    // State for "rolled-up" display (BUG FIX ALREADY PRESENT)
    const [aggregatedSkillIds, setAggregatedSkillIds] = useState([]);
    const [aggregatedPendingSkills, setAggregatedPendingSkills] = useState([]);

    const isEditing = Boolean(initialData);

    // Populate form on load (Unchanged)
    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setDescription(initialData.description || '');
            
            const initExpIds = initialData.related_experience_ids || [];
            if (initExpIds.length === 0 && initialData.related_experience_id) {
                initExpIds.push(initialData.related_experience_id);
            }
            setRelatedExpIds(initExpIds);

            const initEduIds = initialData.related_education_ids || [];
            if (initEduIds.length === 0 && initialData.related_education_id) {
                initEduIds.push(initialData.related_education_id);
            }
            setRelatedEduIds(initEduIds);

            const initHobbyIds = initialData.related_hobby_ids || [];
            setRelatedHobbyIds(initHobbyIds);
            
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
            setRelatedExpIds([]);
            setRelatedEduIds([]);
            setRelatedHobbyIds([]);
            setDirectSkillIds([]); 
            setDirectPendingSkills([]); 
            setLinkedExistingAchievements([]);
            setPendingAchievements([]);
            setShowSkillLinker(false);
            setShowContextLinker(false);
        }
    }, [initialData, isEditing, cvId, allAchievements]); 

    
    // Calculate aggregated lists (Unchanged)
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

    // Context Handler (Unchanged)
    const handleContextToggle = (type, id) => {
        if (type === 'experiences') {
            setRelatedExpIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
        } else if (type === 'education') {
            setRelatedEduIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
        } else if (type === 'hobbies') {
            setRelatedHobbyIds(prev => prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]);
        }
    };

    // Achievement Handler (Unchanged)
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

    // Skill Selection Handler (Unchanged)
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

    // Smart Pending Skills Handler (Unchanged)
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

    // Submit Handler (Unchanged)
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim()) return;

        const dataToSend = {
            title,
            description: description || null,
            
            related_experience_ids: relatedExpIds,
            related_education_ids: relatedEduIds,
            related_hobby_ids: relatedHobbyIds,
            
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
            setRelatedExpIds([]);
            setRelatedEduIds([]);
            setRelatedHobbyIds([]);
            setDirectSkillIds([]);
            setDirectPendingSkills([]);
            setLinkedExistingAchievements([]); 
            setPendingAchievements([]);
            setShowSkillLinker(false);
            setShowContextLinker(false);
        }
    };

    // 7. ADD isPending FLAG
    const linkedWithFlag = linkedExistingAchievements.map(a => ({ ...a, isPending: false }));
    const pendingWithFlag = pendingAchievements.map(a => ({ ...a, isPending: true }));
    const allAchievementsToShow = [...linkedWithFlag, ...pendingWithFlag];

    // 8. NEW RESPONSIVE HANDLER
    const handleSkillToggle = () => {
        if (isMobile) {
            setIsSkillPanelOpen(true);
        } else {
            setShowSkillLinker(!showSkillLinker);
        }
    };

    return (
        <form 
            key={initialData?.id || 'new'} 
            onSubmit={handleSubmit} 
            className="card border-0 shadow-sm p-3 p-md-4 bg-white" // Responsive padding
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

            {/* --- ContextLinker Section (LOGIC UNCHANGED) --- */}
            <div className="mb-4">
                <div 
                    className="d-flex justify-content-between align-items-center mb-2 cursor-pointer"
                    onClick={() => setShowContextLinker(!showContextLinker)}
                >
                    <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0 cursor-pointer">
                        <Link size={16} className="text-info"/> 
                        Linked Context
                        <span className="text-muted fw-normal small">
                            ({relatedExpIds.length + relatedEduIds.length + relatedHobbyIds.length})
                        </span>
                    </label>
                    <button 
                        type="button" 
                        className="btn btn-light btn-sm text-secondary"
                        onClick={() => setShowContextLinker(!showContextLinker)}
                    >
                        {showContextLinker ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
                    </button>
                </div>

                {/* Badge Display */}
                {!showContextLinker && (
                    (relatedExpIds.length + relatedEduIds.length + relatedHobbyIds.length) > 0 ? (
                        <div 
                            className="bg-light p-3 rounded border cursor-pointer hover-bg-slate-100 transition-all d-flex flex-wrap gap-2"
                            onClick={() => setShowContextLinker(true)}
                        >
                            {relatedExpIds.map(id => {
                                const item = allExperiences.find(e => e.id === id);
                                return item ? (
                                    <span key={id} className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-1">
                                        <Briefcase size={10} className="text-primary"/> {item.title}
                                    </span>
                                ) : null;
                            })}
                            {relatedEduIds.map(id => {
                                const item = allEducation.find(e => e.id === id);
                                return item ? (
                                    <span key={id} className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-1">
                                        <BookOpen size={10} className="text-indigo-600"/> {item.degree}
                                    </span>
                                ) : null;
                            })}
                            {relatedHobbyIds.map(id => {
                                const item = allHobbies.find(e => e.id === id);
                                return item ? (
                                    <span key={id} className="badge bg-white text-secondary border fw-normal d-flex align-items-center gap-1">
                                        <Smile size={10} className="text-pink-500"/> {item.name}
                                    </span>
                                ) : null;
                            })}
                        </div>
                    ) : (
                         <div 
                            className="text-muted small fst-italic border border-dashed rounded p-2 text-center cursor-pointer hover:bg-light"
                            onClick={() => setShowContextLinker(true)}
                        >
                            Click to link context (experiences, education..._
                        </div>
                    )
                )}
                

                {showContextLinker && (
                    <div className="animate-fade-in mt-2 p-3 bg-light rounded border">
                        <ContextLinker 
                            allExperiences={allExperiences}
                            allEducation={allEducation}
                            allHobbies={allHobbies} 
                            selectedIds={{
                                experiences: relatedExpIds,
                                education: relatedEduIds,
                                hobbies: relatedHobbyIds
                            }}
                            onToggle={handleContextToggle}
                        />
                    </div>
                )}
            </div>

            {/* --- 9. SKILLS Section (RESPONSIVE) --- */}
            <div className="mb-4">
                <div 
                    className="d-flex justify-content-between align-items-center mb-2 cursor-pointer"
                    onClick={handleSkillToggle} // Use new handler
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
                        {/* Show chevron on mobile, or toggle up/down on desktop */}
                        {isMobile ? 
                            <ChevronDown size={16}/> : 
                            (showSkillLinker ? <ChevronUp size={16}/> : <ChevronDown size={16}/>)
                        }
                    </button>
                </div>

                {/* --- RENDER LOGIC FOR SKILLS --- */}
                {/* A. On Desktop, show inline linker */}
                {!isMobile && showSkillLinker && (
                    <div className="animate-fade-in mt-2 p-3 bg-light rounded border">
                        <SkillLinker
                            allSkills={allSkills}
                            selectedSkillIds={aggregatedSkillIds}
                            setSelectedSkillIds={handleSkillSelectionChange}
                            pendingSkills={aggregatedPendingSkills}
                            setPendingSkills={smartSetAggregatedPendingSkills}
                            sessionSkills={aggregatedPendingSkills} 
                        />
                    </div>
                )}
                
                {/* B. On Desktop AND Mobile, show display card when linker is hidden */}
                {(isMobile || !showSkillLinker) && (aggregatedSkillIds.length > 0 || aggregatedPendingSkills.length > 0) ? (
                    <div 
                        className="bg-light p-3 rounded border cursor-pointer hover-bg-slate-100 transition-all"
                        onClick={handleSkillToggle} // This will now open the panel on mobile
                    >
                        <SelectedSkillsDisplay
                            allSkills={allSkills}
                            selectedSkillIds={aggregatedSkillIds}
                            pendingSkills={aggregatedPendingSkills}
                        />
                    </div>
                ) : null}
                
                {/* C. On Desktop AND Mobile, show empty state when linker is hidden */}
                {(isMobile || !showSkillLinker) && !(aggregatedSkillIds.length > 0 || aggregatedPendingSkills.length > 0) ? (
                    <div 
                        className="text-muted small fst-italic border border-dashed rounded p-2 text-center cursor-pointer hover:bg-light"
                        onClick={handleSkillToggle} // This will now open the panel on mobile
                    >
                        Click to link skills...
                    </div>
                ) : null}
            </div>

            {/* --- 10. ACHIEVEMENTS Section (RESPONSIVE) --- */}
            <div className="mb-4">
                 <div className="d-flex justify-content-between align-items-center mb-2">
                     <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0">
                        <Award size={16} className="text-amber-500"/> Achievements
                     </label>
                     {/* Responsive Button */}
                     <button 
                        type="button" 
                        onClick={() => setIsAchievementPanelOpen(true)}
                        className={`btn btn-sm ${isMobile ? 'btn-light text-secondary' : 'btn-outline-secondary'}`}
                     >
                        {isMobile ? (
                            <ChevronDown size={16}/>
                        ) : (
                            <span className="py-0 px-1" style={{fontSize: '0.8rem'}}>+ Manage</span>
                        )}
                     </button>
                 </div>
                 
                 {allAchievementsToShow.length > 0 ? (
                     <div // Make grid clickable
                        className="bg-light p-3 rounded border cursor-pointer hover-bg-slate-100 transition-all"
                        onClick={() => setIsAchievementPanelOpen(true)} 
                    >
                        <AchievementDisplayGrid
                            achievementsToDisplay={allAchievementsToShow}
                            allSkills={allSkills}
                            isDisplayOnly={true}
                        />
                     </div>
                 ) : (
                     <div // Make empty state clickable
                        className="bg-light p-3 rounded border text-center cursor-pointer hover-bg-slate-100 transition-all"
                        onClick={() => setIsAchievementPanelOpen(true)}
                    >
                        <span className="text-muted small fst-italic">No achievements added. Click to manage.</span>
                     </div>
                 )}
            </div>

            {/* --- 11. RENDER THE PANELS --- */}
             <AchievementManagerPanel
                 isOpen={isAchievementPanelOpen}
                 onClose={() => setIsAchievementPanelOpen(false)}
                 allAchievements={allAchievements}
                 selectedAchievementIds={linkedExistingAchievements.map(a => a.id)}
                 setSelectedAchievementIds={handleExistingAchievementSelection}
                 pendingAchievements={pendingAchievements}
                 setPendingAchievements={setPendingAchievements}
                 allSkills={allSkills}
                 sessionSkills={aggregatedPendingSkills}
             />
             
             <SkillManagerPanel
                isOpen={isSkillPanelOpen}
                onClose={() => setIsSkillPanelOpen(false)}
                allSkills={allSkills}
                selectedSkillIds={aggregatedSkillIds}
                setSelectedSkillIds={handleSkillSelectionChange}
                pendingSkills={aggregatedPendingSkills}
                setPendingSkills={smartSetAggregatedPendingSkills}
                sessionSkills={aggregatedPendingSkills}
             />

            {/* ACTION BUTTONS (Unchanged) */}
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