// frontend/src/components/cv/ExperienceForm.jsx
import React, { useState, useEffect } from 'react';
// fetchAggregatedSkills is no longer needed here, we do it all locally
import SkillManagerModal from './SkillManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementManagerModal from './AchievementManagerModal';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const ExperienceForm = ({
    onSubmit,
    cvId,
    allSkills,
    allAchievements,
    initialData,
    onCancelEdit
}) => {
    // Form fields (unchanged)
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [description, setDescription] = useState('');
    
    // State for *direct* skills (unchanged)
    const [directSkillIds, setDirectSkillIds] = useState([]);
    const [directPendingSkills, setDirectPendingSkills] = useState([]);

    // State for achievements (unchanged)
    const [linkedExistingAchievements, setLinkedExistingAchievements] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    
    // Modal state (unchanged)
    const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
    const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

    // State for "rolled-up" display (unchanged)
    const [aggregatedSkillIds, setAggregatedSkillIds] = useState([]);
    const [aggregatedPendingSkills, setAggregatedPendingSkills] = useState([]);

    // *** 1. NEW STATE: To track skills that are disabled in the modal ***
    const [disabledSkillsForModal, setDisabledSkillsForModal] = useState([]);

    const isEditing = Boolean(initialData);

    // This effect populates the form on load (unchanged)
    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setCompany(initialData.company || '');
            setStartDate(initialData.start_date || '');
            setEndDate(initialData.end_date || '');
            setDescription(initialData.description || '');
            
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
            setCompany('');
            setStartDate('');
            setEndDate('');
            setDescription('');
            setDirectSkillIds([]); 
            setDirectPendingSkills([]); 
            setLinkedExistingAchievements([]);
            setPendingAchievements([]);
        }
    }, [initialData, isEditing, cvId, allAchievements]); 

    
    // *** 2. MODIFIED: This effect now *also* calculates disabled skills ***
    useEffect(() => {
        const allIds = new Set(directSkillIds);
        const achIds = new Set(); // Store achievement skill IDs separately

        // Get all achievement skill IDs
        linkedExistingAchievements.forEach(ach => {
            (ach.skill_ids || []).forEach(id => achIds.add(id));
        });
        pendingAchievements.forEach(ach => {
            (ach.skill_ids || []).forEach(id => achIds.add(id));
        });

        // Add achievement skills to the aggregated list
        achIds.forEach(id => allIds.add(id));
        setAggregatedSkillIds(Array.from(allIds));

        // Calculate disabled skills:
        // A skill is disabled if it's in `achIds` but NOT in `directSkillIds`
        const directSet = new Set(directSkillIds);
        const disabledIds = Array.from(achIds).filter(id => !directSet.has(id));
        setDisabledSkillsForModal(disabledIds);

        // Aggregate all pending skills (unchanged)
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

    // *** 3. MODIFIED: This handler now works with the aggregated list ***
    const handleSkillSelectionChange = (newAggregatedList) => {
        const oldAggregatedList = aggregatedSkillIds; // Get current aggregated list from state

        // Find what was added or removed *from the aggregated list*
        const removedSkillIds = oldAggregatedList.filter(id => !newAggregatedList.includes(id));
        const addedSkillIds = newAggregatedList.filter(id => !oldAggregatedList.includes(id));

        // --- Apply Changes to DIRECT skills ---
        // Additions are *always* added to direct skills
        if (addedSkillIds.length > 0) {
            setDirectSkillIds(prev => [...new Set([...prev, ...addedSkillIds])]);
        }
        
        // Removals are *always* removed from direct skills
        // (Disabled skills from achievements couldn't be removed anyway)
        if (removedSkillIds.length > 0) {
             setDirectSkillIds(prev => prev.filter(id => !removedSkillIds.includes(id)));

            // --- Now, run the achievement-pending logic (unchanged) ---
            // This checks if removing the skill from *direct* also requires modifying an achievement
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


    // handleSubmit (unchanged)
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim() || !company.trim()) return;

        const dataToSend = {
            title,
            company,
            start_date: startDate || null,
            end_date: endDate || null,
            description: description || null,
            
            existing_skill_ids: directSkillIds, 
            new_skills: directPendingSkills,

            existing_achievement_ids: linkedExistingAchievements.map(a => a.id),
            new_achievements: pendingAchievements,
        };

        if (isEditing) {
            dataToSend.id = initialData.id;
        }

        onSubmit(cvId, dataToSend, 'Experience');

        if (!isEditing) {
            setTitle('');
            setCompany('');
            setStartDate('');
            setEndDate('');
            setDescription('');
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
            style={{ borderTop: `4px solid #007bff` }}
        >
            <h4 className="text-primary mt-0 mb-3">
                {isEditing ? 'Edit Experience' : 'Add New Experience'}
            </h4>

            {/* Form fields (unchanged) */}
            <div className="mb-3">
                <label htmlFor="exp-title" className="form-label fw-medium">Job Title</label>
                <input id="exp-title" type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., Senior Developer" required className="form-control" />
            </div>
             <div className="mb-3">
                <label htmlFor="exp-company" className="form-label fw-medium">Company</label>
                <input id="exp-company" type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="e.g., Acme Inc." required className="form-control"/>
            </div>
            <div className="row g-2 mb-3">
                <div className="col-md-6">
                    <label htmlFor="exp-start" className="form-label fw-medium">Start Date</label>
                    <input id="exp-start" type="text" value={startDate} onChange={(e) => setStartDate(e.target.value)} placeholder="e.g., Jan 2020" className="form-control"/>
                </div>
                <div className="col-md-6">
                    <label htmlFor="exp-end" className="form-label fw-medium">End Date</label>
                    <input id="exp-end" type="text" value={endDate} onChange={(e) => setEndDate(e.target.value)} placeholder="e.g., Present" className="form-control"/>
                </div>
            </div>
            <div className="mb-3">
                <label htmlFor="exp-desc" className="form-label fw-medium">Description</label>
                <textarea id="exp-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Brief overview of responsibilities..." className="form-control" rows="3"/>
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

            {/* --- Modals --- */}
            {/* *** 4. MODIFIED: Wire SkillManagerModal to aggregated state *** */}
            <SkillManagerModal
                isOpen={isSkillModalOpen}
                onClose={() => setIsSkillModalOpen(false)}
                allSkills={allSkills}
                // Feed it the *full* list so all are pre-toggled
                selectedSkillIds={aggregatedSkillIds}
                // Feed it the *smart* handler
                setSelectedSkillIds={handleSkillSelectionChange}
                // Still manages *direct* pending skills
                pendingSkills={directPendingSkills}
                setPendingSkills={setDirectPendingSkills}
                // Feed it the new *disabled* list
                disabledSkillIds={disabledSkillsForModal}
            />
            {/* This modal is for achievements and is unchanged */}
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

            {/* --- Action Buttons (unchanged) --- */}
            <div className="mt-3 border-top pt-3">
                <button type="submit" className="btn btn-primary me-2">
                    {isEditing ? 'Save Changes' : 'Add Experience'}
                </button>
                {isEditing && (
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
export default ExperienceForm;