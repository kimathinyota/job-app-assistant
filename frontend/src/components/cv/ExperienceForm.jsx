// frontend/src/components/cv/ExperienceForm.jsx
import React, { useState, useEffect } from 'react';
import { Briefcase, Layers, Award } from 'lucide-react'; // Added Icons
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
    // Form fields
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [description, setDescription] = useState('');
    
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

    // Populate form on load
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

    const handleStartDateChange = (e) => {
        const newStartDate = e.target.value;
        setStartDate(newStartDate);
        if (endDate && newStartDate > endDate) {
            setEndDate('');
        }
    };

    const allAchievementsToShow = [...linkedExistingAchievements, ...pendingAchievements];

    return (
        <form 
            key={initialData?.id || 'new'} 
            onSubmit={handleSubmit} 
            className="card border-0 shadow-sm p-4" // Modern Card
        >
            {/* Header Title */}
            <div className="d-flex align-items-center gap-2 mb-4 border-bottom pb-2">
                <Briefcase className="text-primary" size={20}/>
                <h5 className="mb-0 fw-bold text-dark">
                    {isEditing ? 'Edit Experience' : 'Add New Experience'}
                </h5>
            </div>

            {/* Fields */}
            <div className="row g-3">
                <div className="col-md-6">
                    <label htmlFor="exp-title" className="form-label fw-bold small text-uppercase text-muted">Job Title</label>
                    <input id="exp-title" type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., Senior Developer" required className="form-control" />
                </div>
                <div className="col-md-6">
                    <label htmlFor="exp-company" className="form-label fw-bold small text-uppercase text-muted">Company</label>
                    <input id="exp-company" type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="e.g., Acme Inc." required className="form-control"/>
                </div>
                <div className="col-md-6">
                    <label htmlFor="exp-start" className="form-label fw-bold small text-uppercase text-muted">Start Date</label>
                    <input 
                        id="exp-start" 
                        type="date" 
                        value={startDate} 
                        onChange={handleStartDateChange} 
                        className="form-control"
                    />
                </div>
                <div className="col-md-6">
                    <label htmlFor="exp-end" className="form-label fw-bold small text-uppercase text-muted">End Date</label>
                    <input 
                        id="exp-end" 
                        type="date" 
                        value={endDate} 
                        onChange={(e) => setEndDate(e.target.value)} 
                        min={startDate} 
                        className="form-control"
                    />
                    <div className="form-text small">
                        Leave blank for 'Present' or ongoing.
                    </div>
                </div>
                <div className="col-12">
                    <label htmlFor="exp-desc" className="form-label fw-bold small text-uppercase text-muted">Description</label>
                    <textarea id="exp-desc" value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Brief overview of responsibilities..." className="form-control" rows="3"/>
                </div>
            </div>

            <hr className="my-4 opacity-10" />

            {/* SKILLS Section */}
            <div className="mb-4">
                <div className="d-flex justify-content-between align-items-center mb-2">
                    <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0">
                        <Layers size={16} className="text-emerald-600"/> Skills Used
                    </label>
                    <button 
                        type="button" 
                        onClick={() => setIsSkillModalOpen(true)} 
                        className="btn btn-outline-secondary btn-sm py-0 px-2"
                        style={{fontSize: '0.8rem'}}
                    >
                        + Manage
                    </button>
                </div>
                <div className="bg-light p-3 rounded border">
                    <SelectedSkillsDisplay
                        allSkills={allSkills}
                        selectedSkillIds={aggregatedSkillIds}
                        pendingSkills={aggregatedPendingSkills}
                    />
                    {aggregatedSkillIds.length === 0 && aggregatedPendingSkills.length === 0 && (
                        <span className="text-muted small fst-italic">No skills linked yet.</span>
                    )}
                </div>
            </div>

            {/* ACHIEVEMENTS Section */}
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

            {/* Modals */}
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
                    {isEditing ? 'Save Changes' : 'Add Experience'}
                </button>
            </div>
        </form>
    );
};
export default ExperienceForm;