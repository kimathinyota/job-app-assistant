import React, { useState, useEffect } from 'react';
import { Briefcase, Layers, Award, ChevronDown, ChevronUp } from 'lucide-react';
import SkillLinker from './SkillLinker';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
import AchievementManagerPanel from './AchievementManagerPanel';
import AchievementDisplayGrid from './AchievementDisplayGrid';
import SkillManagerPanel from './SkillManagerPanel';
import { useWindowSize } from '../../hooks/useWindowSize';


const ExperienceForm = ({
    onSubmit,
    cvId,
    allSkills,
    allAchievements,
    initialData,
    onCancelEdit
}) => {
    const { width } = useWindowSize();
    const isMobile = width <= 768; 

    // Form fields
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [location, setLocation] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [description, setDescription] = useState('');
    
    // State for *direct* skills
    const [directSkillIds, setDirectSkillIds] = useState([]);
    const [directPendingSkills, setDirectPendingSkills] = useState([]);

    // State for achievements
    const [linkedExistingAchievements, setLinkedExistingAchievements] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    
    // Toggles
    const [showSkillLinker, setShowSkillLinker] = useState(false);
    const [isAchievementPanelOpen, setIsAchievementPanelOpen] = useState(false);
    const [isSkillPanelOpen, setIsSkillPanelOpen] = useState(false); 

    const [aggregatedSkillIds, setAggregatedSkillIds] = useState([]);
    const [aggregatedPendingSkills, setAggregatedPendingSkills] = useState([]);

    const isEditing = Boolean(initialData);

    // --- HELPER: Ensure date is YYYY-MM-DD for input ---
    const formatDateForInput = (dateString) => {
        console.log("Formatting date:", dateString);
        if (!dateString) return '';
        // If it's already YYYY-MM-DD, return it
        if (/^\d{4}-\d{2}-\d{2}$/.test(dateString)) return dateString;
        // If ISO string (e.g. 2023-01-01T00:00:00), split it
        if (dateString.includes('T')) return dateString.split('T')[0];
        return '';
    };

    // Populate form on load
    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setCompany(initialData.company || '');
            setLocation(initialData.location || '');
            
            // --- FIX: Use helper to format dates ---
            setStartDate(formatDateForInput(initialData.start_date));
            setEndDate(formatDateForInput(initialData.end_date));
            
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
            setLocation('');
            setStartDate('');
            setEndDate('');
            setDescription('');
            setDirectSkillIds([]); 
            setDirectPendingSkills([]); 
            setLinkedExistingAchievements([]);
            setPendingAchievements([]);
            setShowSkillLinker(false);
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
    
    
    // Handler: Achievement Selection
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

    // Handler: Skill Selection
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

    // Handler: Smart Pending Skills
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

    // Handler: Submit
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim() || !company.trim()) return;

        const dataToSend = {
            title,
            company,
            location: location || null,
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
            setLocation('');
            setStartDate('');
            setEndDate('');
            setDescription('');
            setDirectSkillIds([]);
            setDirectPendingSkills([]);
            setLinkedExistingAchievements([]); 
            setPendingAchievements([]);
            setShowSkillLinker(false);
        }
    };

    const handleStartDateChange = (e) => {
        const newStartDate = e.target.value;
        setStartDate(newStartDate);
        if (endDate && newStartDate > endDate) {
            setEndDate('');
        }
    };

    const linkedWithFlag = linkedExistingAchievements.map(a => ({ ...a, isPending: false }));
    const pendingWithFlag = pendingAchievements.map(a => ({ ...a, isPending: true }));
    const allAchievementsToShow = [...linkedWithFlag, ...pendingWithFlag];

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
            className="card border-0 shadow-sm p-3 p-md-4 bg-white" 
        >
            {/* Header */}
            <div className="d-flex align-items-center gap-2 mb-4 border-bottom pb-2">
                <Briefcase className="text-primary" size={20}/>
                <h5 className="mb-0 fw-bold text-dark">
                    {isEditing ? 'Edit Experience' : 'Add New Experience'}
                </h5>
            </div>

            {/* Fields */}
            <div className="row g-3">
                <div className="col-md-4">
                    <label htmlFor="exp-title" className="form-label fw-bold small text-uppercase text-muted">Job Title</label>
                    <input id="exp-title" type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., Senior Developer" required className="form-control" />
                </div>
                <div className="col-md-4">
                    <label htmlFor="exp-company" className="form-label fw-bold small text-uppercase text-muted">Company</label>
                    <input id="exp-company" type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="e.g., Acme Inc." required className="form-control"/>
                </div>
                <div className="col-md-4">
                    <label htmlFor="exp-location" className="form-label fw-bold small text-uppercase text-muted">Location</label>
                    <input id="exp-location" type="text" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="e.g., Remote or City, ST" className="form-control"/>
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

            {/* --- SKILLS SECTION (RESPONSIVE) --- */}
            <div className="mb-4">
                <div 
                    className="d-flex justify-content-between align-items-center mb-2 cursor-pointer"
                    onClick={handleSkillToggle} 
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
                        {isMobile ? 
                            <ChevronDown size={16}/> : 
                            (showSkillLinker ? <ChevronUp size={16}/> : <ChevronDown size={16}/>)
                        }
                    </button>
                </div>

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
                
                {(isMobile || !showSkillLinker) && (aggregatedSkillIds.length > 0 || aggregatedPendingSkills.length > 0) ? ( 
                    <div 
                        className="bg-light p-3 rounded border cursor-pointer hover-bg-slate-100 transition-all"
                        onClick={handleSkillToggle} 
                    >
                        <SelectedSkillsDisplay
                            allSkills={allSkills}
                            selectedSkillIds={aggregatedSkillIds}
                            pendingSkills={aggregatedPendingSkills}
                        />
                    </div>
                ) : null}
                
                {(isMobile || !showSkillLinker) && !(aggregatedSkillIds.length > 0 || aggregatedPendingSkills.length > 0) ? ( 
                    <div 
                        className="text-muted small fst-italic border border-dashed rounded p-2 text-center cursor-pointer hover:bg-light"
                        onClick={handleSkillToggle} 
                    >
                        Click to link skills...
                    </div>
                ) : null}

            </div>

            {/* --- ACHIEVEMENTS SECTION (RESPONSIVE) --- */}
            <div className="mb-4">
                 <div className="d-flex justify-content-between align-items-center mb-2">
                     <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0">
                        <Award size={16} className="text-amber-500"/> Achievements
                     </label>
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
                     <div 
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
                     <div 
                        className="bg-light p-3 rounded border text-center cursor-pointer hover-bg-slate-100 transition-all"
                        onClick={() => setIsAchievementPanelOpen(true)}
                    >
                        <span className="text-muted small fst-italic">No achievements added. Click to manage.</span>
                     </div>
                 )}
            </div>

            {/* --- PANELS --- */}
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