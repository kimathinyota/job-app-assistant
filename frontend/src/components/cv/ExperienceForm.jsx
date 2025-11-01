// frontend/src/components/cv/ExperienceForm.jsx
import React, { useState, useEffect } from 'react';
import { fetchAggregatedSkills } from '../../api/cvClient'; // Keep this import
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
    
    // Skill state (unchanged)
    const [selectedSkillIds, setSelectedSkillIds] = useState([]);
    const [pendingSkills, setPendingSkills] = useState([]);

    // *** 1. STATE CHANGE FOR ACHIEVEMENTS ***
    // We now store full objects, not just IDs, to allow local modification
    const [linkedExistingAchievements, setLinkedExistingAchievements] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    
    // Modal state (unchanged)
    const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
    const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

    const isEditing = Boolean(initialData);

    useEffect(() => {
        if (isEditing) {
            // Populate form fields
            setTitle(initialData.title || '');
            setCompany(initialData.company || '');
            setStartDate(initialData.start_date || '');
            setEndDate(initialData.end_date || '');
            setDescription(initialData.description || '');
            
            // Load aggregated skills for the Skill modal
            const loadAggregatedSkills = async () => {
                try {
                    const aggSkills = await fetchAggregatedSkills(cvId, 'experiences', initialData.id);
                    setSelectedSkillIds(aggSkills.map(s => s.id));
                } catch (err) {
                    console.error("Failed to load aggregated skills, falling back to direct skills.", err);
                    setSelectedSkillIds(initialData.skill_ids || []);
                }
            };
            loadAggregatedSkills();

            // *** 2. UPDATE: Populate new state with full objects ***
            // Find the full achievement objects from the master list
            const initialAchievements = (initialData.achievement_ids || [])
                .map(id => allAchievements.find(a => a.id === id))
                .filter(Boolean)
                .map(ach => ({ ...ach })); // Create copies
            setLinkedExistingAchievements(initialAchievements);

            // Reset pending items
            setPendingSkills([]);
            setPendingAchievements([]);
        } else {
            // Reset form for "create new"
            setTitle('');
            setCompany('');
            setStartDate('');
            setEndDate('');
            setDescription('');
            setSelectedSkillIds([]);
            setPendingSkills([]);
            // *** 3. UPDATE: Reset new state variable ***
            setLinkedExistingAchievements([]);
            setPendingAchievements([]);
        }
    }, [initialData, isEditing, cvId, allAchievements]); // Added allAchievements

    // *** 4. NEW: Wrapper function for the Achievement Modal ***
    // This translates the ID list from the modal back into a list of objects
    const handleExistingAchievementSelection = (newIdList) => {
        // Find newly added IDs
        const newIds = newIdList.filter(id => !linkedExistingAchievements.some(a => a.id === id));
        // Find removed IDs
        const removedIds = linkedExistingAchievements.map(a => a.id).filter(id => !newIdList.includes(id));

        let newList = [...linkedExistingAchievements];
        
        // Remove deselected ones
        newList = newList.filter(a => !removedIds.includes(a.id));
        
        // Add newly selected ones (as full objects)
        newIds.forEach(id => {
            const ach = allAchievements.find(a => a.id === id);
            if (ach) {
                // Add a *copy* so we can modify it locally
                newList.push({ ...ach }); 
            }
        });
        setLinkedExistingAchievements(newList);
    };

    // *** 5. UPDATED: The Skill Selection handler now updates BOTH lists ***
    const handleSkillSelectionChange = (newSkillIdList) => {
        // Find which skills were *removed* from the main list
        const removedSkillIds = selectedSkillIds.filter(id => !newSkillIdList.includes(id));

        if (removedSkillIds.length > 0) {
            // 1. Update pending (like before)
            setPendingAchievements(prevPending => 
                prevPending.map(ach => ({
                    ...ach,
                    skill_ids: (ach.skill_ids || []).filter(id => !removedSkillIds.includes(id))
                }))
            );
            
            // 2. NEW: Update linked existing achievements (our local copies)
            setLinkedExistingAchievements(prevLinked =>
                prevLinked.map(ach => ({
                    ...ach,
                    // Filter out the removed skill IDs from each object
                    skill_ids: (ach.skill_ids || []).filter(id => !removedSkillIds.includes(id))
                }))
            );
        }

        // Finally, update the main skill list
        setSelectedSkillIds(newSkillIdList);
    };

    // *** 6. UPDATED: handleSubmit now reads from the new state ***
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim() || !company.trim()) return;

        const dataToSend = {
            title,
            company,
            start_date: startDate || null,
            end_date: endDate || null,
            description: description || null,
            existing_skill_ids: selectedSkillIds,
            new_skills: pendingSkills,
            // Send the IDs from our local object list
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
            setSelectedSkillIds([]);
            setPendingSkills([]);
            setLinkedExistingAchievements([]); // Reset new state
            setPendingAchievements([]);
        }
    };

    // *** 7. UPDATED: This display list now uses the new state ***
    // This will show the locally modified versions
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

            {/* Form fields are unchanged */}
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

            {/* --- SKILLS Section --- */}
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
                    selectedSkillIds={selectedSkillIds}
                    pendingSkills={pendingSkills}
                />
            </div>

            {/* --- ACHIEVEMENTS Section --- */}
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
                     // This now correctly passes our modified local list
                     achievementsToDisplay={allAchievementsToShow}
                     allSkills={allSkills}
                     isDisplayOnly={true}
                 />
            </div>

            {/* --- Modals --- */}
            <SkillManagerModal
                isOpen={isSkillModalOpen}
                onClose={() => setIsSkillModalOpen(false)}
                allSkills={allSkills}
                selectedSkillIds={selectedSkillIds}
                // *** 8. UPDATE: Pass the new handler ***
                setSelectedSkillIds={handleSkillSelectionChange}
                pendingSkills={pendingSkills}
                setPendingSkills={setPendingSkills}
            />
             <AchievementManagerModal
                 isOpen={isAchievementModalOpen}
                 onClose={() => setIsAchievementModalOpen(false)}
                 allAchievements={allAchievements}
                 // *** 9. UPDATE: Pass IDs from new state and the new handler ***
                 selectedAchievementIds={linkedExistingAchievements.map(a => a.id)}
                 setSelectedAchievementIds={handleExistingAchievementSelection}
                 pendingAchievements={pendingAchievements}
                 setPendingAchievements={setPendingAchievements}
                 allSkills={allSkills}
             />

            {/* --- Action Buttons --- */}
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