// frontend/src/components/cv/ExperienceForm.jsx
import React, { useState, useEffect } from 'react'; // Import useEffect
import SkillManagerModal from './SkillManagerModal';
import SelectedSkillsDisplay from './SelectedSkillsDisplay';
// Assuming Achievement linking might be added later, import Achievement components
import AchievementManagerModal from './AchievementManagerModal';
import AchievementDisplayGrid from './AchievementDisplayGrid';

// Add initialData and onCancelEdit props
const ExperienceForm = ({
    onSubmit, // This will now handle BOTH create and update
    cvId,
    allSkills,
    allAchievements, // Add this prop for Achievement linking
    initialData, // The experience object to edit, or null for creating new
    onCancelEdit // Function to call when cancelling an edit
}) => {
    // State for form fields
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [description, setDescription] = useState('');
    // State for linked items
    const [selectedSkillIds, setSelectedSkillIds] = useState([]);
    const [pendingSkills, setPendingSkills] = useState([]);
    const [selectedAchievementIds, setSelectedAchievementIds] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    // State for modals
    const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
    const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

    const isEditing = Boolean(initialData); // Flag to check if we are editing

    // --- NEW: useEffect to populate form when initialData changes ---
    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setCompany(initialData.company || '');
            setDescription(initialData.description || '');
            // IMPORTANT: Assume initialData contains the *IDs* of already linked skills/achievements
            setSelectedSkillIds(initialData.skill_ids || []);
            setSelectedAchievementIds(initialData.achievement_ids || []);
            // Reset pending items when starting an edit
            setPendingSkills([]);
            setPendingAchievements([]);
        } else {
            // Reset form when switching from edit to create (e.g., after saving or cancelling)
            setTitle('');
            setCompany('');
            setDescription('');
            setSelectedSkillIds([]);
            setPendingSkills([]);
            setSelectedAchievementIds([]);
            setPendingAchievements([]);
        }
    }, [initialData, isEditing]); // Rerun effect if initialData changes

    // --- Modified handleSubmit ---
    const handleSubmit = (e) => {
        e.preventDefault();
        if (!title.trim() || !company.trim()) return;

        const dataToSend = {
            title,
            company,
            description: description || null, // Send null if empty for backend clarity
            // Use different keys for update vs create is complex here.
            // Let's rely on the parent (CVManagerPage) to handle the API call difference.
            // Send ALL skill/achievement info regardless of mode.
            existing_skill_ids: selectedSkillIds,
            new_skills: pendingSkills,
            existing_achievement_ids: selectedAchievementIds,
            new_achievements: pendingAchievements, // Assuming you add achievement handling later
        };

        // If editing, include the ID for the parent handler
        if (isEditing) {
            dataToSend.id = initialData.id;
        }

        onSubmit(cvId, dataToSend, 'Experience'); // Pass the combined data and let parent decide API call

        // Reset only if CREATING a new one. Edit state is reset by parent via initialData prop change.
        if (!isEditing) {
            setTitle('');
            setCompany('');
            setDescription('');
            setSelectedSkillIds([]);
            setPendingSkills([]);
            setSelectedAchievementIds([]);
            setPendingAchievements([]);
        }
    };

    return (
        // Added key={initialData?.id || 'new'} to help React reset state if needed, though useEffect handles it
        <form key={initialData?.id || 'new'} onSubmit={handleSubmit} style={{ margin: '15px 0', padding: '15px', border: '1px solid #007bff', borderRadius: '8px', backgroundColor: '#f0f8ff', textAlign: 'left' }}>
            <h3 style={{ color: '#007bff', marginBottom: '15px', marginTop: 0 }}>
                {isEditing ? 'Edit Experience' : '+ Add New Experience'}
            </h3>

            {/* Input fields remain the same */}
            <input type="text" value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Job Title" required style={{ display: 'block', width: '95%', marginBottom: '10px', padding: '8px' }} />
            <input type="text" value={company} onChange={(e) => setCompany(e.target.value)} placeholder="Company" required style={{ display: 'block', width: '95%', marginBottom: '10px', padding: '8px' }} />
            <textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Description..." style={{ display: 'block', width: '95%', padding: '8px', minHeight: '60px', marginBottom: '10px' }} />

            {/* --- SKILLS Section --- */}
            <div style={{ marginTop: '15px' }}>
                <strong style={{ display: 'block', marginBottom: '5px' }}>Skills:</strong>
                <button type="button" onClick={() => setIsSkillModalOpen(true)} style={{ /* styles */ }}>
                    Manage Skills
                </button>
                <SelectedSkillsDisplay
                    allSkills={allSkills}
                    selectedSkillIds={selectedSkillIds}
                    pendingSkills={pendingSkills}
                />
            </div>

            {/* --- ACHIEVEMENTS Section (Placeholder/Future) --- */}
            <div style={{ marginTop: '15px' }}>
                 <strong style={{ display: 'block', marginBottom: '5px' }}>Achievements:</strong>
                 <button type="button" onClick={() => setIsAchievementModalOpen(true)} style={{ /* styles */ }}>
                     Manage Achievements
                 </button>
                 <AchievementDisplayGrid
                     allSkills={allSkills} // Pass skills for tag display in grid
                     allAchievements={allAchievements} // Pass achievements for tag display in grid
                     selectedAchievementIds={selectedAchievementIds} // Show selected existing ones
                     pendingAchievements={pendingAchievements} // Show pending ones
                     // Add handlers for removing/editing pending achievements if needed in display
                 />
            </div>


            {/* --- Modals --- */}
            <SkillManagerModal
                isOpen={isSkillModalOpen}
                onClose={() => setIsSkillModalOpen(false)}
                allSkills={allSkills}
                selectedSkillIds={selectedSkillIds}
                setSelectedSkillIds={setSelectedSkillIds}
                pendingSkills={pendingSkills}
                setPendingSkills={setPendingSkills}
            />
             <AchievementManagerModal
                 isOpen={isAchievementModalOpen}
                 onClose={() => setIsAchievementModalOpen(false)}
                 allAchievements={allAchievements}
                 selectedAchievementIds={selectedAchievementIds}
                 setSelectedAchievementIds={setSelectedAchievementIds}
                 pendingAchievements={pendingAchievements}
                 setPendingAchievements={setPendingAchievements} // Pass setter
                 allSkills={allSkills} // Pass skills for AchievementForm inside modal
                 // Note: AchievementManagerModal needs internal AchievementForm logic
             />

            {/* --- Action Buttons --- */}
            <div style={{ marginTop: '20px' }}>
                <button type="submit" style={{ /* styles */ }}>
                    {isEditing ? 'Save Changes' : 'Add Experience'}
                </button>
                {/* Show Cancel button only when editing */}
                {isEditing && (
                    <button type="button" onClick={onCancelEdit} style={{ marginLeft: '10px', backgroundColor: '#6c757d', color: 'white', /* styles */ }}>
                        Cancel Edit
                    </button>
                )}
            </div>
        </form>
    );
};
export default ExperienceForm;