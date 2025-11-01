// frontend/src/components/cv/ExperienceForm.jsx
import React, { useState, useEffect } from 'react';
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
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');
    const [startDate, setStartDate] = useState('');
    const [endDate, setEndDate] = useState('');
    const [description, setDescription] = useState('');
    const [selectedSkillIds, setSelectedSkillIds] = useState([]);
    const [pendingSkills, setPendingSkills] = useState([]);
    const [selectedAchievementIds, setSelectedAchievementIds] = useState([]);
    const [pendingAchievements, setPendingAchievements] = useState([]);
    const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
    const [isAchievementModalOpen, setIsAchievementModalOpen] = useState(false);

    const isEditing = Boolean(initialData);

    useEffect(() => {
        if (isEditing) {
            setTitle(initialData.title || '');
            setCompany(initialData.company || '');
            setStartDate(initialData.start_date || '');
            setEndDate(initialData.end_date || '');
            setDescription(initialData.description || '');
            setSelectedSkillIds(initialData.skill_ids || []);
            setSelectedAchievementIds(initialData.achievement_ids || []);
            setPendingSkills([]);
            setPendingAchievements([]);
        } else {
            setTitle('');
            setCompany('');
            setStartDate('');
            setEndDate('');
            setDescription('');
            setSelectedSkillIds([]);
            setPendingSkills([]);
            setSelectedAchievementIds([]);
            setPendingAchievements([]);
        }
    }, [initialData, isEditing]);

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
            existing_achievement_ids: selectedAchievementIds,
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
            setSelectedAchievementIds([]);
            setPendingAchievements([]);
        }
    };

    const existingAchievements = selectedAchievementIds
        .map(id => allAchievements.find(a => a.id === id))
        .filter(Boolean);
    
    const allAchievementsToShow = [...existingAchievements, ...pendingAchievements];


    return (
        <form 
            key={initialData?.id || 'new'} 
            onSubmit={handleSubmit} 
            // Use Bootstrap's card component for the form container
            className="card p-3"
            // Keep the custom top border
            style={{ borderTop: `4px solid #007bff` }}
        >
            {/* Form Title */}
            <h4 className="text-primary mt-0 mb-3">
                {isEditing ? 'Edit Experience' : 'Add New Experience'}
            </h4>

            {/* Title and Company Inputs */}
            <div className="mb-3">
                <label htmlFor="exp-title" className="form-label fw-medium">Job Title</label>
                <input 
                    id="exp-title"
                    type="text" 
                    value={title} 
                    onChange={(e) => setTitle(e.target.value)} 
                    placeholder="e.g., Senior Developer" 
                    required 
                    className="form-control" 
                />
            </div>
            
            <div className="mb-3">
                <label htmlFor="exp-company" className="form-label fw-medium">Company</label>
                <input 
                    id="exp-company"
                    type="text" 
                    value={company} 
                    onChange={(e) => setCompany(e.target.value)} 
                    placeholder="e.g., Acme Inc." 
                    required 
                    className="form-control"
                />
            </div>

            {/* Date Inputs in a row */}
            <div className="row g-2 mb-3">
                <div className="col-md-6">
                    <label htmlFor="exp-start" className="form-label fw-medium">Start Date</label>
                    <input
                        id="exp-start"
                        type="text"
                        value={startDate}
                        onChange={(e) => setStartDate(e.target.value)}
                        placeholder="e.g., Jan 2020"
                        className="form-control"
                    />
                </div>
                <div className="col-md-6">
                    <label htmlFor="exp-end" className="form-label fw-medium">End Date</label>
                    <input
                        id="exp-end"
                        type="text"
                        value={endDate}
                        onChange={(e) => setEndDate(e.target.value)}
                        placeholder="e.g., Present"
                        className="form-control"
                    />
                </div>
            </div>

            {/* Description Textarea */}
            <div className="mb-3">
                <label htmlFor="exp-desc" className="form-label fw-medium">Description</label>
                <textarea 
                    id="exp-desc"
                    value={description} 
                    onChange={(e) => setDescription(e.target.value)} 
                    placeholder="Brief overview of responsibilities..." 
                    className="form-control"
                    rows="3"
                />
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