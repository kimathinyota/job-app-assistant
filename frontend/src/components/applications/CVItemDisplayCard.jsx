// frontend/src/components/cv/CVItemDisplayCard.jsx
import React from 'react';
import AchievementDisplayGrid from '../cv/AchievementDisplayGrid';
import SelectedSkillsDisplay from '../cv/SelectedSkillsDisplay';

// This component renders the full details of any single CV item
const CVItemDisplayCard = ({
    item,
    itemType,
    allSkills = [],
    allAchievements = [],
    // These are needed to resolve names for linked projects
    allExperiences = [],
    allEducation = []
}) => {
    
    if (!item) return null;

    // --- Helper functions to get related item names (for Projects) ---
    const getRelatedExperienceName = (id) => {
        if (!id) return null;
        const exp = allExperiences.find(e => e.id === id);
        return exp ? `${exp.title} @ ${exp.company}` : 'Unknown Experience';
    };
    
    const getRelatedEducationName = (id) => {
        if (!id) return null;
        const edu = allEducation.find(e => e.id === id);
        return edu ? `${edu.degree} @ ${edu.institution}` : 'Unknown Education';
    };

    // --- Helper to get all achievements ---
    const getAchievements = (achievementIds = []) => {
        if (!achievementIds || achievementIds.length === 0) return [];
        return achievementIds
            .map(id => allAchievements.find(a => a.id === id))
            .filter(Boolean);
    };

    const linkedAchievements = getAchievements(item.achievement_ids);

    // --- Helper to get all skills (direct + from achievements) ---
    const allIds = new Set(item.skill_ids || []);
    linkedAchievements.forEach(ach => {
        (ach.skill_ids || []).forEach(id => allIds.add(id));
    });
    const aggregatedSkillIds = Array.from(allIds);

    // --- Render Header based on itemType ---
    const renderHeader = () => {
        switch(itemType) {
            case 'experiences':
                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block">{item.title || 'Untitled Experience'}</strong>
                        {item.company && <span className="fw-medium fs-6 text-muted">@{item.company}</span>}
                        {(item.start_date || item.end_date) && (
                            <span className="ms-2 small text-muted text-uppercase">
                                ({item.start_date || '?'} – {item.end_date || 'Present'})
                            </span>
                        )}
                    </div>
                );
            case 'education':
                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block">{item.degree || 'Untitled Education'}</strong>
                        {item.institution && <span className="fw-medium fs-6 text-muted">@{item.institution}</span>}
                        {item.field && <span className="ms-2 small text-muted">({item.field})</span>}
                        {(item.start_date || item.end_date) && (
                            <span className="ms-2 small text-muted text-uppercase">
                                ({item.start_date || '?'} – {item.end_date || 'Present'})
                            </span>
                        )}
                    </div>
                );
            case 'projects':
                const relatedExpName = getRelatedExperienceName(item.related_experience_id);
                const relatedEduName = getRelatedEducationName(item.related_education_id);
                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block">{item.title || 'Untitled Project'}</strong>
                        {relatedExpName && <span className="badge bg-secondary me-2">Relates to: {relatedExpName}</span>}
                        {relatedEduName && <span className="badge bg-info text-dark">Relates to: {relatedEduName}</span>}
                    </div>
                );
            case 'hobbies':
                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block">{item.name || 'Untitled Hobby'}</strong>
                    </div>
                );
            default:
                return <strong className="fs-5 d-block">Preview</strong>;
        }
    };

    return (
        <div className="card shadow-sm">
            <div className="card-body">
                {renderHeader()}

                {item.description && (
                    <p className="mb-2 mt-2" style={{ whiteSpace: 'pre-wrap' }}>
                        {item.description}
                    </p>
                )}

                {linkedAchievements.length > 0 && (
                    <div className="mb-3">
                        <h6 className="small fw-bold mb-0">Key Achievements:</h6>
                        <AchievementDisplayGrid
                            achievementsToDisplay={linkedAchievements}
                            allSkills={allSkills}
                            isDisplayOnly={true}
                        />
                    </div>
                )}

                {aggregatedSkillIds.length > 0 && (
                    <div className="mt-2 pt-2 border-top">
                        <strong className="form-label d-block mb-2">Related Skills:</strong>
                        <SelectedSkillsDisplay
                            allSkills={allSkills}
                            selectedSkillIds={aggregatedSkillIds}
                            pendingSkills={[]}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};

export default CVItemDisplayCard;