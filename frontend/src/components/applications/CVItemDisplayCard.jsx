// frontend/src/components/applications/CVItemDisplayCard.jsx
import React from 'react';
import { Briefcase, BookOpen, Smile } from 'lucide-react'; // <--- Added Icons
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
    allEducation = [],
    allHobbies = [] // <--- 1. Accept Hobbies
}) => {
    
    if (!item) return null;

    // --- Helper to resolve mixed (singular/plural) references ---
    // Matches the logic in ProjectManager to handle legacy data safely
    const resolveReferences = (itemIds, singularId, sourceList) => {
        const uniqueIds = new Set(itemIds || []);
        if (singularId && uniqueIds.size === 0) {
            uniqueIds.add(singularId);
        }
        return Array.from(uniqueIds).map(id => sourceList.find(i => i.id === id)).filter(Boolean);
    };

    // --- Helper to get all achievements ---
    const getAchievements = (achievementIds = []) => {
        if (!achievementIds || achievementIds.length === 0) return [];
        return achievementIds
            .map(id => allAchievements.find(a => a.id === id))
            .filter(Boolean);
    };

    // --- Handle achievements differently if the item *is* an achievement ---
    // (Preserved from old version: Don't show achievements *of* an achievement)
    const linkedAchievements = (itemType === 'achievement' || itemType === 'achievements') 
        ? [] 
        : getAchievements(item.achievement_ids);

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
            case 'experience':
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
            
            // --- UPDATED PROJECT DISPLAY ---
            case 'projects':    
            case 'project':
                // Resolve all contexts (New Lists + Legacy Fallback)
                const linkedExps = resolveReferences(item.related_experience_ids, item.related_experience_id, allExperiences);
                const linkedEdus = resolveReferences(item.related_education_ids, item.related_education_id, allEducation);
                const linkedHobbies = resolveReferences(item.related_hobby_ids, null, allHobbies);

                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block mb-1">{item.title || 'Untitled Project'}</strong>
                        
                        {/* Pretty Context Pills with Company/Institution info */}
                        <div className="d-flex flex-wrap gap-2">
                            {linkedExps.map(exp => (
                                <span key={exp.id} className="rounded-pill bg-blue-50 text-blue-700 border border-blue-200 small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                    <Briefcase size={12} /> 
                                    {exp.title}{exp.company ? ` @ ${exp.company}` : ''}
                                </span>
                            ))}
                            {linkedEdus.map(edu => (
                                <span key={edu.id} className="rounded-pill bg-indigo-50 text-indigo-700 border border-indigo-200 small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                    <BookOpen size={12} /> 
                                    {edu.degree}{edu.institution ? ` @ ${edu.institution}` : ''}
                                </span>
                            ))}
                            {linkedHobbies.map(hobby => (
                                <span key={hobby.id} className="rounded-pill bg-pink-50 text-pink-600 border border-pink-200 small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                    <Smile size={12} /> {hobby.name}
                                </span>
                            ))}
                        </div>
                    </div>
                );

            case 'hobbies':
                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block">{item.name || 'Untitled Hobby'}</strong>
                    </div>
                );
            
            // --- Case for Skills ---
            case 'skills':
            case 'skill':
                return (
                    <div className="mb-2">
                        <strong className="fs-5 d-block">{item.name || 'Untitled Skill'}</strong>
                        {item.category && <span className="badge bg-primary me-2 text-uppercase">{item.category}</span>}
                        {item.level && <span className="text-muted fw-medium">Level: {item.level}</span>}
                    </div>
                );

            // --- Case for Achievements ---
            case 'achievements':
            case 'achievement':
                return (
                    <div className="mb-2">
                        <p className="fs-5 fw-medium mb-1">{item.text || 'Untitled Achievement'}</p>
                        {item.context && <span className="badge bg-secondary">Context: {item.context}</span>}
                    </div>
                );
            
            default:
                return <strong className="fs-5 d-block">Preview</strong>;
        }
    };

    return (
        <div className="card shadow-sm h-100">
            <div className="card-body">
                {renderHeader()}

                {/* Description */}
                {item.description && (
                    <p className="mb-2 mt-2 small text-secondary" style={{ whiteSpace: 'pre-wrap' }}>
                        {item.description}
                    </p>
                )}

                {/* Hide if item is an Achievement (Preserved logic) */}
                {linkedAchievements.length > 0 && itemType !== 'achievement' && itemType !== 'achievements' && (
                    <div className="mb-3">
                        <h6 className="small fw-bold mb-0 text-muted text-uppercase">Key Achievements:</h6>
                        <AchievementDisplayGrid
                            achievementsToDisplay={linkedAchievements}
                            allSkills={allSkills}
                            isDisplayOnly={true}
                        />
                    </div>
                )}

                {/* Hide if item is a Skill (Preserved logic) */}
                {aggregatedSkillIds.length > 0 && itemType !== 'skill' && itemType !== 'skills' && (
                    <div className="mt-2 pt-2 border-top">
                        <strong className="form-label d-block mb-2 small text-muted">Related Skills:</strong>
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