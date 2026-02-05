// frontend/src/components/applications/CVItemDisplayCard.jsx
import React from 'react';
import { Briefcase, BookOpen, Smile } from 'lucide-react';
import AchievementDisplayGrid from '../cv/AchievementDisplayGrid';
import SelectedSkillsDisplay from '../cv/SelectedSkillsDisplay';

const CVItemDisplayCard = ({
    item,
    itemType,
    allSkills = [],
    allAchievements = [],
    allExperiences = [],
    allEducation = [],
    allHobbies = [],
    highlightText // <--- The match string
}) => {
    
    if (!item) return null;

    // --- SMART HIGHLIGHTING HELPER ---
    const getHighlightedText = (text) => {
        if (!text || !highlightText) return text;
        
        // 1. Break the search string into words (ignore punctuation)
        const words = highlightText.split(/[\s,._-]+/).filter(w => w);
        if (words.length === 0) return text;

        // 2. Escape words for Regex
        const escapedWords = words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
        
        // 3. Create pattern: Word + (any flexible separator) + Word
        // [\s\W]* matches whitespace or non-word chars (comma, slash, etc)
        const patternString = `(${escapedWords.join('[\\s\\W]*')})`;
        const regex = new RegExp(patternString, 'gi');
        
        // 4. Split and reconstruct
        const parts = text.split(regex);

        return (
            <span>
                {parts.map((part, i) => {
                    // Check if this part matches our pattern (case-insensitive)
                    // We use a fresh regex test to be safe
                    const isMatch = new RegExp(`^${patternString}$`, 'i').test(part);
                    
                    return isMatch ? 
                        <mark key={i} className="bg-warning-subtle text-dark fw-bold rounded px-1 border border-warning border-opacity-25">{part}</mark> : 
                        part;
                })}
            </span>
        );
    };

    // --- Helper to resolve references ---
    const resolveReferences = (itemIds, singularId, sourceList) => {
        const uniqueIds = new Set(itemIds || []);
        if (singularId && uniqueIds.size === 0) uniqueIds.add(singularId);
        return Array.from(uniqueIds).map(id => sourceList.find(i => i.id === id)).filter(Boolean);
    };

    // --- Helper to get all achievements ---
    const getAchievements = (achievementIds = []) => {
        if (!achievementIds || achievementIds.length === 0) return [];
        return achievementIds.map(id => allAchievements.find(a => a.id === id)).filter(Boolean);
    };

    const linkedAchievements = (itemType === 'achievement' || itemType === 'achievements') 
        ? [] 
        : getAchievements(item.achievement_ids);

    // --- Helper to get all skills ---
    const allIds = new Set(item.skill_ids || []);
    linkedAchievements.forEach(ach => (ach.skill_ids || []).forEach(id => allIds.add(id)));
    const aggregatedSkillIds = Array.from(allIds);

    // --- Render Header ---
    const renderHeader = () => {
        const title = getHighlightedText(item.title || item.degree || item.name || 'Untitled');
        const subtitle = item.company || item.institution;

        return (
            <div className="mb-3">
                <strong className="fs-5 d-block">{title}</strong>
                {subtitle && <span className="fw-medium fs-6 text-muted">@{subtitle}</span>}
                
                {/* Context Pills for Projects */}
                {(itemType === 'project' || itemType === 'projects') && (
                    <div className="d-flex flex-wrap gap-2 mt-2">
                        {resolveReferences(item.related_experience_ids, item.related_experience_id, allExperiences).map(exp => (
                            <span key={exp.id} className="rounded-pill bg-primary-subtle text-primary border border-primary-subtle small d-flex align-items-center gap-1 fw-bold px-2 py-1">
                                <Briefcase size={12} /> {exp.title}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        );
    };

    return (
        <div className="card shadow-sm h-100 border-0">
            <div className="card-body">
                {renderHeader()}

                {/* Description with Highlighting */}
                {item.description && (
                    <div className="mb-3 small text-secondary" style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>
                        {getHighlightedText(item.description)}
                    </div>
                )}

                {/* Bullets with Highlighting */}
                {item.bullets && (
                    <ul className="mb-3 ps-3 small text-secondary">
                        {item.bullets.split('\n').map((bullet, idx) => (
                            <li key={idx} className="mb-1">{getHighlightedText(bullet.replace(/^•\s*/, ''))}</li>
                        ))}
                    </ul>
                )}

                {/* Achievements */}
                {linkedAchievements.length > 0 && (
                    <div className="mb-3">
                        <h6 className="small fw-bold mb-2 text-muted text-uppercase">Key Achievements</h6>
                        <AchievementDisplayGrid
                            achievementsToDisplay={linkedAchievements}
                            allSkills={allSkills}
                            isDisplayOnly={true}
                        />
                    </div>
                )}

                {/* Skills with Tag Highlighting */}
                {aggregatedSkillIds.length > 0 && (
                    <div className="mt-3 pt-3 border-top">
                        <strong className="form-label d-block mb-2 small text-muted text-uppercase">Related Skills</strong>
                        <div className="d-flex flex-wrap gap-1">
                            {aggregatedSkillIds.map(skillId => {
                                const skill = allSkills.find(s => s.id === skillId);
                                if (!skill) return null;
                                
                                // Check if this skill tag matches the highlight string
                                const isMatch = highlightText && skill.name.toLowerCase().includes(highlightText.toLowerCase());
                                
                                return (
                                    <span 
                                        key={skillId} 
                                        className={`badge border fw-normal ${isMatch ? 'bg-warning text-dark border-warning' : 'bg-light text-dark border-secondary-subtle'}`}
                                    >
                                        {skill.name}
                                    </span>
                                );
                            })}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default CVItemDisplayCard;