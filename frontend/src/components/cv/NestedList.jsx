import React from 'react';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const NestedList = ({
    cvId,
    items,
    title,
    listName,
    onDelete,
    onEdit,
    allSkills = [],
    allAchievements = [],
    isDisplayOnly = false
}) => {

    const getSkillNames = (skillIds = []) => {
        // ... (helper function remains the same)
        if (!skillIds || skillIds.length === 0) return '';
        return skillIds
            .map(id => allSkills.find(s => s.id === id)?.name)
            .filter(name => name)
            .join(', ');
    };

    const getAchievements = (achievementIds = []) => {
        // ... (helper function remains the same)
         if (!achievementIds || achievementIds.length === 0) return [];
        return achievementIds
            .map(id => allAchievements.find(a => a.id === id))
            .filter(ach => ach);
    };

    // Define a base text color that should work on light backgrounds
    const baseTextColor = '#333'; // A dark grey

    return (
        <div style={{ padding: '10px' }}>
            {(!isDisplayOnly || (items && items.length > 0)) && (
                 <h3 style={{ color: baseTextColor }}>{title} ({items?.length || 0})</h3>
            )}

            {!items || items.length === 0 ? (
                 !isDisplayOnly && <p style={{ fontStyle: 'italic', color: '#666' }}>No {listName} added yet.</p> // Keep color for muted text
            ) : (
                <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                    {items.map((item) => {
                        const linkedSkillNames = getSkillNames(item.skill_ids);
                        const linkedAchievements = getAchievements(item.achievement_ids);

                        return (
                            <li key={item.id} style={{
                                border: isDisplayOnly ? 'none' : '1px solid #ddd',
                                borderBottom: '1px solid #eee',
                                marginBottom: '15px',
                                padding: isDisplayOnly ? '10px 0' : '15px',
                                borderRadius: isDisplayOnly ? '0' : '5px',
                                backgroundColor: '#fff', // White background
                                boxShadow: isDisplayOnly ? 'none' : '0 1px 3px rgba(0,0,0,0.1)',
                                color: baseTextColor // Set base text color for the list item
                             }}>

                                {/* Header */}
                                <div style={{ marginBottom: '8px' }}>
                                    {listName === 'experiences' ? (
                                        <>
                                            <strong style={{ fontSize: '1.1em', display: 'block' /* Removed color */ }}>
                                                {item.title || 'Untitled Experience'}
                                            </strong>
                                            {item.company && (
                                                <span style={{ fontWeight: 'bold', fontSize: '1em', color: '#555' }}> {/* Keep slightly lighter color */}
                                                    @{item.company}
                                                </span>
                                            )}
                                            {(item.start_date || item.end_date) && (
                                                <span style={{ marginLeft: item.company ? ' 10px' : '0', fontSize: '0.9em', color: '#777' }}> {/* Keep muted color */}
                                                    ({item.start_date || '?'} – {item.end_date || 'Present'})
                                                </span>
                                            )}
                                        </>
                                    ) : (
                                        <strong style={{ fontSize: '1.1em', display: 'block' /* Removed color */ }}>
                                            {item.title || item.name || item.text || 'Untitled Item'}
                                        </strong>
                                    )}
                                </div>

                                {/* Description */}
                                {item.description && (
                                    <p style={{ margin: '0 0 8px 0', fontSize: '0.95em', whiteSpace: 'pre-wrap' /* Removed color */ }}>
                                        {item.description}
                                    </p>
                                )}

                                {/* Achievements Grid */}
                                {linkedAchievements.length > 0 && (
                                    <AchievementDisplayGrid
                                        achievementsToDisplay={linkedAchievements}
                                        allSkills={allSkills}
                                        isDisplayOnly={true}
                                        // Pass baseTextColor if needed inside AchievementDisplayGrid
                                    />
                                )}

                                {/* Related Skills */}
                                {linkedSkillNames && (
                                    <p style={{ margin: '8px 0 0 0', fontSize: '0.85em', fontStyle: 'italic', color: '#666' }}> {/* Keep muted color */}
                                        Related Skills: {linkedSkillNames}
                                    </p>
                                )}

                                {/* Action Buttons */}
                                {!isDisplayOnly && (
                                    <div style={{ marginTop: '10px', borderTop: '1px solid #eee', paddingTop: '10px' }}>
                                        {onEdit && (
                                            <button onClick={() => onEdit(item)} style={{ /* Keep button styles */ marginRight: '10px', backgroundColor: '#ffc107', color: '#333', padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '0.8em' }}>
                                                Edit
                                            </button>
                                        )}
                                        {onDelete && (
                                            <button onClick={() => onDelete(cvId, item.id, listName)} style={{ /* Keep button styles */ backgroundColor: '#dc3545', color: 'white', padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '0.8em' }}>
                                                Delete
                                            </button>
                                        )}
                                    </div>
                                )}
                            </li>
                        );
                    })}
                </ul>
            )}
        </div>
    );
};

export default NestedList;