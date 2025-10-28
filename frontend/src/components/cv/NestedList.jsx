import React from 'react';
import AchievementDisplayGrid from './AchievementDisplayGrid'; // Import the updated component

// Added allSkills, allAchievements, and isDisplayOnly props
const NestedList = ({
    cvId,
    items,
    title,
    listName,
    onDelete,
    onEdit,
    allSkills = [],
    allAchievements = [],
    isDisplayOnly = false // Default to false for the CV Manager context
}) => {

    // Helper function to get skill names from IDs
    const getSkillNames = (skillIds = []) => {
        if (!skillIds || skillIds.length === 0) return ''; // Return empty string if no IDs
        return skillIds
            .map(id => allSkills.find(s => s.id === id)?.name)
            .filter(name => name)
            .join(', ');
    };

    // Helper function to get achievement objects from IDs
     const getAchievements = (achievementIds = []) => {
        if (!achievementIds || achievementIds.length === 0) return []; // Return empty array if no IDs
        return achievementIds
            .map(id => allAchievements.find(a => a.id === id))
            .filter(ach => ach);
    };


    return (
        <div style={{ padding: '10px' }}>
            {(!isDisplayOnly || (items && items.length > 0)) && (
                 <h3>{title} ({items?.length || 0})</h3>
            )}

            {!items || items.length === 0 ? (
                 !isDisplayOnly && <p style={{ color: '#666', fontStyle: 'italic' }}>No {listName} added yet.</p>
            ) : (
                <ul style={{ listStyle: 'none', paddingLeft: 0 }}>
                    {items.map((item) => {
                        const linkedSkillNames = getSkillNames(item.skill_ids);
                        const linkedAchievements = getAchievements(item.achievement_ids);

                        return (
                            <li key={item.id} style={{ border: isDisplayOnly ? 'none' : '1px solid #ddd', borderBottom: '1px solid #eee', marginBottom: '15px', padding: isDisplayOnly ? '10px 0' : '15px', borderRadius: isDisplayOnly ? '0' : '5px', backgroundColor: '#fff', boxShadow: isDisplayOnly ? 'none' : '0 1px 3px rgba(0,0,0,0.1)' }}>

                                {/* --- Header --- */}
                                <div style={{ marginBottom: '8px' }}>
                                    {/* ... (Conditional header logic remains the same) ... */}
                                     {listName === 'experiences' ? (
                                        <>
                                            <strong style={{ fontSize: '1.1em', display: 'block' }}>
                                                {item.title || 'Untitled Experience'}
                                            </strong>
                                            {item.company && (
                                                <span style={{ fontWeight: 'bold', color: '#555', fontSize: '1em' }}>
                                                    @{item.company}
                                                </span>
                                            )}
                                            {(item.start_date || item.end_date) && (
                                                <span style={{ marginLeft: item.company ? ' 10px' : '0', color: '#777', fontSize: '0.9em' }}>
                                                    ({item.start_date || '?'} – {item.end_date || 'Present'})
                                                </span>
                                            )}
                                        </>
                                    ) : ( // General format for others (simplified for display)
                                        <strong style={{ fontSize: '1.1em', display: 'block' }}>
                                            {item.title || item.name || item.text || 'Untitled Item'}
                                        </strong>
                                    )}
                                </div>

                                {/* --- Description --- */}
                                {item.description && (
                                    <p style={{ margin: '0 0 8px 0', fontSize: '0.95em', color: '#333', whiteSpace: 'pre-wrap' }}>
                                        {item.description}
                                    </p>
                                )}

                                {/* --- USE AchievementDisplayGrid for Linked Achievements --- */}
                                {linkedAchievements.length > 0 && (
                                    <AchievementDisplayGrid
                                        achievementsToDisplay={linkedAchievements} // Pass the found achievements
                                        allSkills={allSkills} // Pass skills for lookup within the grid
                                        isDisplayOnly={true} // Set to true for pure display
                                        // No onEdit or onRemove needed here
                                    />
                                )}

                                {/* --- Linked Skills (directly on experience) --- */}
                                {/* Show this only if there are skills NOT already covered by achievements? Or always show? Let's always show for now. */}
                                {linkedSkillNames && (
                                    <p style={{ margin: '8px 0 0 0', fontSize: '0.85em', color: '#666', fontStyle: 'italic' }}>
                                        Related Skills: {linkedSkillNames}
                                    </p>
                                )}

                                {/* --- Action Buttons (Conditionally Rendered) --- */}
                                {!isDisplayOnly && (
                                    <div style={{ marginTop: '10px', borderTop: '1px solid #eee', paddingTop: '10px' }}>
                                        {onEdit && (
                                            <button
                                                onClick={() => onEdit(item)}
                                                style={{ marginRight: '10px', backgroundColor: '#ffc107', color: '#333', padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '0.8em' }}
                                            >
                                                Edit
                                            </button>
                                        )}
                                        {onDelete && (
                                            <button
                                                onClick={() => onDelete(cvId, item.id, listName)}
                                                style={{ backgroundColor: '#dc3545', color: 'white', padding: '5px 10px', borderRadius: '3px', border: 'none', cursor: 'pointer', fontSize: '0.8em' }}
                                            >
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