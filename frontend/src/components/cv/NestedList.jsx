// frontend/src/components/cv/NestedList.jsx
import React from 'react';
import AchievementDisplayGrid from './AchievementDisplayGrid';
// 1. Import the new component
import AggregatedSkillsDisplay from './AggregatedSkillsDisplay';

const NestedList = ({
    cvId,
    items,
    title,
    listName,
    onDelete,
    onEdit,
    allSkills = [], // Still needed for AchievementDisplayGrid
    allAchievements = [],
    isDisplayOnly = false
}) => {

    // 2. The getSkillNames helper function is no longer needed here.
    /*
    const getSkillNames = (skillIds = []) => {
        // ... (REMOVED) ...
    };
    */

    // This function is still used for the Achievement grid.
    const getAchievements = (achievementIds = []) => {
         if (!achievementIds || achievementIds.length === 0) return [];
        return achievementIds
            .map(id => allAchievements.find(a => a.id === id))
            .filter(ach => ach);
    };

    return (
        <div>
            {(!isDisplayOnly || (items && items.length > 0)) && (
                 <h3 className="h4 text-capitalize">{title} ({items?.length || 0})</h3>
            )}

            {!items || items.length === 0 ? (
                 !isDisplayOnly && <p className="text-muted fst-italic">No {listName} added yet.</p>
            ) : (
                <ul className="list-group list-group-flush">
                    {items.map((item) => {
                        // 3. We no longer calculate linkedSkillNames here
                        const linkedAchievements = getAchievements(item.achievement_ids);

                        return (
                            <li 
                                key={item.id} 
                                className="list-group-item p-3 mb-3 border shadow-sm rounded"
                            >
                                {/* Header */}
                                <div className="mb-2">
                                    {listName === 'experiences' ? (
                                        <>
                                            <strong className="fs-5 d-block">
                                                {item.title || 'Untitled Experience'}
                                            </strong>
                                            {item.company && (
                                                <span className="fw-medium fs-6 text-muted">
                                                    @{item.company}
                                                </span>
                                            )}
                                            {(item.start_date || item.end_date) && (
                                                <span className="ms-2 small text-muted text-uppercase">
                                                    ({item.start_date || '?'} – {item.end_date || 'Present'})
                                                </span>
                                            )}
                                        </>
                                    ) : (
                                        <strong className="fs-5 d-block">
                                            {item.title || item.name || item.text || 'Untitled Item'}
                                        </strong>
                                    )}
                                </div>

                                {/* Description */}
                                {item.description && (
                                    <p className="mb-2" style={{ whiteSpace: 'pre-wrap' }}>
                                        {item.description}
                                    </p>
                                )}

                                {/* Achievements Grid */}
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

                                {/* 4. Replaced the old skill paragraph with the new component */}
                                <div className="mt-2 pt-2 border-top">
                                    <AggregatedSkillsDisplay
                                        cvId={cvId}
                                        listName={listName}
                                        itemId={item.id}
                                    />
                                </div>
                                
                                {/* Action Buttons */}
                                {!isDisplayOnly && (
                                    <div className="mt-3 border-top pt-3 text-end">
                                        {onEdit && (
                                            <button onClick={() => onEdit(item)} className="btn btn-warning btn-sm me-2">
                                                Edit
                                            </button>
                                        )}
                                        {onDelete && (
                                            <button onClick={() => onDelete(cvId, item.id, listName)} className="btn btn-danger btn-sm">
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