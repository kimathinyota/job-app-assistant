import React from 'react';
import { Edit2, X, Award, Link as LinkIcon, Plus, Copy } from 'lucide-react';

// A new sub-component to render the status badge
const StatusBadge = ({ achievement, isDisplayOnly }) => {
    if (achievement.isPending) {
        const isModified = !!achievement.original_id;
        return (
            <span 
                className="badge d-flex align-items-center gap-1 bg-warning-subtle text-warning-emphasis border border-warning-subtle"
            >
                {/* --- FIX IS HERE --- */}
                {isModified ? <Copy size={12} /> : <Plus size={12} />}
                {isModified ? 'Modified' : 'New'} 
                {/* --- END FIX --- */}
            </span>
        );
    }

    // It's a "Linked" item.
    // Only show the badge if we are NOT in display-only mode (i.e., we are inside the panel).
    if (!isDisplayOnly) {
        return (
            <span 
                className="badge d-flex align-items-center gap-1 bg-primary-subtle text-primary-emphasis border border-primary-subtle"
            >
                <LinkIcon size={12} />
                Linked
            </span>
        );
    }

    // In display-only mode, don't show a badge for linked items.
    return null; 
};


const AchievementDisplayGrid = ({
  achievementsToDisplay = [],
  allSkills = [],
  onRemoveAchievement,
  onEditAchievement,
  isDisplayOnly = false 
}) => {

  if (!achievementsToDisplay || achievementsToDisplay.length === 0) {
    // Return null instead of a message, the parent component will handle the empty state
    return null; 
  }

  const getSkillName = (id) => {
    const skill = allSkills.find((s) => s.id === id);
    return skill ? skill.name : id;
  };

  return (
    <div className="row g-3 mt-1">
      {achievementsToDisplay.map((ach) => ( 
        <div key={ach.id || ach.index} className="col-12 col-md-6">
            <div 
                className={`
                    card border-0 shadow-sm h-100 group hover-lift transition-all bg-white
                `}
                style={ach.isPending ? {
                    borderStyle: 'dashed',
                    borderWidth: '1px',
                    borderColor: 'var(--bs-warning-border-subtle)',
                    opacity: 0.9
                } : {}}
            >
                <div className="card-body p-3 position-relative">
                    
                    {/* Actions */}
                    {!isDisplayOnly && (
                        <div className="position-absolute top-0 end-0 p-2 d-flex gap-1">
                            {onEditAchievement && (
                                <button
                                    type="button"
                                    onClick={() => onEditAchievement(ach)}
                                    className="btn btn-light btn-sm text-primary p-1"
                                    title="Edit"
                                    style={{ width: '24px', height: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                                >
                                    <Edit2 size={12} />
                                </button>
                            )}
                            {onRemoveAchievement && (
                                <button
                                    type="button"
                                    onClick={() => onRemoveAchievement(ach)}
                                    className="btn btn-light btn-sm text-danger p-1"
                                    title="Remove"
                                    style={{ width: '24px', height: '24px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                                >
                                    <X size={14} />
                                </button>
                            )}
                        </div>
                    )}

                    {/* Content */}
                    <div className="d-flex gap-3">
                        <div className="mt-1 flex-shrink-0">
                            <div className={`p-2 rounded-circle ${ach.isPending ? 'bg-light' : 'bg-amber-50 text-amber-600'}`}>
                                <Award size={16} className={ach.isPending ? 'text-muted' : ''} />
                            </div>
                        </div>
                        <div className="flex-grow-1" style={{ paddingRight: isDisplayOnly ? '0' : '3rem' }}>
                            
                            {/* Status Badge */}
                            <div className="mb-2">
                                <StatusBadge achievement={ach} isDisplayOnly={isDisplayOnly} />
                            </div>

                            <p className="mb-2 small text-dark fw-medium lh-sm">
                                {ach.text}
                            </p>
                            
                            {ach.context && (
                                <span className="d-inline-block px-2 py-1 rounded-pill bg-light text-secondary border small mb-2" style={{fontSize: '0.75rem'}}>
                                    Context: {ach.context}
                                </span>
                            )}

                            {/* Skills (FIXED COLORS) */}
                            {(ach.skill_ids?.length > 0 || ach.new_skills?.length > 0) && (
                                <div className="d-flex flex-wrap gap-1 mt-1">
                                    {(ach.skill_ids || []).map((id) => (
                                        <span 
                                            key={id} 
                                            className="px-2 py-0.5 rounded-pill bg-slate-100 text-slate-700 border border-slate-200 fw-normal"
                                            style={{ fontSize: '0.75rem' }}
                                        >
                                            {getSkillName(id)}
                                        </span>
                                    ))}
                                    {(ach.new_skills || []).map((s, i) => (
                                        <span 
                                            key={i} 
                                            className="px-2 py-0.5 rounded-pill bg-emerald-50 text-emerald-700 border border-emerald-200 fw-normal"
                                            style={{ fontSize: '0.75rem' }}
                                        >
                                            +{s.name}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
      ))}
    </div>
  );
};

export default AchievementDisplayGrid;