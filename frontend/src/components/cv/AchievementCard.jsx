// frontend/src/components/cv/AchievementCard.jsx
import React from 'react';
import { Edit2, Trash2, Award } from 'lucide-react';

const AchievementCard = ({ 
  achievement, 
  allSkills = [], 
  onDelete, 
  onEdit,
  isPending = false 
}) => {
  if (!achievement) return null;

  const getSkillName = (id) => {
    const skill = allSkills.find((s) => s.id === id);
    return skill ? skill.name : id;
  };

  return (
    <div className={`
      card border-0 shadow-sm h-100 hover-lift transition-all bg-white
      ${isPending ? 'border-start border-4 border-warning' : ''}
    `}>
      <div className="card-body p-3 position-relative">
        
        {/* Action Buttons */}
        <div className="position-absolute top-0 end-0 p-2 d-flex gap-1">
          {onEdit && (
            <button
              type="button"
              onClick={onEdit}
              className="btn btn-light btn-sm text-primary p-1"
              title="Edit"
              style={{ width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <Edit2 size={14} />
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              onClick={onDelete}
              className="btn btn-light btn-sm text-danger p-1"
              title="Remove"
              style={{ width: '28px', height: '28px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <Trash2 size={14} />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="d-flex gap-3">
          <div className="mt-1 flex-shrink-0">
            <div className="p-2 bg-amber-50 text-amber-600 rounded-circle">
                <Award size={18} />
            </div>
          </div>

          <div className="flex-grow-1" style={{ paddingRight: '3rem' }}>
            <p className="mb-2 text-dark fw-medium small lh-sm">
              {achievement.text}
            </p>
            
            {achievement.context && (
                <div className="mb-2">
                    <span className="d-inline-block px-2 py-1 rounded-pill bg-light text-secondary border small" style={{fontSize: '0.75rem'}}>
                        Context: {achievement.context}
                    </span>
                </div>
            )}

            {/* Skill Tags (FIXED COLORS) */}
            <div className="d-flex flex-wrap gap-1 mt-2">
              {(achievement.skill_ids || []).map((id) => (
                <span
                  key={id}
                  className="px-2 py-0.5 rounded-pill bg-slate-100 text-slate-700 border border-slate-200 fw-normal"
                  style={{ fontSize: '0.75rem' }}
                >
                  {getSkillName(id)}
                </span>
              ))}
              {(achievement.new_skills || []).map((s, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 rounded-pill bg-emerald-50 text-emerald-700 border border-emerald-200 fw-normal"
                  style={{ fontSize: '0.75rem' }}
                >
                  +{s.name}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AchievementCard;