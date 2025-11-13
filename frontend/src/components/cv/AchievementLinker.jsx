// frontend/src/components/cv/AchievementLinker.jsx
import React, { useState } from 'react';
import { Check, Lock, Link, Search } from 'lucide-react'; // Added Search icon

const AchievementLinker = ({
  allAchievements = [],
  selectedAchievementIds = [],
  setSelectedAchievementIds,
  disabledAchievementIds = [],
}) => {
  // --- NEW: Search State ---
  const [searchTerm, setSearchTerm] = useState('');
  
  const handleToggleAchievement = (achId) => {
    if (selectedAchievementIds.includes(achId)) {
      setSelectedAchievementIds(selectedAchievementIds.filter((id) => id !== achId));
    } else {
      setSelectedAchievementIds([...selectedAchievementIds, achId]);
    }
  };

  // --- NEW: Filter Logic ---
  const filteredAchievements = allAchievements.filter(ach => 
    ach.text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
      <div className="bg-light px-3 py-2 border-bottom d-flex justify-content-between align-items-center">
        <strong className="small text-uppercase text-muted">
            Master Repository
        </strong>
        <span className="badge bg-secondary-subtle text-secondary border">{allAchievements.length} Available</span>
      </div>

      <div className="p-3">
        {/* --- NEW: Search Bar --- */}
        <div className="position-relative mb-3">
            <Search className="position-absolute text-muted" size={16} style={{ top: '10px', left: '12px' }} />
            <input 
                type="text" 
                className="form-control ps-5 form-control-sm"
                placeholder="Search master achievements..."
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
            />
        </div>

        <div 
            className="pe-1"
            style={{ maxHeight: '250px', overflowY: 'auto' }}
        >
            {filteredAchievements.length === 0 ? (
                <p className="text-center text-muted small py-3">
                    {searchTerm ? "No matches found." : "No master achievements found."}
                </p>
            ) : (
                <div className="d-flex flex-column gap-2">
                    {filteredAchievements.map((ach) => {
                    const isDisabled = disabledAchievementIds.includes(ach.id);
                    const isChecked = selectedAchievementIds.includes(ach.id);

                    return (
                        <div
                            key={ach.id}
                            onClick={() => !isDisabled && handleToggleAchievement(ach.id)}
                            className={`
                                d-flex align-items-start gap-3 p-3 rounded border transition-all
                                ${isDisabled ? 'bg-light opacity-50 cursor-not-allowed border-transparent' : 'cursor-pointer hover:bg-slate-50'}
                                ${isChecked && !isDisabled ? 'border-primary bg-blue-50' : 'border-light'}
                            `}
                        >
                            {/* Checkbox visual */}
                            <div className={`
                                mt-1 flex-shrink-0 rounded-circle border d-flex align-items-center justify-content-center
                                ${isChecked ? 'bg-primary border-primary text-white' : 'bg-white border-secondary'}
                            `} style={{ width: '20px', height: '20px' }}>
                                {isDisabled ? <Lock size={12} className="text-muted"/> : isChecked && <Check size={12} />}
                            </div>

                            {/* Content */}
                            <div className="flex-grow-1">
                                <p className={`mb-0 small fw-medium ${isChecked ? 'text-dark' : 'text-secondary'}`}>
                                    {ach.text}
                                </p>
                                {isDisabled && (
                                    <span className="text-xs text-danger d-block mt-1">
                                        * Currently being modified
                                    </span>
                                )}
                            </div>
                        </div>
                    );
                    })}
                </div>
            )}
        </div>
      </div>
    </div>
  );
};

export default AchievementLinker;