// frontend/src/components/cv/AchievementForm.jsx
import React, { useState, useEffect } from 'react';
import { Award, Layers, ChevronDown, ChevronUp } from 'lucide-react';
import SkillLinker from './SkillLinker'; // <--- IMPORT DIRECTLY
import SelectedSkillsDisplay from './SelectedSkillsDisplay';

const AchievementForm = ({ 
  onSubmit, 
  cvId, 
  allSkills, 
  initialData, 
  onCancelEdit 
}) => {
  const [text, setText] = useState('');
  const [context, setContext] = useState(''); 
  const [selectedSkillIds, setSelectedSkillIds] = useState([]);
  const [pendingSkills, setPendingSkills] = useState([]);
  
  // Replaces the modal state with a simple toggle
  const [showSkillLinker, setShowSkillLinker] = useState(false);

  const isEditing = Boolean(initialData);

  useEffect(() => {
    if (isEditing) {
      setText(initialData.text || '');
      setContext(initialData.context || '');
      setSelectedSkillIds(initialData.skill_ids || initialData.existing_skill_ids || []);
      setPendingSkills(initialData.new_skills || []);
      // Auto-open linker if editing a new item that has no text yet (optional UX tweak)
      if (!initialData.text) setShowSkillLinker(true);
    } else {
      setText('');
      setContext('');
      setSelectedSkillIds([]);
      setPendingSkills([]);
      setShowSkillLinker(false);
    }
  }, [initialData, isEditing]);

  const handleFormSubmit = () => {
    if (!text.trim()) return;

    const dataToSend = {
      text: text,
      context: context || null, 
      existing_skill_ids: selectedSkillIds, 
      new_skills: pendingSkills
    };

    if (isEditing && initialData.id && !String(initialData.id).startsWith('pending-')) {
      dataToSend.id = initialData.id;
    }

    onSubmit(cvId, dataToSend, 'Achievement');

    if (!isEditing) {
        setText('');
        setContext('');
        setSelectedSkillIds([]);
        setPendingSkills([]);
        setShowSkillLinker(false);
    }
  };

  return (
    <div className="card border-0 shadow-sm p-4 bg-white">
      
      {/* Header */}
      <div className="d-flex align-items-center gap-2 mb-4 border-bottom pb-2">
        <Award className="text-amber-500" size={20}/>
        <h5 className="mb-0 fw-bold text-dark">
            {isEditing ? 'Edit Achievement' : 'Add New Achievement'}
        </h5>
      </div>

      {/* Core Fields */}
      <div className="mb-3">
        <label className="form-label fw-bold small text-uppercase text-muted">Achievement Details</label>
        <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g., Optimized database queries reducing load by 40%"
            className="form-control mb-3"
            rows="3"
            autoFocus={!isEditing} // Focus text on new
        />
        <input
            type="text"
            value={context}
            onChange={(e) => setContext(e.target.value)}
            placeholder="Context (e.g., 'Global', 'Project X')"
            className="form-control"
        />
      </div>

      {/* Inline Skill Management */}
      <div className="mb-4 pt-2">
        <div 
            className="d-flex justify-content-between align-items-center mb-2 cursor-pointer"
            onClick={() => setShowSkillLinker(!showSkillLinker)}
        >
            <label className="form-label fw-bold text-dark d-flex align-items-center gap-2 mb-0 cursor-pointer">
                <Layers size={16} className="text-emerald-600"/> 
                Related Skills
                <span className="text-muted fw-normal small">
                    ({selectedSkillIds.length + pendingSkills.length})
                </span>
            </label>
            <button 
                type="button" 
                className="btn btn-light btn-sm text-secondary"
            >
                {showSkillLinker ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
            </button>
        </div>

        {/* Logic: Show Linker when open, show Summary when closed */}
        {showSkillLinker ? (
            <div className="animate-fade-in mt-2">
                <SkillLinker
                    allSkills={allSkills}
                    selectedSkillIds={selectedSkillIds}
                    setSelectedSkillIds={setSelectedSkillIds}
                    pendingSkills={pendingSkills}
                    setPendingSkills={setPendingSkills}
                />
            </div>
        ) : (
            (selectedSkillIds.length > 0 || pendingSkills.length > 0) ? (
                <div 
                    className="bg-light p-3 rounded border cursor-pointer hover:bg-slate-100 transition-all"
                    onClick={() => setShowSkillLinker(true)}
                >
                    <SelectedSkillsDisplay
                        allSkills={allSkills}
                        selectedSkillIds={selectedSkillIds}
                        pendingSkills={pendingSkills}
                    />
                </div>
            ) : (
                <div 
                    className="text-muted small fst-italic border border-dashed rounded p-2 text-center cursor-pointer hover:bg-light"
                    onClick={() => setShowSkillLinker(true)}
                >
                    Click to link skills...
                </div>
            )
        )}
      </div>

      {/* Actions */}
      <div className="d-flex gap-2 justify-content-end mt-4 pt-3 border-top">
        {onCancelEdit && (
          <button
            type="button"
            onClick={onCancelEdit}
            className="btn btn-light border"
          >
            Cancel
          </button>
        )}
        <button
          type="button"
          onClick={handleFormSubmit} 
          className="btn btn-primary px-4"
        >
          {isEditing ? 'Save Changes' : 'Add to List'}
        </button>
      </div>
    </div>
  );
};

export default AchievementForm;