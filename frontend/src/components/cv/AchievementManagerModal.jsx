import React, { useState } from 'react';
import AchievementLinker from './AchievementLinker';
import AchievementForm from './AchievementForm';

const AchievementManagerModal = ({
  isOpen,
  onClose,
  allAchievements,
  allSkills,
  selectedAchievementIds,
  setSelectedAchievementIds,
  pendingAchievements,
  setPendingAchievements
}) => {
  const [localAchievements, setLocalAchievements] = useState([...pendingAchievements]);
  const [editingIndex, setEditingIndex] = useState(null);

  if (!isOpen) return null;

  const handleTempAchievementSubmit = (cvId, data) => {
    if (editingIndex !== null) {
      // Edit mode
      const updated = [...localAchievements];
      updated[editingIndex] = {
        ...updated[editingIndex],
        ...data,
      };
      setLocalAchievements(updated);
      setEditingIndex(null);
    } else {
      // Add new
      setLocalAchievements(prev => [
        ...prev,
        { ...data, id: `temp-${Date.now()}` }
      ]);
    }
  };

  const handleAddToList = () => {
    setPendingAchievements(localAchievements);
    onClose();
  };

  const handleEdit = (index) => {
    setEditingIndex(index);
  };

  const handleDelete = (index) => {
    const updated = [...localAchievements];
    updated.splice(index, 1);
    setLocalAchievements(updated);
    if (editingIndex === index) setEditingIndex(null);
  };

  const handleCancelEdit = () => {
    setEditingIndex(null);
  };

  const getSkillName = (id) => {
    const skill = allSkills.find(s => s.id === id);
    return skill ? skill.name : 'Unknown Skill';
  };

  const editingAchievement = editingIndex !== null ? localAchievements[editingIndex] : null;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex',
      justifyContent: 'center', alignItems: 'center', zIndex: 2000
    }}>
      <div style={{
        backgroundColor: '#fff', padding: '20px', borderRadius: '10px',
        width: '750px', maxHeight: '90vh', overflowY: 'auto'
      }}>
        <h3 style={{ marginTop: 0, color: '#007bff' }}>Manage Achievements</h3>

        {/* Linked achievements from master */}
        <AchievementLinker
          allAchievements={allAchievements}
          selectedAchievementIds={selectedAchievementIds}
          setSelectedAchievementIds={setSelectedAchievementIds}
        />

        <hr style={{ margin: '20px 0' }} />

        {/* Achievement form (add or edit) */}
        <AchievementForm
          onSubmit={handleTempAchievementSubmit}
          cvId={null}
          allSkills={allSkills}
          initialData={editingAchievement}
          onCancelEdit={handleCancelEdit}
        />

        {/* Pending achievement cards */}
        {localAchievements.length > 0 && (
          <>
            <hr style={{ margin: '20px 0' }} />
            <strong>Pending Achievements:</strong>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '10px',
              marginTop: '10px'
            }}>
              {localAchievements.map((ach, idx) => (
                <div
                  key={ach.id || idx}
                  style={{
                    position: 'relative',
                    border: '1px solid #ddd',
                    borderRadius: '10px',
                    padding: '10px 12px',
                    backgroundColor: '#f9f9f9',
                    boxShadow: '0 1px 4px rgba(0,0,0,0.1)'
                  }}
                >
                  {/* Delete Button */}
                  <button
                    type="button"
                    onClick={() => handleDelete(idx)}
                    style={{
                      position: 'absolute',
                      top: '5px',
                      right: '5px',
                      border: 'none',
                      background: 'none',
                      cursor: 'pointer',
                      color: '#dc3545',
                      fontSize: '16px',
                      fontWeight: 'bold'
                    }}
                    title="Remove Achievement"
                  >
                    ✕
                  </button>

                  <div style={{ fontWeight: 'bold', color: '#333' }}>{ach.text}</div>

                  {/* Skill Tags */}
                  <div style={{ marginTop: '6px', display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
                    {(ach.skill_ids || []).map(id => (
                      <span
                        key={id}
                        style={{
                          backgroundColor: '#007bff',
                          color: 'white',
                          borderRadius: '12px',
                          padding: '2px 8px',
                          fontSize: '0.8em'
                        }}
                      >
                        {getSkillName(id)}
                      </span>
                    ))}
                    {(ach.new_skills || []).map((s, i) => (
                      <span
                        key={`new-${i}`}
                        style={{
                          backgroundColor: '#28a745',
                          color: 'white',
                          borderRadius: '12px',
                          padding: '2px 8px',
                          fontSize: '0.8em'
                        }}
                      >
                        {s.name}
                      </span>
                    ))}
                  </div>

                  {/* Edit Button */}
                  <div style={{ textAlign: 'right', marginTop: '8px' }}>
                    <button
                      type="button"
                      onClick={() => handleEdit(idx)}
                      style={{
                        backgroundColor: '#ffc107',
                        color: '#333',
                        border: 'none',
                        padding: '4px 10px',
                        borderRadius: '5px',
                        cursor: 'pointer'
                      }}
                    >
                      Edit
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}

        <div style={{ marginTop: '20px', textAlign: 'right' }}>
          <button
            type="button"
            onClick={handleAddToList}
            style={{
              marginRight: '10px',
              backgroundColor: '#007bff',
              color: 'white',
              padding: '6px 12px',
              borderRadius: '5px'
            }}
          >
            Add to Experience
          </button>
          <button
            type="button"
            onClick={onClose}
            style={{
              backgroundColor: '#6c757d',
              color: 'white',
              padding: '6px 12px',
              borderRadius: '5px'
            }}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default AchievementManagerModal;
