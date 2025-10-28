import React, { useState } from 'react';
import AchievementForm from './AchievementForm';
import AchievementDisplayGrid from './AchievementDisplayGrid';

const AchievementManagerModal = ({
  isOpen,
  onClose,
  allAchievements = [],
  selectedAchievementIds = [],
  setSelectedAchievementIds = () => {},
  pendingAchievements = [],
  setPendingAchievements = () => {},
  allSkills = []
}) => {
  const [editingIndex, setEditingIndex] = useState(null);
  const [formKey, setFormKey] = useState(0); // 👈 Forces re-render of AchievementForm

  if (!isOpen) return null;

  const handleTempAchievementSubmit = (_, newAchievement) => {
    if (editingIndex !== null) {
      // update existing
      const updated = [...pendingAchievements];
      updated[editingIndex] = {
        ...updated[editingIndex],
        ...newAchievement
      };
      setPendingAchievements(updated);
      setEditingIndex(null);
    } else {
      // add new
      setPendingAchievements([
        ...pendingAchievements,
        { ...newAchievement, id: Date.now() }
      ]);
    }

    // reset the form cleanly
    setFormKey((k) => k + 1);
  };

  const handleRemovePending = (index) => {
    const updated = pendingAchievements.filter((_, i) => i !== index);
    setPendingAchievements(updated);
    if (editingIndex === index) {
      setEditingIndex(null);
      setFormKey((k) => k + 1);
    }
  };

  const handleEditAchievement = (index) => {
    setEditingIndex(index);
    setFormKey((k) => k + 1); // 🔄 force rerender form with new data
  };

  return (
    <div
      style={{
        position: 'fixed', top: 0, left: 0,
        width: '100%', height: '100%',
        backgroundColor: 'rgba(0,0,0,0.4)',
        display: 'flex', justifyContent: 'center', alignItems: 'center',
        zIndex: 3000
      }}
    >
      <div
        style={{
          backgroundColor: '#f8f9fa',
          padding: '20px',
          borderRadius: '10px',
          width: '750px',
          maxHeight: '90vh',
          overflowY: 'auto'
        }}
      >
        <h3>Manage Achievements</h3>

        {/* Existing Achievements */}
        <div style={{ borderTop: '1px solid #ddd', paddingTop: '10px' }}>
          <strong>Link Existing Achievements</strong>
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '6px',
              marginTop: '8px'
            }}
          >
            {(allAchievements || []).map((a) => (
              <label
                key={a.id}
                style={{
                  backgroundColor: selectedAchievementIds.includes(a.id)
                    ? '#d4edda'
                    : 'white',
                  padding: '5px 10px',
                  borderRadius: '15px',
                  border: '1px solid #ccc',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="checkbox"
                  checked={selectedAchievementIds.includes(a.id)}
                  onChange={() => {
                    if (selectedAchievementIds.includes(a.id)) {
                      setSelectedAchievementIds(
                        selectedAchievementIds.filter((id) => id !== a.id)
                      );
                    } else {
                      setSelectedAchievementIds([
                        ...selectedAchievementIds,
                        a.id
                      ]);
                    }
                  }}
                  style={{ marginRight: '5px' }}
                />
                {a.text}
              </label>
            ))}
          </div>
        </div>

        {/* Pending Achievements */}
        <div
          style={{
            marginTop: '15px',
            borderTop: '1px solid #ddd',
            paddingTop: '10px'
          }}
        >
          <strong>Pending Achievements</strong>
          <AchievementDisplayGrid
            allSkills={allSkills}
            pendingAchievements={pendingAchievements || []}
            onRemoveAchievement={(idOrIndex) => {
              const index = pendingAchievements.findIndex(
                (a) => a.id === idOrIndex || pendingAchievements.indexOf(a) === idOrIndex
              );
              if (index !== -1) handleRemovePending(index);
            }}
            onEditAchievement={handleEditAchievement}
          />
        </div>

        {/* Add/Edit Achievement */}
        <div
          style={{
            marginTop: '15px',
            borderTop: '1px solid #ddd',
            paddingTop: '10px'
          }}
        >
          <AchievementForm
            key={formKey} // 👈 ensures proper state reset and editing
            onSubmit={handleTempAchievementSubmit}
            cvId={null}
            allSkills={allSkills}
            initialData={
              editingIndex !== null
                ? pendingAchievements[editingIndex]
                : null
            }
            onCancelEdit={() => {
              setEditingIndex(null);
              setFormKey((k) => k + 1);
            }}
          />
        </div>

        <div style={{ textAlign: 'right', marginTop: '15px' }}>
          <button
            onClick={onClose}
            style={{
              padding: '8px 12px',
              backgroundColor: '#6c757d',
              color: 'white',
              borderRadius: '5px'
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default AchievementManagerModal;
