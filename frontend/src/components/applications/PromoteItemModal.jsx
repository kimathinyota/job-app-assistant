// frontend/src/components/applications/PromoteItemModal.jsx
import React, { useState } from 'react';
import { addMappingPair } from '../../api/applicationClient';

/**
 * This modal allows a user to "promote" an unmapped CV item
 * by forcing them to create a new mapping pair for it.
 */
const PromoteItemModal = ({
  isOpen,
  onClose,
  job,          // For the list of requirements
  mapping,      // For the mapping ID
  itemToPromote, // An object like { item: {...}, type: '...' }
  onMappingChanged // Callback to trigger a data reload
}) => {
  const [selectedReqId, setSelectedReqId] = useState('');
  const [annotation, setAnnotation] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen || !itemToPromote) return null;

  const { item, type } = itemToPromote;

  // Get the simple title for the item
  let itemTitle = item.title || item.name || item.degree || 'Unknown Item';
  if (type === 'experiences') itemTitle = `${item.title} @ ${item.company}`;
  if (type === 'education') itemTitle = `${item.degree} @ ${item.institution}`;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!selectedReqId) {
      alert('Please select a job requirement to link this item to.');
      return;
    }
    
    setIsSubmitting(true);
    try {
      await addMappingPair(
        mapping.id,
        selectedReqId,
        item.id,
        type,
        annotation || `Promoted in Step 2` // Add a default annotation
      );
      
      // Success! Reload all data and close.
      await onMappingChanged();
      onClose();

    } catch (err) {
      alert(`Failed to create pair: ${err.response?.data?.detail || err.message}`);
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div 
      className="modal" 
      style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}
      onClick={onClose}
    >
      <div 
        className="modal-dialog modal-dialog-centered"
        onClick={e => e.stopPropagation()}
      >
        <form className="modal-content" onSubmit={handleSubmit}>
          <div className="modal-header">
            <h5 className="modal-title">Promote CV Item</h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>
          <div className="modal-body">
            <p>To add this item to your CV, you must link it to a job requirement.</p>
            
            <div className="mb-3">
              <label className="form-label fw-medium">CV Item:</label>
              <input 
                type="text" 
                className="form-control" 
                value={itemTitle} 
                disabled 
                readOnly 
              />
            </div>

            <div className="mb-3">
              <label htmlFor="req-select" className="form-label fw-medium">
                Link to Requirement:
              </label>
              <select
                id="req-select"
                className="form-select"
                value={selectedReqId}
                onChange={(e) => setSelectedReqId(e.target.value)}
                required
              >
                <option value="" disabled>-- Select a requirement --</option>
                {job.features.map(req => (
                  <option key={req.id} value={req.id}>
                    {req.description}
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-3">
                <label htmlFor="promote-annotation" className="form-label fw-medium">Annotation (Optional):</label>
                <textarea 
                    id="promote-annotation"
                    className="form-control"
                    rows="2"
                    value={annotation}
                    onChange={(e) => setAnnotation(e.target.value)}
                    placeholder="e.g., This hobby demonstrates my strategic thinking..."
                />
            </div>
            
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={isSubmitting}>
              {isSubmitting ? 'Pairing...' : 'Create Pair & Add Item'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PromoteItemModal;