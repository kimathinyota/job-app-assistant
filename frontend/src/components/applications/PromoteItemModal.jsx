// frontend/src/components/applications/PromoteItemModal.jsx
import React, { useState } from 'react';
import { ArrowUpCircle, Link as LinkIcon, FileText, Check } from 'lucide-react';
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
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }} tabIndex="-1">
      <div className="modal-dialog modal-dialog-centered">
        <form className="modal-content border-0 shadow-lg rounded-4 overflow-hidden" onSubmit={handleSubmit}>
          
          {/* Header */}
          <div className="modal-header border-bottom-0 bg-white pb-0">
            <div className="d-flex align-items-center gap-3">
                <div className="bg-success bg-opacity-10 p-2 rounded-circle text-success">
                    <ArrowUpCircle size={24} />
                </div>
                <div>
                    <h5 className="modal-title fw-bold text-dark mb-0">Promote CV Item</h5>
                    <p className="text-muted small mb-0">Force this item into your CV by linking it.</p>
                </div>
            </div>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>

          <div className="modal-body p-4">
            
            {/* Source Item Display */}
            <div className="mb-4">
                <label className="form-label small fw-bold text-muted text-uppercase">Item to Promote</label>
                <div className="p-3 bg-light rounded-3 border d-flex align-items-center gap-3">
                    <FileText size={20} className="text-muted"/>
                    <span className="fw-bold text-dark">{itemTitle}</span>
                </div>
            </div>

            {/* Link Selection */}
            <div className="mb-4">
              <label htmlFor="req-select" className="form-label small fw-bold text-muted text-uppercase d-flex align-items-center gap-2">
                <LinkIcon size={14}/> Link to Requirement
              </label>
              <select
                id="req-select"
                className="form-select shadow-sm border-secondary-subtle"
                value={selectedReqId}
                onChange={(e) => setSelectedReqId(e.target.value)}
                required
              >
                <option value="" disabled>-- Choose a Job Requirement --</option>
                {job.features.map(req => (
                  <option key={req.id} value={req.id}>
                    {req.description}
                  </option>
                ))}
              </select>
              <div className="form-text small text-muted mt-2">
                Select the job requirement that this item best satisfies.
              </div>
            </div>

            {/* Annotation */}
            <div className="mb-2">
                <label htmlFor="promote-annotation" className="form-label small fw-bold text-muted text-uppercase">Reasoning (Optional)</label>
                <textarea 
                    id="promote-annotation"
                    className="form-control shadow-sm border-secondary-subtle"
                    rows="2"
                    value={annotation}
                    onChange={(e) => setAnnotation(e.target.value)}
                    placeholder="e.g., This hobby demonstrates my strategic thinking..."
                />
            </div>
            
          </div>

          {/* Footer */}
          <div className="modal-footer border-top-0 bg-light bg-opacity-50">
            <button type="button" className="btn btn-white border shadow-sm px-4" onClick={onClose}>
                Cancel
            </button>
            <button type="submit" className="btn btn-success shadow-sm px-4 d-flex align-items-center gap-2" disabled={isSubmitting}>
              {isSubmitting ? 'Linking...' : <><Check size={16}/> Promote & Add</>}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default PromoteItemModal;