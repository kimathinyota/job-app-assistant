// frontend/src/components/rolecase/ManualMatchModal.jsx
import React, { useState } from 'react';

export const ManualMatchModal = ({ isOpen, onClose, onSubmit }) => {
  const [text, setText] = useState("");

  if (!isOpen) return null;

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content shadow">
          <div className="modal-header border-bottom-0">
            <h5 className="modal-title fw-bold">Add Manual Evidence</h5>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>
          <div className="modal-body">
            <p className="text-muted small mb-3">
              Paste the specific text from your CV or describe your experience that proves you meet this requirement.
            </p>
            <textarea
              className="form-control"
              rows="4"
              placeholder="e.g., 'I managed a team of 5 developers...'"
              value={text}
              onChange={(e) => setText(e.target.value)}
              autoFocus
            />
          </div>
          <div className="modal-footer border-top-0">
            <button type="button" className="btn btn-light" onClick={onClose}>Cancel</button>
            <button 
              type="button" 
              className="btn btn-primary"
              onClick={() => onSubmit({ evidence_text: text })}
              disabled={!text.trim()}
            >
              Save Match
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};