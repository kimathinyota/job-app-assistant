// frontend/src/components/rolecase/EvidenceLinkerModal.jsx
import React, { useState } from 'react';

export const EvidenceLinkerModal = ({ isOpen, feature, onClose, onSubmit }) => {
  const [text, setText] = useState("");

  if (!isOpen) return null;

  return (
    <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}>
      <div className="modal-dialog modal-dialog-centered">
        <div className="modal-content shadow-lg border-0">
          <div className="modal-header">
            <div>
              <h5 className="modal-title fw-bold">Link Evidence</h5>
              <p className="mb-0 text-muted small">Show that you meet: <strong>{feature.requirement_text}</strong></p>
            </div>
            <button type="button" className="btn-close" onClick={onClose}></button>
          </div>
          
          <div className="modal-body">
            <label className="form-label small fw-bold text-secondary">
              Describe your experience:
            </label>
            <textarea
              className="form-control"
              rows="4"
              placeholder="e.g., 'I have 3 years of experience in this skill from my role at Company X...'"
              value={text}
              onChange={(e) => setText(e.target.value)}
              autoFocus
            />
            <div className="form-text text-muted small mt-2">
              <i className="bi bi-info-circle me-1"></i>
              Adding this will update your match score instantly.
            </div>
          </div>

          <div className="modal-footer bg-light">
            <button type="button" className="btn btn-link text-muted text-decoration-none" onClick={onClose}>Cancel</button>
            <button 
              type="button" 
              className="btn btn-primary px-4"
              onClick={() => onSubmit({ evidence_text: text, cv_item_name: "Manual Entry" })}
              disabled={!text.trim()}
            >
              Add Evidence
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};