// frontend/src/components/rolecase/ForensicItemCard.jsx
import React from 'react';
import { Check, X, Plus, AlertTriangle } from 'lucide-react';

export const ForensicItemCard = ({ item, onReject, onAdd }) => {
  const { requirement_text, status, match_summary, lineage } = item;
  const isMissing = status === "missing";
  const isVerified = status === "verified";

  // Chip Styles
  const typeBadgeClass = (type) => {
    switch(type) {
      case 'experience': return "bg-primary bg-opacity-10 text-primary border-primary border-opacity-10";
      case 'project': return "bg-info bg-opacity-10 text-info border-info border-opacity-10";
      case 'skill': return "bg-success bg-opacity-10 text-success border-success border-opacity-10";
      default: return "bg-secondary bg-opacity-10 text-secondary border-secondary border-opacity-10";
    }
  };

  return (
    <div className={`card border shadow-sm transition-all position-relative ${isMissing ? 'border-start-4 border-danger bg-light' : ''} ${isVerified ? 'border-start-4 border-success' : ''}`}>
      <div className="card-body p-3">
        
        {/* Header */}
        <div className="d-flex justify-content-between align-items-start mb-2">
          {/* Status Badge */}
          {isMissing && <span className="badge bg-danger bg-opacity-10 text-danger border border-danger border-opacity-25">MISSING</span>}
          {isVerified && <span className="badge bg-success bg-opacity-10 text-success border border-success border-opacity-25">VERIFIED</span>}
          {status === 'pending' && <span className="badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-25">WEAK MATCH</span>}

          {/* Action Buttons */}
          <div className="d-flex gap-1">
            <button className="btn btn-sm btn-light border py-0 px-1 text-primary" onClick={onAdd} title="Manual Match">
              <Plus size={14} />
            </button>
            {!isMissing && (
              <button className="btn btn-sm btn-light border py-0 px-1 text-danger" onClick={onReject} title="Reject Match">
                <X size={14} />
              </button>
            )}
          </div>
        </div>

        {/* Requirement Text */}
        <p className="card-text small fw-bold text-dark mb-3">
          {requirement_text}
        </p>

        {/* Evidence Section */}
        {!isMissing ? (
          <div className="bg-light p-2 rounded border">
            {/* Summary */}
            <div className="small text-muted fst-italic mb-2 text-truncate">
              "{item.best_match_excerpt || match_summary}"
            </div>
            
            {/* Chips */}
            <div className="d-flex flex-wrap gap-1">
              {lineage && lineage.map((step, idx) => (
                <div key={idx} className="d-flex align-items-center">
                  <span className={`badge border fw-normal text-dark ${typeBadgeClass(step.type)}`}>
                    {step.name}
                  </span>
                  {idx < lineage.length - 1 && <span className="text-muted small mx-1">›</span>}
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="d-grid">
            <button 
              onClick={onAdd}
              className="btn btn-sm btn-outline-secondary border-dashed text-muted d-flex align-items-center justify-content-center gap-2"
            >
              <AlertTriangle size={14} /> Find Evidence
            </button>
          </div>
        )}
      </div>
    </div>
  );
};