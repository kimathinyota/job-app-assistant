// frontend/src/components/rolecase/RequirementList.jsx
import React from 'react';
import { RequirementItem } from './RequirementItem';

const SECTIONS = [
  { id: 'Critical', label: "Must Haves", description: "The job description emphasizes these." },
  { id: 'High', label: "Key Skills", description: "Important for day-to-day work." },
  { id: 'Standard', label: "Nice to Haves", description: "General requirements and soft skills." },
  { id: 'Bonus', label: "Bonus Points", description: "Things that set you apart." },
];

export const RequirementList = ({ 
  groups, 
  onReject, 
  onLinkEvidence, 
  onPromote, 
  onApprove, 
  onViewEvidence 
}) => {
  return (
    <div className="d-flex flex-column gap-4">
      {SECTIONS.map(section => {
        const items = groups[section.id] || [];
        if (items.length === 0) return null;

        return (
          <div key={section.id} className="card border-0 shadow-sm">
            <div className="card-header bg-white border-bottom border-light py-3">
              <div className="d-flex justify-content-between align-items-center">
                <div>
                  <h6 className="fw-bold mb-0 text-dark">{section.label}</h6>
                  <small className="text-muted">{section.description}</small>
                </div>
                <span className="badge bg-light text-secondary rounded-pill">
                  {items.length}
                </span>
              </div>
            </div>
            
            <div className="list-group list-group-flush">
              {items.map(item => (
                <RequirementItem 
                  key={item.requirement_id} 
                  item={item}
                  onReject={() => onReject(item.requirement_id)}
                  onLinkEvidence={() => onLinkEvidence(item)}
                  // --- PASS DOWN HANDLERS ---
                  onPromote={onPromote}
                  onApprove={onApprove}
                  onViewEvidence={onViewEvidence}
                />
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
};