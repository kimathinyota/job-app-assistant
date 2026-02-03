// frontend/src/components/rolecase/EvidenceBoard.jsx
import React from 'react';
import { ForensicItemCard } from './ForensicItemCard';

const COLUMNS = [
  { key: "Critical", label: "Critical Requirements", sub: "Must Have", color: "border-danger", badge: "bg-danger" },
  { key: "High", label: "High Priority", sub: "Should Have", color: "border-warning", badge: "bg-warning text-dark" },
  { key: "Standard", label: "Standard", sub: "Nice to Have", color: "border-primary", badge: "bg-primary" },
  { key: "Bonus", label: "Bonus", sub: "Differentiators", color: "border-secondary", badge: "bg-secondary" }
];

export const EvidenceBoard = ({ groups, onReject, onOpenManual }) => {
  return (
    <div className="d-flex h-100 gap-4" style={{ minWidth: 'min-content' }}>
      {COLUMNS.map(col => {
        const items = groups[col.key] || [];
        return (
          <div key={col.key} className="d-flex flex-column rounded-3 bg-light border shadow-sm" style={{ width: '350px', minWidth: '350px' }}>
            
            {/* Column Header */}
            <div className={`p-3 bg-white rounded-top border-bottom border-top border-top-4 ${col.color}`}>
              <div className="d-flex justify-content-between align-items-center mb-1">
                <h6 className="fw-bold text-dark mb-0">{col.label}</h6>
                <span className={`badge rounded-pill ${col.badge}`}>{items.length}</span>
              </div>
              <small className="text-uppercase text-muted fw-bold" style={{fontSize: '0.65rem'}}>{col.sub}</small>
            </div>

            {/* Scrollable List */}
            <div className="flex-grow-1 overflow-auto p-2 custom-scrollbar">
              <div className="d-flex flex-column gap-2">
                {items.map(item => (
                  <ForensicItemCard 
                    key={item.requirement_id} 
                    item={item} 
                    onReject={() => onReject(item.requirement_id)}
                    onAdd={() => onOpenManual(item.requirement_id)}
                  />
                ))}
                
                {items.length === 0 && (
                  <div className="text-center p-4 text-muted border rounded-3 border-dashed bg-white mt-2">
                    <small>No items in this category</small>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};