// frontend/src/components/rolecase/RequirementItem.jsx
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Check, X, Link as LinkIcon, Trash2, Layers } from 'lucide-react';

export const RequirementItem = ({ item, onReject, onLinkEvidence }) => {
  const [expanded, setExpanded] = useState(false);
  const [showAlternatives, setShowAlternatives] = useState(false); // Toggle for alt matches
  
  const { 
    requirement_text, 
    requirement_type, // Now using this!
    status, 
    match_summary, 
    lineage, 
    best_match_text,
    alternatives // List of other matches
  } = item;

  const isMissing = status === "missing";
  const isVerified = status === "verified";

  // Helper to format type (e.g. "hard_skill" -> "Hard Skill")
  const formatType = (t) => t ? t.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Requirement';

  // Helper for badge color based on type
  const getTypeColor = (t) => {
    switch(t) {
      case 'hard_skill': return 'bg-blue-100 text-blue-700';
      case 'soft_skill': return 'bg-purple-100 text-purple-700';
      case 'qualification': return 'bg-amber-100 text-amber-700';
      default: return 'bg-gray-100 text-gray-600';
    }
  };

  return (
    <div className={`list-group-item p-0 border-light ${expanded ? 'bg-light bg-opacity-25' : 'bg-white'}`}>
      
      {/* 1. The Summary Row */}
      <div 
        className="d-flex align-items-center p-3 cursor-pointer" 
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer' }}
      >
        {/* Status Icon */}
        <div className="me-3" style={{ width: '24px' }}>
          {isVerified && <div className="text-success"><Check size={20} /></div>}
          {status === 'pending' && <div className="text-warning"><div className="spinner-grow spinner-grow-sm" /></div>}
          {isMissing && <div className="text-danger opacity-50"><X size={20} /></div>}
        </div>

        {/* Content */}
        <div className="flex-grow-1">
          <div className="d-flex align-items-center gap-2 mb-1">
            {/* TYPE BADGE (New) */}
            <span className={`badge rounded-pill fw-normal ${getTypeColor(requirement_type)}`} style={{ fontSize: '0.65rem' }}>
              {formatType(requirement_type)}
            </span>
          </div>
          
          <div className={`fw-medium ${isMissing ? 'text-secondary' : 'text-dark'}`}>
            {requirement_text}
          </div>
          
          {!expanded && !isMissing && (
            <div className="text-muted small mt-1 d-flex align-items-center" style={{ fontSize: '0.75rem' }}>
              <LinkIcon size={10} className="me-1"/> 
              Found in: <span className="fw-semibold ms-1">{best_match_text || "Resume"}</span>
              {alternatives && alternatives.length > 0 && (
                <span className="ms-2 badge bg-light text-secondary border">+ {alternatives.length} others</span>
              )}
            </div>
          )}
        </div>

        {/* Chevron */}
        <div className="text-muted ms-3">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>

      {/* 2. The Expanded Details */}
      {expanded && (
        <div className="px-3 pb-3 ps-5">
          <div className="border-start border-2 ps-3 py-1">
            
            {isMissing ? (
              <div className="bg-white p-3 rounded border border-dashed border-danger border-opacity-25">
                <p className="text-muted small mb-2">
                  We couldn't find evidence of this in your CV. Do you have this skill?
                </p>
                <button 
                  onClick={(e) => { e.stopPropagation(); onLinkEvidence(); }}
                  className="btn btn-sm btn-outline-primary d-flex align-items-center gap-2"
                >
                  <LinkIcon size={14} /> Yes, I have this
                </button>
              </div>
            ) : (
              <>
                <div className="mb-2 d-flex justify-content-between align-items-center">
                  <div>
                    <span className="badge bg-success bg-opacity-10 text-success border border-success border-opacity-10 me-2">Best Match</span>
                    <span className="text-muted small">
                      {Math.round(item.best_match_confidence * 100)}% Confidence
                    </span>
                  </div>
                </div>
                
                {/* Main Evidence Card */}
                <div className="bg-white p-3 rounded border mb-3 shadow-sm border-start-4 border-start-success">
                  <div className="d-flex justify-content-between mb-2">
                    <span className="small fw-bold text-dark">{best_match_text}</span>
                  </div>
                  <div className="text-muted small fst-italic mb-2">
                    "{item.best_match_excerpt || match_summary}"
                  </div>
                  <div className="d-flex flex-wrap gap-1">
                    {lineage && lineage.map((step, idx) => (
                      <span key={idx} className="badge bg-secondary bg-opacity-10 text-secondary border fw-normal">
                        {step.name}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Alternatives Section (New) */}
                {alternatives && alternatives.length > 0 && (
                  <div className="mb-3">
                    <button 
                      onClick={(e) => { e.stopPropagation(); setShowAlternatives(!showAlternatives); }}
                      className="btn btn-sm btn-light border d-flex align-items-center gap-2 text-secondary w-100 justify-content-center"
                    >
                      <Layers size={14} /> 
                      {showAlternatives ? "Hide" : `See ${alternatives.length} Alternative Matches`}
                    </button>

                    {showAlternatives && (
                      <div className="mt-2 d-flex flex-column gap-2">
                        {alternatives.map((alt, idx) => (
                          <div key={idx} className="bg-light p-2 rounded border d-flex justify-content-between align-items-center">
                            <div className="small text-truncate" style={{ maxWidth: '70%' }}>
                              <span className="fw-bold d-block text-dark" style={{ fontSize: '0.75rem' }}>{alt.source_name}</span>
                              <span className="text-muted fst-italic" style={{ fontSize: '0.75rem' }}>"{alt.match_text.substring(0, 50)}..."</span>
                            </div>
                            <span className="badge bg-white text-secondary border">
                              {Math.round(alt.score * 100)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Action Buttons */}
                <div className="d-flex gap-2 mt-3 pt-2 border-top">
                  <button 
                    onClick={(e) => { e.stopPropagation(); onLinkEvidence(); }}
                    className="btn btn-sm btn-link text-decoration-none text-muted p-0 d-flex align-items-center gap-1"
                  >
                    <LinkIcon size={12} /> Change Evidence
                  </button>
                  <div className="vr opacity-25"></div>
                  <button 
                    onClick={(e) => { e.stopPropagation(); onReject(); }}
                    className="btn btn-sm btn-link text-decoration-none text-danger p-0 d-flex align-items-center gap-1"
                  >
                    <Trash2 size={12} /> Reject Match
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};