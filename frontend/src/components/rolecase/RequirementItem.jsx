// frontend/src/components/rolecase/RequirementItem.jsx
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Check, X, Link as LinkIcon, Trash2, Layers, ThumbsUp, ArrowUpCircle, Eye } from 'lucide-react';

export const RequirementItem = ({ item, onReject, onLinkEvidence, onPromote, onApprove, onViewEvidence }) => {
  const [expanded, setExpanded] = useState(false);
  const [showAlternatives, setShowAlternatives] = useState(false);
  
  const { 
    requirement_text, 
    requirement_type, 
    status, 
    match_summary, 
    lineage, 
    best_match_text,
    best_match_confidence,
    best_match_excerpt,
    alternatives 
  } = item;

  const isMissing = status === "missing";
  const isVerified = status === "verified";
  const isApproved = best_match_confidence === 1.0;

  // 1. BOOTSTRAP COLOR HELPERS (Fixed visibility issue)
  const formatType = (t) => t ? t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Requirement';
  
  const getTypeBadgeClass = (t) => {
    switch(t) {
      // Using Bootstrap 'bg-subtle' classes if available, or custom opacity approach
      case 'hard_skill': return 'bg-primary bg-opacity-10 text-primary border border-primary border-opacity-10';
      case 'soft_skill': return 'bg-info bg-opacity-10 text-info border border-info border-opacity-10';
      case 'qualification': return 'bg-warning bg-opacity-10 text-warning border border-warning border-opacity-10';
      case 'experience': return 'bg-secondary bg-opacity-10 text-secondary border border-secondary border-opacity-10';
      default: return 'bg-light text-secondary border';
    }
  };

  const getLineageBadgeClass = (type) => {
     if(type === 'project') return 'bg-primary bg-opacity-10 text-primary border-primary border-opacity-10';
     if(type === 'experience') return 'bg-success bg-opacity-10 text-success border-success border-opacity-10';
     return 'bg-secondary bg-opacity-10 text-secondary border-secondary border-opacity-10';
  };

  return (
    <div className={`list-group-item p-0 border-light ${expanded ? 'bg-light bg-opacity-50' : 'bg-white'}`}>
      
      {/* --- SUMMARY ROW --- */}
      <div 
        className="d-flex align-items-start p-3 gap-3" 
        onClick={() => setExpanded(!expanded)}
        style={{ cursor: 'pointer' }}
      >
        {/* Status Icon */}
        <div className="mt-1">
          {isVerified && <div className="text-success"><Check size={20} /></div>}
          {status === 'pending' && <div className="text-warning"><div className="spinner-grow spinner-grow-sm" style={{width: '1rem', height: '1rem'}} /></div>}
          {isMissing && <div className="text-danger opacity-50"><X size={20} /></div>}
        </div>

        {/* Content */}
        <div className="flex-grow-1">
          <div className="d-flex align-items-center gap-2 mb-1 flex-wrap">
            {/* --- FIX: TYPE BADGE VISIBILITY --- */}
            <span className={`badge rounded-pill fw-normal ${getTypeBadgeClass(requirement_type)}`} style={{ fontSize: '0.65rem' }}>
              {formatType(requirement_type)}
            </span>
            {isApproved && <span className="badge bg-success text-white" style={{fontSize: '0.65rem'}}>Approved</span>}
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

        <div className="text-muted align-self-center">
          {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </div>

      {/* --- EXPANDED DETAILS --- */}
      {expanded && (
        <div className="px-3 pb-3 ps-5">
          <div className="border-start border-2 ps-3 py-1">
            
            {isMissing ? (
              <div className="bg-white p-3 rounded border border-dashed border-danger border-opacity-25">
                <p className="text-muted small mb-2">We couldn't find evidence of this.</p>
                <button 
                  onClick={(e) => { e.stopPropagation(); onLinkEvidence(); }}
                  className="btn btn-sm btn-outline-primary d-flex align-items-center gap-2"
                >
                  <LinkIcon size={14} /> Link Evidence
                </button>
              </div>
            ) : (
              <>
                {/* Header Actions */}
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span className="text-muted small">
                    Match Confidence: <strong>{Math.round(best_match_confidence * 100)}%</strong>
                  </span>
                  
                  <div className="d-flex gap-1">
                    {!isApproved && (
                      <button 
                        onClick={(e) => { e.stopPropagation(); onApprove(item.requirement_id); }}
                        className="btn btn-sm btn-light border text-success d-flex align-items-center gap-1"
                        title="Approve this match"
                      >
                        <ThumbsUp size={12} /> Approve
                      </button>
                    )}
                    <button 
                      onClick={(e) => { e.stopPropagation(); onReject(); }}
                      className="btn btn-sm btn-light border text-danger d-flex align-items-center gap-1"
                      title="Reject this match"
                    >
                      <Trash2 size={12} /> Reject
                    </button>
                  </div>
                </div>
                
                {/* --- EVIDENCE CARD (White Box) --- */}
                <div className={`bg-white p-3 rounded border mb-3 shadow-sm border-start-4 ${isApproved ? 'border-start-success bg-success bg-opacity-10' : 'border-start-primary'}`}>
                  
                  <div className="d-flex justify-content-between mb-2">
                    <span className="small fw-bold text-dark">{best_match_text}</span>
                    
                    {/* VIEW FULL TEXT BUTTON */}
                    <button 
                        onClick={(e) => { e.stopPropagation(); onViewEvidence(item); }}
                        className="btn btn-sm btn-link p-0 text-decoration-none d-flex align-items-center"
                        style={{fontSize: '0.75rem'}}
                    >
                        <Eye size={12} className="me-1"/> View Full Context
                    </button>
                  </div>

                  <div className="text-muted small fst-italic mb-2">
                    "{best_match_excerpt || match_summary}"
                  </div>
                  
                  {/* Lineage Chips */}
                  <div className="d-flex flex-wrap gap-1">
                    {lineage && lineage.map((step, idx) => (
                      <div key={idx} className="d-flex align-items-center">
                        <span className={`badge border fw-normal text-dark ${getLineageBadgeClass(step.type)}`}>
                          {step.name}
                        </span>
                        {idx < lineage.length - 1 && <span className="text-muted small mx-1">›</span>}
                      </div>
                    ))}
                  </div>
                </div>

                {/* --- ALTERNATIVES --- */}
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
                      <div className="mt-2 d-flex flex-column gap-2 ps-3 border-start">
                        {alternatives.map((alt, idx) => (
                          <div key={alt.id || idx} className="bg-light p-2 rounded border position-relative">
                            <div className="d-flex justify-content-between align-items-start mb-1">
                                <div 
                                    className="small text-truncate" 
                                    style={{ maxWidth: '75%', cursor: 'pointer' }}
                                    onClick={(e) => { e.stopPropagation(); onViewEvidence(item, alt.match_text); }}
                                    title="Click to view full context"
                                >
                                    <span className="fw-bold d-block text-dark" style={{ fontSize: '0.8rem' }}>{alt.source_name}</span>
                                    <span className="text-muted fst-italic" style={{ fontSize: '0.75rem' }}>"{alt.match_text.substring(0, 60)}..."</span>
                                </div>
                                
                                <button 
                                    onClick={(e) => { e.stopPropagation(); onPromote(item.requirement_id, alt.id); }}
                                    className="btn btn-sm btn-white border px-2 py-0 text-primary fw-bold d-flex align-items-center gap-1"
                                    style={{fontSize: '0.7rem'}}
                                >
                                    <ArrowUpCircle size={12} /> Promote
                                </button>
                            </div>
                            
                            <div className="d-flex flex-wrap gap-1 mb-2">
                                {alt.lineage && alt.lineage.map((step, sIdx) => (
                                    <span key={sIdx} className={`badge border fw-normal bg-white text-secondary`} style={{fontSize: '0.65rem'}}>
                                        {step.name}
                                    </span>
                                ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Footer */}
                <div className="d-flex gap-2 mt-2 border-top pt-2">
                  <button onClick={(e) => { e.stopPropagation(); onLinkEvidence(); }} className="btn btn-sm btn-link text-decoration-none text-muted p-0 d-flex align-items-center gap-1">
                    <LinkIcon size={12} /> Link Manual Evidence
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