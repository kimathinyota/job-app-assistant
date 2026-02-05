// frontend/src/components/rolecase/ForensicItemCard.jsx
import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Check, X, Link as LinkIcon, Trash2, Layers, ThumbsUp, ArrowUpCircle, Eye } from 'lucide-react';

export const ForensicItemCard = ({ item, onReject, onLinkEvidence, onPromote, onApprove, onViewEvidence }) => {
  const [expanded, setExpanded] = useState(false);
  const [showAlternatives, setShowAlternatives] = useState(false);
  
  const { 
    requirement_text, requirement_type, status, match_summary, lineage, 
    best_match_text, best_match_confidence, best_match_excerpt, alternatives 
  } = item;

  const isMissing = status === "missing";
  const isVerified = status === "verified";
  const isApproved = best_match_confidence === 1.0;

  const formatType = (t) => t ? t.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Requirement';
  
  const getTypeBadgeClass = (t) => {
    switch(t) {
      case 'hard_skill': return 'bg-primary bg-opacity-10 text-primary border border-primary border-opacity-10';
      case 'soft_skill': return 'bg-info bg-opacity-10 text-info border border-info border-opacity-10';
      default: return 'bg-light text-secondary border';
    }
  };

  const getLineageBadgeClass = (type) => {
     if(type === 'project') return 'bg-primary bg-opacity-10 text-primary border-primary border-opacity-10 cursor-pointer';
     if(type === 'experience') return 'bg-success bg-opacity-10 text-success border-success border-opacity-10 cursor-pointer';
     return 'bg-secondary bg-opacity-10 text-secondary border-secondary border-opacity-10 cursor-pointer';
  };

  return (
    <div className={`list-group-item p-0 border-light ${expanded ? 'bg-light bg-opacity-50' : 'bg-white'}`}>
      
      {/* 1. Summary Row */}
      <div className="d-flex align-items-start p-3 gap-3 cursor-pointer" onClick={() => setExpanded(!expanded)}>
        <div className="mt-1">
          {isVerified && <div className="text-success"><Check size={20} /></div>}
          {status === 'pending' && <div className="text-warning"><div className="spinner-grow spinner-grow-sm" /></div>}
          {isMissing && <div className="text-danger opacity-50"><X size={20} /></div>}
        </div>
        <div className="flex-grow-1">
          <div className="d-flex align-items-center gap-2 mb-1">
            <span className={`badge rounded-pill fw-normal ${getTypeBadgeClass(requirement_type)}`} style={{fontSize: '0.65rem'}}>
              {formatType(requirement_type)}
            </span>
            {isApproved && <span className="badge bg-success text-white" style={{fontSize: '0.65rem'}}>Approved</span>}
          </div>
          <div className={`fw-medium ${isMissing ? 'text-secondary' : 'text-dark'}`}>{requirement_text}</div>
          
          {!expanded && !isMissing && (
            <div className="text-muted small mt-1 d-flex align-items-center" style={{fontSize: '0.75rem'}}>
              <LinkIcon size={10} className="me-1"/> 
              Found in: <span className="fw-semibold ms-1">{best_match_text || "Resume"}</span>
              {alternatives?.length > 0 && <span className="ms-2 badge bg-light text-secondary border">+ {alternatives.length} others</span>}
            </div>
          )}
        </div>
        <div className="text-muted align-self-center">{expanded ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}</div>
      </div>

      {/* 2. Expanded Content */}
      {expanded && (
        <div className="px-3 pb-3 ps-5">
          <div className="border-start border-2 ps-3 py-1">
            
            {isMissing ? (
              <div className="bg-white p-3 rounded border border-dashed border-danger border-opacity-25">
                <p className="text-muted small mb-2">Evidence missing.</p>
                <button onClick={(e) => {e.stopPropagation(); onLinkEvidence();}} className="btn btn-sm btn-outline-primary d-flex align-items-center gap-2">
                  <LinkIcon size={14}/> Link Evidence
                </button>
              </div>
            ) : (
              <>
                <div className="d-flex justify-content-between align-items-center mb-2">
                  <span className="text-muted small">Match Confidence: <strong>{Math.round(best_match_confidence * 100)}%</strong></span>
                  <div className="d-flex gap-1">
                    {!isApproved && <button onClick={(e) => {e.stopPropagation(); onApprove(item.requirement_id);}} className="btn btn-sm btn-light border text-success"><ThumbsUp size={12}/> Approve</button>}
                    <button onClick={(e) => {e.stopPropagation(); onReject();}} className="btn btn-sm btn-light border text-danger"><Trash2 size={12}/> Reject</button>
                  </div>
                </div>
                
                {/* --- Best Match Card --- */}
                <div className={`bg-white p-3 rounded border mb-3 shadow-sm border-start-4 ${isApproved ? 'border-start-success' : 'border-start-primary'}`}>
                  <div className="d-flex justify-content-between mb-2">
                    <span className="small fw-bold text-dark">{best_match_text}</span>
                    <button 
                        onClick={(e) => {e.stopPropagation(); onViewEvidence(lineage?.[0], best_match_excerpt);}} 
                        className="btn btn-sm btn-link p-0 text-decoration-none text-muted d-flex align-items-center" 
                        title="View Full Context"
                    >
                        <Eye size={12} className="me-1"/> View
                    </button>
                  </div>
                  <div className="text-muted small fst-italic mb-2">"{best_match_excerpt || match_summary}"</div>
                  
                  {/* Clickable Lineage Chips */}
                  <div className="d-flex flex-wrap gap-1">
                    {lineage && lineage.map((step, idx) => (
                      <div key={idx} className="d-flex align-items-center">
                        <button 
                            className={`badge border fw-normal text-dark ${getLineageBadgeClass(step.type)}`}
                            onClick={(e) => {e.stopPropagation(); onViewEvidence(step, best_match_excerpt);}}
                            title={`View this ${step.type}`}
                        >
                          {step.name}
                        </button>
                        {idx < lineage.length - 1 && <span className="text-muted small mx-1">›</span>}
                      </div>
                    ))}
                  </div>
                </div>

                {/* --- Alternatives --- */}
                {alternatives?.length > 0 && (
                  <div className="mb-3">
                    <button onClick={(e) => {e.stopPropagation(); setShowAlternatives(!showAlternatives);}} className="btn btn-sm btn-light border w-100 text-secondary">
                      <Layers size={14} className="me-2"/> {showAlternatives ? "Hide" : `See ${alternatives.length} Alternatives`}
                    </button>

                    {showAlternatives && (
                      <div className="mt-2 d-flex flex-column gap-2 ps-3 border-start">
                        {alternatives.map((alt, idx) => (
                          <div key={alt.id || idx} className="bg-light p-2 rounded border position-relative">
                            <div className="d-flex justify-content-between align-items-start mb-1">
                                <div className="small text-truncate" style={{maxWidth: '70%'}}>
                                    <span className="fw-bold d-block text-dark">{alt.source_name}</span>
                                    <span className="text-muted fst-italic">"{alt.match_text?.substring(0, 50)}..."</span>
                                </div>
                                <div className="d-flex align-items-center gap-2">
                                    <span className="badge bg-white text-secondary border">{Math.round(alt.score * 100)}%</span>
                                    
                                    {/* Action: View Alternative Context */}
                                    <button 
                                        onClick={(e) => {e.stopPropagation(); onViewEvidence(alt.lineage?.[0], alt.match_text);}}
                                        className="btn btn-sm btn-white border px-1 py-0 text-muted"
                                        title="View Context"
                                    >
                                        <Eye size={12}/>
                                    </button>
                                    
                                    {/* Action: Promote */}
                                    <button 
                                        onClick={(e) => {e.stopPropagation(); onPromote(item.requirement_id, alt.id);}}
                                        className="btn btn-sm btn-white border px-1 py-0 text-primary fw-bold"
                                        title="Promote to Best Match"
                                    >
                                        <ArrowUpCircle size={12}/>
                                    </button>
                                </div>
                            </div>
                            
                            {/* Alternative Lineage (Clickable) */}
                            <div className="d-flex flex-wrap gap-1">
                                {alt.lineage && alt.lineage.map((step, sIdx) => (
                                    <button 
                                        key={sIdx} 
                                        className="badge border fw-normal bg-white text-secondary cursor-pointer"
                                        style={{fontSize: '0.65rem'}}
                                        onClick={(e) => {e.stopPropagation(); onViewEvidence(step, alt.match_text);}}
                                    >
                                        {step.name}
                                    </button>
                                ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};