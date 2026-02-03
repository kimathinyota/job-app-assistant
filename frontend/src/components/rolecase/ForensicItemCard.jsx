// frontend/src/components/rolecase/ForensicItemCard.jsx
import React, { useState } from 'react';
import { 
  Check, X, Link as LinkIcon, Trash2, Layers, 
  ArrowUpCircle, Eye, ThumbsUp, AlertCircle 
} from 'lucide-react';

export const ForensicItemCard = ({ item, onReject, onAdd, onPromote, onApprove, onViewExtract }) => {
  const { requirement_text, status, match_summary, lineage, alternatives, best_match_confidence } = item;
  
  const [showAlternatives, setShowAlternatives] = useState(false);
  
  const isMissing = status === "missing";
  const isVerified = status === "verified";
  const isApproved = best_match_confidence === 1.0;

  // Type Colors Helper
  const getTypeColor = (type) => {
    switch(type) {
      case 'experience': return "bg-primary bg-opacity-10 text-primary border-primary border-opacity-10";
      case 'project': return "bg-info bg-opacity-10 text-info border-info border-opacity-10";
      case 'skill': return "bg-success bg-opacity-10 text-success border-success border-opacity-10";
      default: return "bg-secondary bg-opacity-10 text-secondary border-secondary border-opacity-10";
    }
  };

  return (
    <div className={`card border shadow-sm mb-3 transition-all 
      ${isMissing ? 'border-start-4 border-danger bg-white' : ''} 
      ${isVerified ? 'border-start-4 border-success' : ''}`}>
      
      <div className="card-body p-3">
        {/* Header Row */}
        <div className="d-flex justify-content-between align-items-start mb-2">
          <div className="d-flex align-items-center gap-2">
            {isMissing ? (
              <span className="badge bg-danger bg-opacity-10 text-danger border border-danger border-opacity-25">MISSING</span>
            ) : (
              <span className={`badge ${isApproved ? 'bg-success text-white' : 'bg-success bg-opacity-10 text-success'} border border-success border-opacity-25`}>
                {isApproved ? <><ThumbsUp size={10} className="me-1"/> APPROVED</> : `MATCH: ${Math.round(best_match_confidence * 100)}%`}
              </span>
            )}
          </div>
          
          {/* Quick Add Button (Always visible) */}
          <button className="btn btn-sm btn-light border py-0 px-2 text-primary" onClick={onAdd} title="Manual Match">
             <LinkIcon size={14} />
          </button>
        </div>

        <p className="card-text small fw-bold text-dark mb-3">{requirement_text}</p>

        {/* --- MAIN MATCH SECTION --- */}
        {!isMissing && (
          <div className={`p-2 rounded border mb-2 ${isApproved ? 'bg-success bg-opacity-10 border-success border-opacity-25' : 'bg-light border'}`}>
            <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="small fw-bold text-success">Selected Evidence</span>
                <div className="d-flex gap-1">
                    {!isApproved && (
                        <button onClick={() => onApprove(item.requirement_id)} className="btn btn-sm btn-light py-0 px-2 text-success border" title="Approve Match">
                            <ThumbsUp size={12} />
                        </button>
                    )}
                    <button onClick={() => onReject(item.requirement_id)} className="btn btn-sm btn-light py-0 px-2 text-danger border" title="Reject Match">
                        <Trash2 size={12} />
                    </button>
                </div>
            </div>
            
            {/* Snippet */}
            <div className="small text-muted fst-italic mb-2 border-start border-3 border-secondary ps-2">
              "{item.best_match_excerpt || match_summary}"
            </div>
            
            {/* View Full Extract Button */}
            <button onClick={() => onViewExtract(item)} className="btn btn-link btn-sm p-0 text-decoration-none text-muted mb-2" style={{fontSize: '0.75rem'}}>
                <Eye size={10} className="me-1"/> View Full Text
            </button>

            {/* Lineage Chips */}
            <div className="d-flex flex-wrap gap-1">
              {lineage && lineage.map((step, idx) => (
                <div key={idx} className="d-flex align-items-center">
                  <span className={`badge border fw-normal ${getTypeColor(step.type)}`}>
                    {step.name}
                  </span>
                  {idx < lineage.length - 1 && <span className="text-muted small mx-1">›</span>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* --- ALTERNATIVES TOGGLE --- */}
        {alternatives && alternatives.length > 0 && (
            <button 
                onClick={() => setShowAlternatives(!showAlternatives)}
                className="btn btn-sm btn-outline-secondary border-dashed w-100 mb-2 d-flex align-items-center justify-content-center gap-2"
                style={{fontSize: '0.75rem'}}
            >
                <Layers size={12} /> 
                {showAlternatives ? "Hide Alternatives" : `View ${alternatives.length} Other Matches`}
            </button>
        )}

        {/* --- ALTERNATIVES LIST --- */}
        {showAlternatives && (
            <div className="d-flex flex-column gap-2 mb-2 ps-2 border-start border-3">
                {alternatives.map((alt) => (
                    <div key={alt.id} className="bg-white p-2 rounded border shadow-sm position-relative">
                        <div className="d-flex justify-content-between align-items-start mb-1">
                            <span className="badge bg-secondary bg-opacity-10 text-dark border">{Math.round(alt.score * 100)}%</span>
                            <button 
                                onClick={() => onPromote(item.requirement_id, alt.id)}
                                className="btn btn-sm btn-primary py-0 px-2 fw-bold"
                                style={{fontSize: '0.7rem'}}
                                title="Promote to Best Match"
                            >
                                <ArrowUpCircle size={12} className="me-1"/> Promote
                            </button>
                        </div>
                        
                        <div className="small text-muted fst-italic mb-1 text-truncate">"{alt.match_text}"</div>
                        
                        {/* Alternative Lineage Chips */}
                        <div className="d-flex flex-wrap gap-1">
                            {alt.lineage && alt.lineage.map((step, idx) => (
                                <span key={idx} className={`badge border fw-normal ${getTypeColor(step.type)}`} style={{fontSize: '0.6rem'}}>
                                    {step.name}
                                </span>
                            ))}
                        </div>
                    </div>
                ))}
            </div>
        )}

        {/* --- MISSING STATE --- */}
        {isMissing && (
          <div className="d-grid mt-2">
            <button onClick={onAdd} className="btn btn-sm btn-outline-danger border-dashed d-flex align-items-center justify-content-center gap-2">
               <AlertCircle size={14} /> Did we miss this? Add Link
            </button>
          </div>
        )}
        
        {/* "Did we miss any?" Footer */}
        {!isMissing && !showAlternatives && (
             <div className="text-center mt-1">
                <button onClick={onAdd} className="btn btn-link btn-sm text-muted text-decoration-none" style={{fontSize: '0.7rem'}}>
                    Did we miss a better match?
                </button>
             </div>
        )}

      </div>
    </div>
  );
};