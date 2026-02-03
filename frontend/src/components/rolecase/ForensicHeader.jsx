// frontend/src/components/rolecase/ForensicHeader.jsx
import React from 'react';
import { ShieldAlert, ShieldCheck, Briefcase, GraduationCap, User } from 'lucide-react';

export const ForensicHeader = ({ stats }) => {
  const { 
    overall_match_score, 
    coverage_pct, 
    critical_gaps_count, 
    evidence_sources 
  } = stats;

  const getScoreClass = (score) => {
    if (score >= 80) return "text-success";
    if (score >= 50) return "text-warning";
    return "text-danger";
  };

  return (
    <div className="bg-white border-bottom px-4 py-3 shadow-sm d-flex align-items-center justify-content-between sticky-top z-2">
      
      {/* LEFT: Score */}
      <div className="d-flex align-items-center gap-4">
        <div className="text-center">
          <div className={`display-6 fw-bold mb-0 ${getScoreClass(overall_match_score)}`}>
            {overall_match_score}%
          </div>
          <div className="text-uppercase text-muted fw-bold" style={{fontSize: '0.65rem', letterSpacing: '1px'}}>Match Score</div>
        </div>
        
        <div className="vr mx-2 text-muted opacity-25"></div>
        
        <div className="text-center">
          <div className="h4 fw-bold text-secondary mb-0">{coverage_pct}%</div>
          <div className="text-uppercase text-muted fw-bold" style={{fontSize: '0.65rem', letterSpacing: '1px'}}>Coverage</div>
        </div>
      </div>

      {/* CENTER: Critical Alerts */}
      <div className={`d-flex align-items-center gap-3 px-4 py-2 rounded-3 border ${critical_gaps_count > 0 ? 'bg-danger bg-opacity-10 border-danger border-opacity-25' : 'bg-success bg-opacity-10 border-success border-opacity-25'}`}>
        <div className={critical_gaps_count > 0 ? 'text-danger' : 'text-success'}>
          {critical_gaps_count > 0 ? <ShieldAlert size={28} /> : <ShieldCheck size={28} />}
        </div>
        <div>
          <div className={`fw-bold small ${critical_gaps_count > 0 ? 'text-danger' : 'text-success'}`}>
            {critical_gaps_count === 0 ? "All Clear" : `${critical_gaps_count} Critical Gaps`}
          </div>
          <div className="text-muted small" style={{fontSize: '0.75rem'}}>
            {critical_gaps_count > 0 ? "Must resolve to pass." : "No auto-rejection risks."}
          </div>
        </div>
      </div>

      {/* RIGHT: Authority DNA */}
      <div className="d-flex gap-4">
        {[
          { label: "Professional", count: evidence_sources.Professional || 0, color: "text-primary", icon: Briefcase },
          { label: "Academic", count: evidence_sources.Academic || 0, color: "text-info", icon: GraduationCap },
          { label: "Personal", count: evidence_sources.Personal || 0, color: "text-warning", icon: User }
        ].map((source) => (
          <div key={source.label} className="text-center">
            <div className={`h5 fw-bold mb-0 ${source.color}`}>
              {source.count}
            </div>
            <div className="text-uppercase text-muted fw-bold" style={{fontSize: '0.65rem'}}>{source.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
};