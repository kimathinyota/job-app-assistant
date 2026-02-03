// frontend/src/components/rolecase/ForensicHeader.jsx
import React from 'react';

export const ForensicHeader = ({ stats }) => {
  const { 
    overall_match_score, 
    coverage_pct, 
    critical_gaps_count, 
    evidence_sources 
  } = stats;

  const getScoreColor = (score) => {
    if (score >= 80) return "text-emerald-600";
    if (score >= 50) return "text-amber-600";
    return "text-rose-600";
  };

  return (
    <div className="bg-white border-b border-slate-200 px-8 py-5 shadow-sm flex items-center justify-between shrink-0 sticky top-0 z-20">
      
      {/* LEFT: The Big Score */}
      <div className="flex items-center space-x-6">
        <div className="text-center">
          <div className={`text-4xl font-bold tracking-tight ${getScoreColor(overall_match_score)}`}>
            {overall_match_score}%
          </div>
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mt-1">Match Score</div>
        </div>
        
        <div className="h-12 w-px bg-slate-100 mx-2"></div>
        
        <div className="text-center min-w-[80px]">
          <div className="text-2xl font-semibold text-slate-700">{coverage_pct}%</div>
          <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mt-1">Coverage</div>
        </div>
      </div>

      {/* CENTER: Critical Alerts */}
      <div className={`flex items-center space-x-4 px-5 py-3 rounded-xl border ${critical_gaps_count > 0 ? 'bg-rose-50 border-rose-100' : 'bg-emerald-50 border-emerald-100'}`}>
        <div className={`flex items-center justify-center h-10 w-10 rounded-full ${critical_gaps_count > 0 ? 'bg-white text-rose-500 shadow-sm' : 'bg-white text-emerald-500 shadow-sm'}`}>
          {critical_gaps_count > 0 ? (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
          ) : (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          )}
        </div>
        <div>
          <div className={`font-bold text-sm ${critical_gaps_count > 0 ? 'text-rose-800' : 'text-emerald-800'}`}>
            {critical_gaps_count === 0 ? "All Clear" : `${critical_gaps_count} Critical Gaps`}
          </div>
          <div className="text-xs text-slate-600 mt-0.5">
            {critical_gaps_count > 0 ? "Must resolve to pass screening." : "No auto-rejection risks detected."}
          </div>
        </div>
      </div>

      {/* RIGHT: Authority DNA */}
      <div className="flex space-x-8">
        {[
          { label: "Professional", count: evidence_sources.Professional || 0, color: "text-blue-600", bg: "bg-blue-50" },
          { label: "Academic", count: evidence_sources.Academic || 0, color: "text-purple-600", bg: "bg-purple-50" },
          { label: "Personal", count: evidence_sources.Personal || 0, color: "text-orange-600", bg: "bg-orange-50" }
        ].map((source) => (
          <div key={source.label} className="text-center group">
            <div className={`mx-auto h-8 w-12 rounded-md flex items-center justify-center font-bold text-sm mb-1 transition-colors ${source.bg} ${source.color}`}>
              {source.count}
            </div>
            <div className="text-[10px] font-medium text-slate-400 uppercase tracking-wide group-hover:text-slate-600 transition-colors">
              {source.label}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};