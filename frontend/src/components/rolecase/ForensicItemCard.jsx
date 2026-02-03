// frontend/src/components/rolecase/ForensicItemCard.jsx
import React from 'react';

export const ForensicItemCard = ({ item, onReject, onAdd }) => {
  const { requirement_text, status, match_summary, lineage } = item;
  const isMissing = status === "missing";
  const isVerified = status === "verified";

  // Chip Styles
  const typeColors = {
    experience: "bg-blue-100 text-blue-700 border-blue-200",
    project: "bg-purple-100 text-purple-700 border-purple-200",
    skill: "bg-emerald-100 text-emerald-700 border-emerald-200",
    education: "bg-indigo-100 text-indigo-700 border-indigo-200",
    default: "bg-slate-100 text-slate-700 border-slate-200"
  };

  return (
    <div className={`group relative bg-white p-4 rounded-lg shadow-sm border transition-all duration-200 hover:shadow-md 
      ${isMissing ? 'border-l-4 border-l-rose-400 border-slate-200 bg-slate-50/50' : ''}
      ${isVerified ? 'border-l-4 border-l-emerald-500 border-slate-200' : ''}
      ${status === 'pending' ? 'border-l-4 border-l-amber-400 border-slate-200' : ''}
    `}>
      
      {/* 1. Header & Actions */}
      <div className="flex justify-between items-start mb-2 gap-2">
        <div className={`text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-sm
          ${isMissing ? 'text-rose-600 bg-rose-50' : ''}
          ${isVerified ? 'text-emerald-600 bg-emerald-50' : ''}
          ${status === 'pending' ? 'text-amber-600 bg-amber-50' : ''}
        `}>
          {status}
        </div>

        {/* Hover Actions */}
        <div className="flex space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button 
            onClick={onAdd} 
            className="p-1.5 rounded bg-slate-100 hover:bg-blue-100 text-slate-400 hover:text-blue-600 transition-colors"
            title="Override/Add Manual Evidence"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
          </button>
          {!isMissing && (
            <button 
              onClick={onReject} 
              className="p-1.5 rounded bg-slate-100 hover:bg-rose-100 text-slate-400 hover:text-rose-600 transition-colors"
              title="Reject this match"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path></svg>
            </button>
          )}
        </div>
      </div>

      {/* 2. Requirement Text */}
      <p className={`text-sm font-medium leading-relaxed mb-3 ${isMissing ? 'text-slate-500' : 'text-slate-800'}`}>
        {requirement_text}
      </p>

      {/* 3. Evidence Block */}
      {!isMissing ? (
        <div className="bg-slate-50 p-2.5 rounded-md border border-slate-100">
          {/* A. Summary Text */}
          <div className="text-xs text-slate-600 italic mb-2 line-clamp-3">
            "{item.best_match_excerpt || match_summary}"
          </div>
          
          {/* B. Breadcrumbs */}
          <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-slate-200/50">
            {lineage && lineage.map((step, idx) => {
              const style = typeColors[step.type] || typeColors.default;
              return (
                <div key={idx} className="flex items-center max-w-full">
                  <button 
                    className={`px-2 py-0.5 rounded text-[10px] font-semibold border truncate max-w-[120px] hover:opacity-80 transition-opacity ${style}`}
                    title={step.name}
                  >
                    {step.name}
                  </button>
                  {idx < lineage.length - 1 && (
                    <span className="text-slate-300 text-[10px] mx-1">›</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : (
        <button 
          onClick={onAdd}
          className="w-full py-2 border-2 border-dashed border-slate-200 rounded text-xs text-slate-400 hover:border-blue-300 hover:text-blue-500 hover:bg-blue-50 transition-all flex items-center justify-center gap-2 group/btn"
        >
          <span>Find Missing Evidence</span>
          <span className="opacity-0 group-hover/btn:opacity-100">→</span>
        </button>
      )}
    </div>
  );
};