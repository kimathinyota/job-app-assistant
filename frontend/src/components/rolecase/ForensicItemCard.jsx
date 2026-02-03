import React, { useState } from 'react';

export const ForensicItemCard = ({ item, onReject, onAdd }) => {
  const { requirement_text, status, match_summary, lineage, authority_bucket } = item;
  const isMissing = status === "missing";

  // Chip Colors
  const typeColors = {
    experience: "bg-blue-100 text-blue-800",
    project: "bg-purple-100 text-purple-800",
    skill: "bg-emerald-100 text-emerald-800",
    education: "bg-indigo-100 text-indigo-800",
    default: "bg-gray-100 text-gray-800"
  };

  return (
    <div className={`bg-white p-3 rounded-lg shadow-sm border transition-all hover:shadow-md ${isMissing ? 'border-l-4 border-l-red-400 opacity-90' : 'border-gray-200'}`}>
      
      {/* 1. Header & Status */}
      <div className="flex justify-between items-start mb-2">
        <span className={`text-xs font-bold uppercase px-1.5 py-0.5 rounded ${isMissing ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
          {status}
        </span>
        {/* Actions */}
        <div className="flex space-x-1">
          <button onClick={onAdd} className="text-gray-400 hover:text-blue-600 p-1" title="Manual Match">
            ➕
          </button>
          {!isMissing && (
            <button onClick={onReject} className="text-gray-400 hover:text-red-600 p-1" title="Reject Match">
              ✕
            </button>
          )}
        </div>
      </div>

      {/* 2. Requirement Text */}
      <p className="text-sm text-gray-800 font-medium mb-3">
        {requirement_text}
      </p>

      {/* 3. Evidence Area */}
      {!isMissing ? (
        <div className="bg-gray-50 p-2 rounded text-xs border border-gray-100">
          {/* A. Summary Text */}
          <div className="text-gray-600 italic mb-2">
            "{item.best_match_excerpt || match_summary}"
          </div>
          
          {/* B. Interactive Lineage Chips */}
          <div className="flex flex-wrap gap-1">
            {lineage && lineage.map((step, idx) => {
              // Simple heuristic for coloring based on type
              const colorClass = typeColors[step.type] || typeColors.default;
              return (
                <span 
                  key={idx} 
                  className={`px-1.5 py-0.5 rounded text-[10px] font-semibold flex items-center ${colorClass}`}
                >
                  {step.name}
                  {idx < lineage.length - 1 && <span className="ml-1 text-gray-400">›</span>}
                </span>
              );
            })}
          </div>
        </div>
      ) : (
        <div className="text-xs text-red-500 flex items-center">
          <span className="mr-1">⚠️</span> No evidence found in CV.
        </div>
      )}
    </div>
  );
};