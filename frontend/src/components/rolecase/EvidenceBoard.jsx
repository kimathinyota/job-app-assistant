// frontend/src/components/rolecase/EvidenceBoard.jsx
import React from 'react';
import { ForensicItemCard } from './ForensicItemCard';

const COLUMNS = [
  { key: "Critical", label: "Critical Requirements", sub: "Must Have", color: "border-t-rose-500", bg: "bg-rose-50/30" },
  { key: "High", label: "High Priority", sub: "Should Have", color: "border-t-orange-500", bg: "bg-orange-50/30" },
  { key: "Standard", label: "Standard", sub: "Nice to Have", color: "border-t-blue-500", bg: "bg-blue-50/30" },
  { key: "Bonus", label: "Bonus", sub: "Differentiators", color: "border-t-slate-400", bg: "bg-slate-50/50" }
];

export const EvidenceBoard = ({ groups, onReject, onOpenManual }) => {
  return (
    <div className="flex h-full gap-6 min-w-max pb-4">
      {COLUMNS.map(col => {
        const items = groups[col.key] || [];
        return (
          <div key={col.key} className={`flex flex-col w-[22rem] rounded-xl border border-slate-200 bg-slate-50/50 shadow-sm h-full overflow-hidden`}>
            
            {/* Column Header */}
            <div className={`p-4 bg-white border-b border-slate-100 border-t-4 ${col.color}`}>
              <div className="flex justify-between items-baseline mb-1">
                <h3 className="font-bold text-slate-800 text-sm">
                  {col.label}
                </h3>
                <span className="bg-slate-100 text-slate-600 text-xs font-bold px-2 py-0.5 rounded-full">
                  {items.length}
                </span>
              </div>
              <p className="text-xs text-slate-400 font-medium uppercase tracking-wider">{col.sub}</p>
            </div>

            {/* Scrollable List */}
            <div className={`flex-1 overflow-y-auto p-3 space-y-3 custom-scrollbar ${col.bg}`}>
              {items.map(item => (
                <ForensicItemCard 
                  key={item.requirement_id} 
                  item={item} 
                  onReject={() => onReject(item.requirement_id)}
                  onAdd={() => onOpenManual(item.requirement_id)}
                />
              ))}
              
              {items.length === 0 && (
                <div className="h-32 flex flex-col items-center justify-center text-slate-400 border-2 border-dashed border-slate-200 rounded-lg m-2">
                  <span className="text-sm">Empty</span>
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};