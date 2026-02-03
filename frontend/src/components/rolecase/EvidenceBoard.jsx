import React from 'react';
import { ForensicItemCard } from './ForensicItemCard';

const COLUMNS = [
  { key: "Critical", label: "🚨 Critical Requirements", color: "bg-red-50 border-red-100" },
  { key: "High", label: "⭐ High Priority", color: "bg-orange-50 border-orange-100" },
  { key: "Standard", label: "🔹 Standard", color: "bg-blue-50 border-blue-100" },
  { key: "Bonus", label: "🎁 Bonus", color: "bg-gray-50 border-gray-100" }
];

export const EvidenceBoard = ({ groups, onReject, onOpenManual }) => {
  return (
    <div className="flex h-full space-x-4 min-w-max">
      {COLUMNS.map(col => {
        const items = groups[col.key] || [];
        return (
          <div key={col.key} className={`flex flex-col w-80 rounded-xl border ${col.color} h-full`}>
            {/* Column Header */}
            <div className="p-3 border-b border-gray-200/50 bg-white/50 backdrop-blur-sm rounded-t-xl">
              <h3 className="font-semibold text-gray-700 flex justify-between">
                {col.label}
                <span className="bg-gray-200 text-gray-600 text-xs px-2 py-0.5 rounded-full">
                  {items.length}
                </span>
              </h3>
            </div>

            {/* Scrollable List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-3">
              {items.map(item => (
                <ForensicItemCard 
                  key={item.requirement_id} 
                  item={item} 
                  onReject={() => onReject(item.requirement_id)}
                  onAdd={() => onOpenManual(item.requirement_id)}
                />
              ))}
              {items.length === 0 && (
                <div className="text-center text-gray-400 italic text-sm mt-10">
                  No items in this category.
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
};