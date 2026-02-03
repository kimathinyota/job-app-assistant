import React, { useState } from 'react';

export const ManualMatchModal = ({ isOpen, onClose, onSubmit }) => {
  const [text, setText] = useState("");

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-md shadow-xl">
        <h3 className="text-lg font-bold mb-4">Add Manual Evidence</h3>
        <p className="text-sm text-gray-600 mb-4">
          Paste the text from your CV or describe the experience that proves you meet this requirement.
        </p>
        
        <textarea
          className="w-full border rounded-md p-3 text-sm min-h-[100px] focus:ring-2 focus:ring-blue-500 outline-none"
          placeholder="e.g., 'I have 3 years of React experience from my time at...'"
          value={text}
          onChange={(e) => setText(e.target.value)}
          autoFocus
        />

        <div className="flex justify-end space-x-3 mt-4">
          <button onClick={onClose} className="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded">
            Cancel
          </button>
          <button 
            onClick={() => onSubmit({ evidence_text: text })}
            disabled={!text.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            Save Match
          </button>
        </div>
      </div>
    </div>
  );
};