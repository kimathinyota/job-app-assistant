// frontend/src/components/applications/ProseEditor.jsx
// This is the *HEAVY* editor for writing prose.
import React, { useState } from 'react';
import IntelligentEditorBase from './IntelligentEditorBase.jsx';

const ProseEditor = ({ 
    paragraph, 
    ideas, 
    fullCV, 
    jobFeatures, 
    onSave, 
    onShowPreview 
}) => {
    const [focusedIdea, setFocusedIdea] = useState(null); // The Idea clicked in the rail

    const handleSave = (newText) => {
        // This is the "Writer" flow. It just saves the draft text.
        // The API call in the parent component handles the DB update.
        onSave(paragraph.id, { draft_text: newText });
    };

    return (
        <div className="d-flex flex-row gap-3">
            {/* 1. Main Editor */}
            <div style={{ flex: 3 }}>
                <IntelligentEditorBase
                    initialValue={paragraph.draft_text || ""}
                    onSave={handleSave}
                    fullCV={fullCV}
                    jobFeatures={jobFeatures}
                    onShowPreview={onShowPreview}
                    // Configuration for the HEAVY editor
                    enableAtLinking={true}
                    enableSlashCommands={true} // Enable / commands
                    enableStrategyRail={true}
                    focusedIdea={focusedIdea} // Pass the focused idea
                    placeholder={`Write prose for "${paragraph.purpose}" here...`}
                    minHeight="200px"
                />
            </div>
            
            {/* 2. Strategy Rail (The "Writer's" Helper) */}
            <div style={{ flex: 1, borderLeft: '1px solid #eee', paddingLeft: '1rem' }}>
                <h6 className="small text-muted fw-bold text-uppercase">Strategy Rail</h6>
                <p className="small text-muted" style={{fontSize: '0.8rem'}}>
                    Click an argument to filter evidence with the '@' command.
                </p>
                <div className="d-flex flex-column gap-1">
                    {ideas.length === 0 && (
                        <span className="small text-muted">No arguments defined.</span>
                    )}
                    {ideas.map(idea => (
                        <button
                            key={idea.id}
                            onClick={() => setFocusedIdea(idea)}
                            className={`btn btn-sm text-start ${
                                focusedIdea?.id === idea.id 
                                    ? 'btn-primary' 
                                    : 'btn-outline-secondary'
                            }`}
                        >
                            {idea.title}
                        </button>
                    ))}
                    {focusedIdea && (
                        <button
                            className="btn btn-sm btn-link text-muted"
                            onClick={() => setFocusedIdea(null)}
                        >
                            (Show all)
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default ProseEditor;