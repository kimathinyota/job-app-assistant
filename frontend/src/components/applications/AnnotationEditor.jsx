// frontend/src/components/applications/AnnotationEditor.jsx
// This is the *LITE* editor for notes.
import React from 'react';
import IntelligentEditorBase from './IntelligentEditorBase.jsx';

const AnnotationEditor = ({ initialValue, onSave, fullCV, onShowPreview }) => {
s
    return (
        <IntelligentEditorBase
            initialValue={initialValue}
            onSave={onSave}
            fullCV={fullCV}
            onShowPreview={onShowPreview}
            // Configuration for the LITE editor
            enableAtLinking={true}
            enableSlashCommands={false} // No slash commands in annotations
            enableStrategyRail={false} // No strategy rail
            placeholder="Add notes for the AI or your own reference... Type @ to link CV items..."
            minHeight="80px"
        />
    );
};

export default AnnotationEditor;