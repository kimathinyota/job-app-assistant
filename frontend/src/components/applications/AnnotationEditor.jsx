// frontend/src/components/applications/AnnotationEditor.jsx
import React, { forwardRef } from 'react';
import IntelligentTextArea from './IntelligentTextArea.jsx';

const AnnotationEditor = forwardRef(({ 
    initialValue, 
    onSave, 
    fullCV, 
    onMention, 
    extraSuggestions = [] 
}, ref) => {
    return (
        <IntelligentTextArea
            ref={ref}
            initialValue={initialValue}
            onSave={onSave}
            cv={fullCV}
            extraSuggestions={extraSuggestions}
            onMention={onMention}
            placeholder="Type @ to link evidence or mappings..."
        />
    );
});

export default AnnotationEditor;