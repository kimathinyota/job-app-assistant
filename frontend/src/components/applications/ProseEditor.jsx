// frontend/src/components/applications/ProseEditor.jsx
import React, { useMemo } from 'react';
import RichTextEditor from './RichTextEditor.jsx';

const ProseEditor = ({ 
    paragraph, 
    ideas, 
    pairMap,
    fullCV,
    jobFeatures, // <--- Make sure this prop is passed!
    onSave,
    onShowPreview 
}) => {
    
    const strategyArgs = useMemo(() => {
        return ideas.map(idea => ({
            id: idea.id,
            title: idea.title,
            evidence: idea.mapping_pair_ids
                .map(pid => pairMap.get(pid))
                .filter(Boolean)
                .map(p => ({ 
                    id: `ev-${p.id}`, 
                    label: p.context_item_text,
                    detail: "Evidence",
                    isStrategy: true 
                }))
        }));
    }, [ideas, pairMap]);

    // --- NEW: Build Linkable Items for Codex Sections ---
    const linkableItems = useMemo(() => {
        const items = [];
        // 1. Requirements
        (jobFeatures || []).forEach(f => {
            if (f.type === 'requirement' || f.type === 'qualification') {
                items.push({ id: f.id, label: f.description, type: 'requirement' });
            }
        });
        // 2. Strategy Arguments
        ideas.forEach(idea => {
            items.push({ id: idea.id, label: idea.title, type: 'strategy' });
        });
        return items;
    }, [jobFeatures, ideas]);

    const cvCategories = useMemo(() => {
        if (!fullCV) return {};
        const format = (items, labelFn, detailFn) => (items || []).map(i => ({
            id: i.id,
            label: labelFn(i),
            detail: detailFn ? detailFn(i) : null,
        })).sort((a, b) => a.label.localeCompare(b.label));

        return {
            Experience: format(fullCV.experiences, i => `${i.title} @ ${i.company}`, i => `${i.start_date} - ${i.end_date}`),
            Education: format(fullCV.education, i => i.degree, i => i.institution),
            Skills: format(fullCV.skills, i => i.name, i => i.category),
            Achievements: format(fullCV.achievements, i => i.text.substring(0,40)+'...', i => "Achievement"),
            Hobbies: format(fullCV.hobbies, i => i.name, i => null),
        };
    }, [fullCV]);

    return (
        <div className="h-100">
             <RichTextEditor
                initialContent={paragraph.draft_text || ""}
                onUpdate={(html) => onSave(paragraph.id, { draft_text: html })}
                placeholder={`Draft your "${paragraph.purpose}" section...`}
                strategyArgs={strategyArgs}
                cvCategories={cvCategories}
                linkableItems={linkableItems} // <--- Pass the new prop
                onPreview={onShowPreview}
            />
        </div>
    );
};

export default ProseEditor;