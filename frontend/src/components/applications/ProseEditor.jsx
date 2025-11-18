// frontend/src/components/applications/ProseEditor.jsx
import React, { useMemo } from 'react';
import RichTextEditor from './RichTextEditor.jsx';

const ProseEditor = ({ 
    paragraph, 
    ideas, 
    pairMap,
    fullCV, 
    onSave 
}) => {
    
    // 1. Format Strategy Arguments
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
                    isStrategy: true // Mark as strategy
                }))
        }));
    }, [ideas, pairMap]);

    // 2. Identify "Strategic" CV Items (for highlighting in general lists)
    const strategyItemIds = useMemo(() => {
        const ids = new Set();
        ideas.forEach(idea => {
            idea.mapping_pair_ids.forEach(pid => {
                const pair = pairMap.get(pid);
                if(pair && pair.context_item_id) ids.add(pair.context_item_id);
            });
        });
        return ids;
    }, [ideas, pairMap]);

    // 3. Format Full CV Categories with "Strategy" Flags
    const cvCategories = useMemo(() => {
        if (!fullCV) return {};
        
        const format = (items, labelFn, detailFn) => (items || []).map(i => ({
            id: i.id,
            label: labelFn(i),
            detail: detailFn ? detailFn(i) : null,
            // The Magic: Check if this item is used in our strategy
            isStrategy: strategyItemIds.has(i.id)
        })).sort((a, b) => (b.isStrategy ? 1 : 0) - (a.isStrategy ? 1 : 0)); // Sort strategy items to top

        return {
            Experience: format(fullCV.experiences, i => `${i.title} @ ${i.company}`, i => `${i.start_date} - ${i.end_date}`),
            Education: format(fullCV.education, i => i.degree, i => i.institution),
            Skills: format(fullCV.skills, i => i.name, i => i.category),
            Achievements: format(fullCV.achievements, i => i.text.substring(0,40)+'...', i => "Achievement"),
            Hobbies: format(fullCV.hobbies, i => i.name, i => null),
        };
    }, [fullCV, strategyItemIds]);

    // 4. Generate "Smart Hints"
    const hints = useMemo(() => {
        const unmentioned = [];
        const currentText = (paragraph.draft_text || "").toLowerCase();
        
        ideas.forEach(idea => {
            // Check if the main idea title is mentioned
            // This is a naive check; a real one would check for chips
            // But for a "Hint", this is sufficient encouragement
            if (!currentText.includes(idea.title.toLowerCase())) {
                unmentioned.push(`Consider mentioning your "${idea.title}" strategy.`);
            }
        });
        return unmentioned;
    }, [ideas, paragraph.draft_text]);

    return (
        <div className="h-100">
             <RichTextEditor
                initialContent={paragraph.draft_text || ""}
                onUpdate={(html) => onSave(paragraph.id, { draft_text: html })}
                placeholder={`Draft your "${paragraph.purpose}" section...`}
                strategyArgs={strategyArgs}
                cvCategories={cvCategories}
                hints={hints} // <--- Pass hints
            />
        </div>
    );
};

export default ProseEditor;