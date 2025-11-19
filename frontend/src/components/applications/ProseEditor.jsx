// frontend/src/components/applications/ProseEditor.jsx
import React, { useMemo } from 'react';
import RichTextEditor from './RichTextEditor.jsx';

const ProseEditor = ({ 
    paragraph, 
    ideas, 
    pairMap,
    fullCV,
    jobFeatures,
    onSave,
    onShowPreview 
}) => {
    
const strategyArgs = useMemo(() => {
        // Helper to find the full item object in the CV data
        const findFullItem = (id) => {
            if (!fullCV) return null;
            // Map backend keys to the 'type' strings your PreviewModal expects
            const sources = [
                { list: fullCV.experiences, type: 'experiences' },
                { list: fullCV.education, type: 'education' },
                { list: fullCV.projects, type: 'projects' },
                { list: fullCV.skills, type: 'skills' },
                { list: fullCV.achievements, type: 'achievements' },
                { list: fullCV.hobbies, type: 'hobbies' }
            ];
            
            for (const source of sources) {
                const found = source.list?.find(i => i.id === id);
                if (found) return { item: found, type: source.type };
            }
            return null;
        };

        return ideas.map(idea => {
            let linkedReq = null;
            
            // 1. Mapped Evidence
            const pairEvidence = (idea.mapping_pair_ids || [])
                .map(pid => pairMap.get(pid))
                .filter(Boolean)
                .map(p => {
                     if(p.feature_id && jobFeatures) linkedReq = jobFeatures.find(f => f.id === p.feature_id);
                     
                     const fullData = findFullItem(p.context_item_id);

                     return { 
                        id: `ev-${p.id}`, 
                        sourceId: p.context_item_id, 
                        label: p.context_item_text, 
                        detail: "Mapped Evidence", 
                        isStrategy: true,
                        requirement: p.feature_text,
                        reason: p.annotation,
                        type: 'mapping',
                        // --- DATA FOR PREVIEW ---
                        fullItem: fullData?.item,
                        categoryType: fullData?.type
                    };
                });

            // 2. Loose Evidence
            const looseEvidence = (idea.related_entity_ids || [])
                .map(eid => {
                    const fullData = findFullItem(eid);
                    if (!fullData) return null;

                    // Determine label based on type
                    let label = fullData.item.name || fullData.item.title || fullData.item.degree;
                    if (fullData.type === 'experiences') label = `${fullData.item.title} @ ${fullData.item.company}`;
                    
                    return { 
                        id: `loose-${eid}`, 
                        sourceId: eid, 
                        label: label, 
                        detail: fullData.type.charAt(0).toUpperCase() + fullData.type.slice(1), // Capitalize
                        type: 'loose',
                        isStrategy: true,
                        // --- DATA FOR PREVIEW ---
                        fullItem: fullData.item,
                        categoryType: fullData.type
                    };
                })
                .filter(Boolean);

            return {
                id: idea.id,
                title: idea.title,
                note: idea.annotation,
                requirementId: linkedReq?.id,
                requirementLabel: linkedReq?.description, 
                evidence: [...pairEvidence, ...looseEvidence]
            };
        });
    }, [ideas, pairMap, jobFeatures, fullCV]);

    const linkableItems = useMemo(() => {
        const items = [];
        (jobFeatures || []).forEach(f => {
            if (f.type === 'requirement' || f.type === 'qualification') {
                items.push({ id: f.id, label: f.description, type: 'requirement' });
            }
        });
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
                linkableItems={linkableItems} 
                onPreview={onShowPreview}
                sectionTitle={paragraph.purpose}
                fullCV={fullCV}
            />
        </div>
    );
};

export default ProseEditor;