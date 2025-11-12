// frontend/src/components/applications/Step2_GenerateCV.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { 
    generateCvPrompt,
    inferMappingPairs,  // 1. Import the inferer
    addMappingPair      // 2. Import add pair
} from '../../api/applicationClient'; // <-- FIX: Removed .js extension
import PromptModal from './PromptModal'; // <-- FIX: Removed .jsx extension
import CVItemDisplayCard from './CVItemDisplayCard'; // <-- FIX: Removed .jsx extension
import PromoteItemModal from './PromoteItemModal'; // <-- FIX: Removed .jsx extension
import { getCVDisplayName } from '../../utils/cvHelpers'; // <--- IMPORT

// --- 3. NEW "AI GHOST" COMPONENT ---
// This card is for unmapped items that HAVE an AI suggestion.
const AISuggestionCard = ({ item, itemType, suggestion, onAccept, onIgnore, isAccepting }) => {
    let itemTitle = item.title || item.name || item.degree || 'Unknown Item';
    if (itemType === 'experiences') itemTitle = `${item.title} @ ${item.company}`;
    if (itemType === 'education') itemTitle = `${item.degree} @ ${item.institution}`;

    return (
        <div 
            className="card card-body p-3 mb-2 border-dashed border-info" 
            style={{ opacity: 0.9 }}
        >
            <div className="d-flex justify-content-between align-items-center mb-2">
                <span className="fst-italic fw-medium">{itemTitle}</span>
            </div>

            {/* AI Suggestion Details */}
            <div className="alert alert-info p-2" role="alert">
                <strong className="d-block small">💡 AI Suggestion:</strong>
                <p className="small mb-1">
                    <strong>Matches:</strong> {suggestion.feature_text}
                </p>
                <p className="small fst-italic mb-0">
                    <strong>Reason:</strong> {suggestion.annotation || "Good conceptual match."}
                </p>
            </div>

            <div className="d-flex justify-content-end gap-2">
                <button
                    type="button"
                    className="btn btn-sm btn-outline-secondary"
                    onClick={() => onIgnore(suggestion.id)} // Use suggestion's temp ID
                    disabled={isAccepting}
                >
                    Ignore
                </button>
                <button
                    type="button"
                    className="btn btn-sm btn-success"
                    onClick={() => onAccept(suggestion)}
                    disabled={isAccepting}
                >
                    {isAccepting ? "Adding..." : "Accept & Add"}
                </button>
            </div>
        </div>
    );
};


// --- ORIGINAL "GHOST" COMPONENT ---
// This card is for unmapped items that HAVE NO AI suggestion.
const GhostItemCard = ({ item, itemType, onPromote }) => {
  let itemTitle = item.title || item.name || item.degree || 'Unknown Item';
  if (itemType === 'experiences') itemTitle = `${item.title} @ ${item.company}`;
  if (itemType === 'education') itemTitle = `${item.degree} @ ${item.institution}`;

  return (
    <div 
        className="card card-body p-2 mb-2 border-dashed" 
        style={{ opacity: 0.6 }}
    >
        <div className="d-flex justify-content-between align-items-center">
            <span className="fst-italic">{itemTitle}</span>
            <button 
                type="button" 
                className="btn btn-sm btn-outline-primary"
                onClick={() => onPromote(item, itemType)}
            >
                + Map Manually...
            </button>
        </div>
    </div>
  );
};


const Step2_GenerateCV = ({ job, cv, mapping, onPrev, onNext, onMappingChanged }) => {
    const [cvPromptJson, setCvPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);

    const [isPromoteModalOpen, setIsPromoteModalOpen] = useState(false);
    const [itemToPromote, setItemToPromote] = useState(null); 
    const [isSubmittingManual, setIsSubmittingManual] = useState(false);

    // --- 4. ADD STATE FOR AI SUGGESTIONS ---
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
    const [allSuggestions, setAllSuggestions] = useState([]);
    const [suggestionError, setSuggestionError] = useState(null);
    const [isAccepting, setIsAccepting] = useState(null); // Tracks the ID of the suggestion being accepted

    // --- 5. FETCH SUGGESTIONS ON LOAD ---
    useEffect(() => {
        setIsLoadingSuggestions(true);
        setSuggestionError(null);
        // Use 'eager_mode' to show more possibilities in the review step
        inferMappingPairs(mapping.id, "eager_mode")
            .then(res => {
                // Give each suggestion a unique temp ID for React keys & state
                setAllSuggestions(res.data.map((s, i) => ({ ...s, id: `sugg-${i}` })));
            })
            .catch(err => {
                console.error("Failed to fetch suggestions:", err);
                setSuggestionError("Could not load AI suggestions.");
            })
            .finally(() => {
                setIsLoadingSuggestions(false);
            });
    }, [mapping.id]);


    // --- 6. UPDATE useMemo TO TRIAGE ALL CV ITEMS ---
    const {
        mappedExperiences,
        mappedProjects,
        mappedEducation,
        mappedHobbies,
        // NEW LISTS FOR SUGGESTIONS
        suggestedExperiences,
        suggestedProjects,
        suggestedEducation,
        suggestedHobbies,
        // RENAMED LISTS for other unmapped items
        otherUnmappedExperiences,
        otherUnmappedProjects,
        otherUnmappedEducation,
        otherUnmappedHobbies,
        aggregatedSkillIds, 
        groupedSkills
    } = useMemo(() => {
        const mappedItemIds = new Set(
            mapping.pairs.map(p => p.context_item_id)
        );
        
        // Create a quick lookup map for suggestions by *their* context_item_id
        const suggestionMap = new Map(
            allSuggestions.map(s => [s.context_item_id, s])
        );

        // Triage helper function
        const triageList = (cvList) => {
            const mapped = [];
            const suggested = [];
            const otherUnmapped = [];

            for (const item of cvList) {
                if (mappedItemIds.has(item.id)) {
                    mapped.push(item);
                } else if (suggestionMap.has(item.id)) {
                    // It's not mapped, but the AI has a suggestion for it.
                    // Attach the suggestion object to the item.
                    // We need to create a copy to avoid mutation issues
                    const itemWithSuggestion = { ...item, suggestion: suggestionMap.get(item.id) };
                    suggested.push(itemWithSuggestion);
                } else {
                    // Not mapped, and no suggestion.
                    otherUnmapped.push(item);
                }
            }
            return { mapped, suggested, otherUnmapped };
        };

        const { mapped: mappedExperiences, suggested: suggestedExperiences, otherUnmapped: otherUnmappedExperiences } = triageList(cv.experiences);
        const { mapped: mappedProjects, suggested: suggestedProjects, otherUnmapped: otherUnmappedProjects } = triageList(cv.projects);
        const { mapped: mappedEducation, suggested: suggestedEducation, otherUnmapped: otherUnmappedEducation } = triageList(cv.education);
        const { mapped: mappedHobbies, suggested: suggestedHobbies, otherUnmapped: otherUnmappedHobbies } = triageList(cv.hobbies);

        // Skill logic (unchanged)
        let skillIds = new Set();
        const allMappedItems = [...mappedExperiences, ...mappedProjects, ...mappedEducation, ...mappedHobbies];
        
        allMappedItems.forEach(item => {
            (item.skill_ids || []).forEach(id => skillIds.add(id));
            (item.achievement_ids || []).forEach(achId => {
                const ach = cv.achievements.find(a => a.id === achId);
                if (ach) {
                    (ach.skill_ids || []).forEach(id => skillIds.add(id));
                }
            });
        });

        const groupedSkills = { technical: [], soft: [], language: [], other: [] };
        cv.skills.forEach(skill => {
            if(skill) { // Add safety check
                (groupedSkills[skill.category] || groupedSkills.other).push(skill);
            }
        });
        
        return { 
            mappedExperiences, mappedProjects, mappedEducation, mappedHobbies, 
            suggestedExperiences, suggestedProjects, suggestedEducation, suggestedHobbies,
            otherUnmappedExperiences, otherUnmappedProjects, otherUnmappedEducation, otherUnmappedHobbies,
            aggregatedSkillIds: skillIds,
            groupedSkills
        };
    }, [cv, mapping, allSuggestions]); // <-- NOW DEPENDS ON allSuggestions

    const [selectedSkillIds, setSelectedSkillIds] = useState(new Set(aggregatedSkillIds));

    useEffect(() => {
        setSelectedSkillIds(new Set(aggregatedSkillIds));
    }, [aggregatedSkillIds]);

    const handleToggleSkill = (skillId) => {
        setSelectedSkillIds(prev => {
            const newSet = new Set(prev);
            if (newSet.has(skillId)) {
                newSet.delete(skillId);
            } else {
                newSet.add(skillId);
            }
            return newSet;
        });
    };
    
    const handleGeneratePrompt = async () => {
        setIsLoadingPrompt(true);
        try {
            const skillIdArray = Array.from(selectedSkillIds);
            const res = await generateCvPrompt(cv.id, job.id, skillIdArray); 
            
            setCvPromptJson(JSON.stringify(res.data, null, 2));
            setIsModalOpen(true);
        } catch (err) {
            alert("Failed to generate prompt.");
            console.error(err);
        } finally {
            setIsLoadingPrompt(false);
        }
    };

    // --- 7. HANDLERS FOR AI SUGGESTIONS ---
    const handleAcceptSuggestion = async (suggestion) => {
        setIsAccepting(suggestion.id); // Use the temp ID
        try {
            await addMappingPair(
                mapping.id, 
                suggestion.feature_id, 
                suggestion.context_item_id, 
                suggestion.context_item_type, 
                suggestion.annotation,
                suggestion.feature_text,    // Pass the exact text
                suggestion.context_item_text  // Pass the exact text
            );
            // This prop comes from ApplicationWorkspace and refetches the mapping
            // This will trigger the useMemo to re-triage the lists
            await onMappingChanged(); 
        } catch (err) {
            alert(`Failed to accept pair: ${err.response?.data?.detail || err.message}`);
            console.error(err);
        } finally {
            setIsAccepting(null);
        }
    };

    const handleIgnoreSuggestion = (suggestionId) => {
        // Just remove it from the local suggestions list.
        // On next re-render, useMemo will see it's gone and move
        // the item to the "otherUnmapped" list.
        setAllSuggestions(prev => prev.filter(s => s.id !== suggestionId));
    };

    // --- 8. HANDLERS FOR MANUAL PROMOTION (from old code) ---
    const handleOpenPromoteModal = (item, itemType) => {
        setItemToPromote({ item: item, type: itemType });
        setIsPromoteModalOpen(true);
    };

    const handleClosePromoteModal = () => {
        setItemToPromote(null);
        setIsPromoteModalOpen(false);
    };
    
    // This handler now needs to be passed *down* to the modal
    const handleManualPromoteSubmit = async (featureId) => {
        if (!itemToPromote || !featureId) return;
        
        setIsSubmittingManual(true);
        try {
            await addMappingPair(
                mapping.id,
                featureId,
                itemToPromote.item.id,
                itemToPromote.type,
                "Manually promoted in Step 2" // Add a default note
            );
            await onMappingChanged(); // Reload mapping
            handleClosePromoteModal();
        } catch (err) {
            alert(`Failed to promote item: ${err.response?.data?.detail || err.message}`);
            console.error(err);
        } finally {
            setIsSubmittingManual(false);
        }
    };


    // --- 9. RENDER THE FULL CV PREVIEW ---
    return (
        <div>
            <h4 className="h5">Step 2: Review Your Tailored CV</h4>
            <p className="text-muted">
                This is a preview of the content that will be sent to the AI.
                Review AI suggestions or manually promote unmapped items to include them.
            </p>
            
            {isLoadingSuggestions && (
                <div className="d-flex justify-content-center align-items-center my-3">
                    <div className="spinner-border text-info" role="status">
                        <span className="visually-hidden">Loading...</span>
                    </div>
                    <span className="ms-2">Loading AI suggestions...</span>
                </div>
            )}
            {suggestionError && <div className="alert alert-danger">{suggestionError}</div>}
            
            <div 
                className="border p-3 p-md-4 rounded bg-light" 
                style={{ maxHeight: '60vh', overflowY: 'auto', filter: isLoadingSuggestions ? 'blur(2px)' : 'none' }}
            >
                
                {/* Header */}
                <div className="text-center border-bottom pb-3 mb-3">
                    <h3 className="h4 m-0">{getCVDisplayName(cv)}</h3>
                    {cv.summary && (
                        <p className="m-0 mt-1 fst-italic text-muted" style={{whiteSpace: 'pre-wrap'}}>
                            {cv.summary}
                        </p>
                    )}
                </div>

                {/* Mapped Experiences */}
                {(mappedExperiences.length > 0 || suggestedExperiences.length > 0 || otherUnmappedExperiences.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Professional Experience</h5>
                        {mappedExperiences.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="experiences" allSkills={cv.skills} allAchievements={cv.achievements} />
                        ))}
                        
                        {/* --- AI Suggested Experiences --- */}
                        {suggestedExperiences.length > 0 && (
                            <div className="mt-3">
                                <h6 className="text-muted small">💡 AI Suggestions (Not Mapped):</h6>
                                {suggestedExperiences.map(item => (
                                    <AISuggestionCard 
                                        key={item.suggestion.id} 
                                        item={item} 
                                        itemType="experiences" 
                                        suggestion={item.suggestion}
                                        onAccept={handleAcceptSuggestion}
                                        onIgnore={handleIgnoreSuggestion}
                                        isAccepting={isAccepting === item.suggestion.id}
                                    />
                                ))}
                            </div>
                        )}

                        {/* --- Other Unmapped Experiences --- */}
                        {otherUnmappedExperiences.length > 0 && (
                            <div className="mt-3">
                                <h6 className="text-muted small">Available (Not Mapped):</h6>
                                {otherUnmappedExperiences.map(item => (
                                    <GhostItemCard key={item.id} item={item} itemType="experiences" onPromote={handleOpenPromoteModal} />
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Mapped Education */}
                {(mappedEducation.length > 0 || suggestedEducation.length > 0 || otherUnmappedEducation.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Education</h5>
                        {mappedEducation.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="education" allSkills={cv.skills} allAchievements={cv.achievements} />
                        ))}
                        {/* --- AI Suggested Education --- */}
                        {suggestedEducation.map(item => (
                            <AISuggestionCard 
                                key={item.suggestion.id} 
                                item={item} 
                                itemType="education" 
                                suggestion={item.suggestion}
                                onAccept={handleAcceptSuggestion}
                                onIgnore={handleIgnoreSuggestion}
                                isAccepting={isAccepting === item.suggestion.id}
                            />
                        ))}
                        {/* --- Other Unmapped Education --- */}
                        {otherUnmappedEducation.map(item => (
                            <GhostItemCard key={item.id} item={item} itemType="education" onPromote={handleOpenPromoteModal} />
                        ))}
                    </div>
                )}
                
                {/* Mapped Projects */}
                {(mappedProjects.length > 0 || suggestedProjects.length > 0 || otherUnmappedProjects.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Projects</h5>
                        {mappedProjects.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="projects" allSkills={cv.skills} allAchievements={cv.achievements} allExperiences={cv.experiences} allEducation={cv.education} />
                        ))}
                        {/* --- AI Suggested Projects --- */}
                        {suggestedProjects.map(item => (
                            <AISuggestionCard 
                                key={item.suggestion.id} 
                                item={item} 
                                itemType="projects" 
                                suggestion={item.suggestion}
                                onAccept={handleAcceptSuggestion}
                                onIgnore={handleIgnoreSuggestion}
                                isAccepting={isAccepting === item.suggestion.id}
                            />
                        ))}
                        {/* --- Other Unmapped Projects --- */}
                        {otherUnmappedProjects.map(item => (
                            <GhostItemCard key={item.id} item={item} itemType="projects" onPromote={handleOpenPromoteModal} />
                        ))}
                    </div>
                )}

                {/* Mapped Hobbies */}
                {(mappedHobbies.length > 0 || suggestedHobbies.length > 0 || otherUnmappedHobbies.length > 0) && (
                    <div className="mb-4">
                        <h5 className="text-primary border-bottom pb-1">Hobbies & Interests</h5>
                        {mappedHobbies.map(item => (
                            <CVItemDisplayCard key={item.id} item={item} itemType="hobbies" allSkills={cv.skills} allAchievements={cv.achievements} />
                        ))}
                        {/* --- AI Suggested Hobbies --- */}
                        {suggestedHobbies.map(item => (
                            <AISuggestionCard 
                                key={item.suggestion.id} 
                                item={item} 
                                itemType="hobbies" 
                                suggestion={item.suggestion}
                                onAccept={handleAcceptSuggestion}
                                onIgnore={handleIgnoreSuggestion}
                                isAccepting={isAccepting === item.suggestion.id}
                            />
                        ))}
                        {/* --- Other Unmapped Hobbies --- */}
                        {otherUnmappedHobbies.map(item => (
                            <GhostItemCard key={item.id} item={item} itemType="hobbies" onPromote={handleOpenPromoteModal} />
                        ))}
                    </div>
                )}

                {/* --- Interactive Skills Section (Unchanged) --- */}
                <div className="mb-4">
                    <h5 className="text-primary border-bottom pb-1">Skills</h5>
                    <p className="form-text mt-0 mb-2">
                        Skills highlighted in green are automatically selected based on your mapping.
                        Click any skill to include or exclude it from the prompt.
                    </p>
                    
                    {['technical', 'soft', 'language', 'other'].map(category => (
                        groupedSkills[category].length > 0 && (
                            <div key={category} className="mb-3">
                                <strong className="text-capitalize d-block mb-2">{category}</strong>
                                <div className="d-flex flex-wrap gap-2">
                                    {groupedSkills[category].map(skill => {
                                        if (!skill) return null; // Safety check
                                        const isSelected = selectedSkillIds.has(skill.id);
                                        const isAutoSelected = aggregatedSkillIds.has(skill.id);
                                        
                                        return (
                                            <React.Fragment key={skill.id}>
                                                <input
                                                    type="checkbox"
                                                    className="btn-check"
                                                    id={`skill-check-${skill.id}`}
                                                    checked={isSelected}
                                                    onChange={() => handleToggleSkill(skill.id)}
                                                    autoComplete="off"
                                                />
                                                <label 
                                                    className={`btn btn-sm ${isSelected ? (isAutoSelected ? 'btn-success' : 'btn-primary') : 'btn-outline-secondary'}`}
                                                    htmlFor={`skill-check-${skill.id}`}
                                                >
                                                    {skill.name}
                                                </label>
                                            </React.Fragment>
                                        );
                                    })}
                                </div>
                            </div>
                        )
                    ))}
                </div>
            </div>

            <button 
                className="btn btn-info mt-3" 
                onClick={handleGeneratePrompt}
                disabled={isLoadingPrompt || isLoadingSuggestions}
            >
                {isLoadingPrompt ? "Generating..." : "Generate CV Prompt"}
            </button>

            <PromptModal
                isOpen={isModalOpen}
                jsonString={cvPromptJson}
                onClose={() => setIsModalOpen(false)}
            />

            {/* --- 10. RENDER THE PROMOTE MODAL (passing the new handler) --- */}
            {isPromoteModalOpen && (
                <PromoteItemModal
                    isOpen={isPromoteModalOpen}
                    onClose={handleClosePromoteModal}
                    job={job}
                    mapping={mapping}
                    itemToPromote={itemToPromote}
                    onMappingChanged={onMappingChanged}
                    onPromoteSubmit={handleManualPromoteSubmit} // <-- Pass the correct handler
                    isSubmitting={isSubmittingManual}
                />
            )}
            
            <div className="d-flex justify-content-between mt-4">
                <button className="btn btn-outline-secondary" onClick={onPrev}>
                    &lt; Back: Mapping
                </button>
                <button className="btn btn-primary" onClick={onNext}>
                    Next: Build Cover Letter &gt;
                </button>
            </div>
        </div>
    );
};

export default Step2_GenerateCV;