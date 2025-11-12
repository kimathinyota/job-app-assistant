// frontend/src/components/applications/CL_SuggestionModal.jsx
import React, { useState, useMemo } from 'react';
import {
    addCoverLetterIdea,
    updateCoverLetterIdea,
    updateCoverLetterParagraph,
} from '../../api/applicationClient.js';
import { PairChip } from './Step3_BuildCoverLetter.jsx'; // Assuming PairChip is exported

/**
 * A single suggestion card inside the modal.
 */
const SuggestionCard = ({ suggestion, onAccept, isAccepting }) => {
    const [customText, setCustomText] = useState('');
    
    // --- NEW: State for the "Add to Outline" toggle ---
    // Defaults to true for all new suggestions.
    const [addToOutline, setAddToOutline] = useState(true);

    const handleAccept = () => {
        // Pass both the custom text and the toggle state up
        onAccept(suggestion, customText, addToOutline);
    };

    // --- NEW: Determine if this card should show the toggle ---
    // We don't show it for "add_proof" since that just modifies an existing idea.
    const showAddToOutlineToggle = suggestion.type !== 'add_proof';

    return (
        <div className="card card-body mb-3 shadow-sm">
            <h6 className="h6 card-title mb-1">{suggestion.title}</h6>
            <p className="small text-muted">{suggestion.description}</p>

            {/* --- MODIFIED: Render textarea for 'company', 'conclusion_cta', OR 'intro_hook' --- */}
            {(suggestion.type === 'company' || suggestion.type === 'conclusion_cta' || suggestion.type === 'intro_hook') && (
                <div className="my-2">
                    <label htmlFor={`custom-text-${suggestion.id}`} className="form-label small">
                        {suggestion.type === 'company' && "How do you align with their mission? (Optional)"}
                        {suggestion.type === 'conclusion_cta' && "Add your call to action text (Optional)"}
                        {suggestion.type === 'intro_hook' && "Draft your opening hook (Optional)"}
                    </label>
                    <textarea
                        id={`custom-text-${suggestion.id}`}
                        className="form-control form-control-sm"
                        rows="3"
                        placeholder={
                            suggestion.type === 'company'
                                ? "e.g., 'Their focus on renewable energy aligns...'"
                                : suggestion.type === 'intro_hook'
                                    ? "e.g., 'As a lifelong fan of [Company], I was thrilled to see...'"
                                    : "e.g., 'I am available for an interview at your earliest convenience...'"
                        }
                        value={customText}
                        onChange={(e) => setCustomText(e.target.value)}
                    />
                </div>
            )}
            {/* --- END MODIFICATION --- */}


            {/* Render pairs for 'new_arg' or 'add_proof' types */}
            {suggestion.pairs && suggestion.pairs.length > 0 && (
                <div className="mt-2 border-top pt-2">
                    <strong className="small d-block mb-2">Relevant Proof:</strong>
                    <div style={{ 
                        maxHeight: '150px', 
                        overflowY: 'auto', 
                        padding: '8px', 
                        background: '#f8f9fa', 
                        border: '1px solid #dee2e6',
                        borderRadius: '0.25rem'
                    }}>
                        {suggestion.pairs.map(pair => (
                            <PairChip key={pair.id} pair={pair} />
                        ))}
                    </div>
                </div>
            )}

            <div className="d-flex justify-content-between align-items-center mt-2">
                {/* --- NEW: "Add to Outline" Toggle --- */}
                {showAddToOutlineToggle ? (
                    <div className="form-check form-switch">
                        <input
                            className="form-check-input"
                            type="checkbox"
                            role="switch"
                            id={`add-to-outline-${suggestion.id}`}
                            checked={addToOutline}
                            onChange={(e) => setAddToOutline(e.target.checked)}
                        />
                        <label 
                            className="form-check-label small" 
                            htmlFor={`add-to-outline-${suggestion.id}`}
                        >
                            Add to outline
                        </label>
                    </div>
                ) : (
                    <div></div> // Empty div to keep button to the right
                )}
                {/* --- END NEW --- */}

                <button
                    className="btn btn-sm btn-success"
                    onClick={handleAccept}
                    disabled={isAccepting}
                >
                    {isAccepting ? 'Adding...' : suggestion.actionText}
                </button>
            </div>
        </div>
    );
};

/**
 * The main "Cover Letter Assistant" modal.
 */
const CL_SuggestionModal = ({
    isOpen,
    onClose,
    coverLetter,
    mapping,
    job,
    onSuggestionsAccepted // Callback to reload data in parent
}) => {
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [acceptingId, setAcceptingId] = useState(null);
    const [newIdeaTitle, setNewIdeaTitle] = useState('');
    const [isCreatingBlank, setIsCreatingBlank] = useState(false);

    // The core "smart" logic to generate suggestions
    const suggestions = useMemo(() => {
        if (!isOpen) return []; 
        
        const allIdeas = coverLetter.ideas || [];
        const allPairs = mapping.pairs || [];
        const allFeatures = job.features || [];
        const allParagraphs = coverLetter.paragraphs || [];

        const suggestionsList = [];

        // 1. Check for "Opening Hook" in Introduction
        const introPara = allParagraphs.find(p => p.purpose.toLowerCase().includes('introduction'));
        if (introPara) {
            const hasHookIdea = allIdeas.some(i => i.title === "Opening Hook");
            // Suggest if the intro para is empty AND a Hook idea doesn't already exist
            if (introPara.idea_ids.length === 0 && !hasHookIdea) {
                suggestionsList.push({
                    id: 'sugg_intro_hook',
                    type: 'intro_hook',
                    title: "Add an Opening Hook",
                    description: "Your 'Introduction' paragraph is empty. Start strong with a hook that grabs the reader's attention.",
                    actionText: "Add Hook",
                });
            }
        }

        // 2. Check for "Why You" (Company Research)
        const hasCompanyArg = allIdeas.some(idea => idea.title === "Company Mission & Values");
        if (!hasCompanyArg) {
            suggestionsList.push({
                id: 'sugg_company_research',
                type: 'company',
                title: "Create 'Why You' Argument",
                description: "Add your research on the company to create a strong 'Why this company' paragraph.",
                actionText: "Create Argument",
            });
        }

        // 3. Check for "Why Me" (Job Requirements)
        const usedPairIds = new Set(allIdeas.flatMap(idea => idea.mapping_pair_ids || []));
        for (const feature of allFeatures) {
            const argTitle = `Regarding: ${feature.description}`;
            const allPairsForFeature = allPairs.filter(p => p.feature_id === feature.id);
            if (allPairsForFeature.length === 0) continue; 
            const unusedPairsForFeature = allPairsForFeature.filter(p => !usedPairIds.has(p.id));
            const existingArg = allIdeas.find(idea => idea.title === argTitle);

            if (existingArg) {
                if (unusedPairsForFeature.length > 0) {
                    suggestionsList.push({
                        id: `sugg_add_proof_${feature.id}`,
                        type: 'add_proof',
                        title: `Add Missing Proof to "${feature.description.substring(0, 40)}..."`,
                        description: `You have ${unusedPairsForFeature.length} more mapped item(s) for this requirement.`,
                        actionText: "Add Proof",
                        pairs: unusedPairsForFeature,
                        targetIdeaId: existingArg.id,
                    });
                }
            } else {
                if (unusedPairsForFeature.length > 0) {
                    suggestionsList.push({
                        id: `sugg_new_arg_${feature.id}`,
                        type: 'new_arg',
                        title: `Create Argument for "${feature.description.substring(0, 40)}..."`,
                        description: `Create a new argument using ${unusedPairsForFeature.length} mapped item(s).`,
                        actionText: "Create Argument",
                        pairs: unusedPairsForFeature,
                        argTitle: argTitle,
                    });
                }
            }
        }
        
        // 4. Check for "Call to Action" in Conclusion
        const conclusionPara = allParagraphs.find(p => p.purpose.toLowerCase().includes('conclusion'));
        if (conclusionPara) {
            const hasCtaIdea = allIdeas.some(i => i.title === "Call to Action");
            // Only suggest if the conclusion para is empty AND a CTA idea doesn't already exist
            if (conclusionPara.idea_ids.length === 0 && !hasCtaIdea) {
                suggestionsList.push({
                    id: 'sugg_conclusion_cta',
                    type: 'conclusion_cta',
                    title: "Add a Call to Action",
                    description: "Your 'Conclusion' paragraph is empty. A good conclusion reinforces your interest and invites a response.",
                    actionText: "Add CTA Argument",
                });
            }
        }

        return suggestionsList;
    }, [isOpen, coverLetter, mapping, job]);

    // --- MODIFIED: Handler now accepts `customText` and `addToOutline` ---
    const handleAcceptSuggestion = async (suggestion, customText, addToOutline) => {
        setAcceptingId(suggestion.id);
        setIsSubmitting(true);
        const clId = coverLetter.id;
        
        try {
            let newIdea = null; // Store the newly created idea
            let targetParagraph = null; // Store the paragraph to add it to

            if (suggestion.type === 'intro_hook') {
                const defaultAnnotation = "I am writing to express my enthusiastic interest in the [Job Title] position at [Company].";
                const annotation = customText.trim() || defaultAnnotation;
                const res = await addCoverLetterIdea(clId, "Opening Hook", [], annotation);
                newIdea = res.data;
                targetParagraph = coverLetter.paragraphs.find(p => p.purpose.toLowerCase().includes('introduction'));

            } else if (suggestion.type === 'company') {
                const annotation = customText.trim() || "Researched company mission and values.";
                const res = await addCoverLetterIdea(clId, "Company Mission & Values", [], annotation);
                newIdea = res.data;
                targetParagraph = coverLetter.paragraphs.find(p => p.purpose.includes("Why You"));

            } else if (suggestion.type === 'new_arg') {
                const pairIds = suggestion.pairs.map(p => p.id);
                const res = await addCoverLetterIdea(clId, suggestion.argTitle, pairIds, null);
                newIdea = res.data;
                targetParagraph = coverLetter.paragraphs.find(p => p.purpose.includes("Why Me"));

            } else if (suggestion.type === 'add_proof') {
                // This type doesn't create a new idea, so it ignores `addToOutline`
                const targetIdea = coverLetter.ideas.find(i => i.id === suggestion.targetIdeaId);
                if (targetIdea) {
                    const newPairIds = [...targetIdea.mapping_pair_ids, ...suggestion.pairs.map(p => p.id)];
                    await updateCoverLetterIdea(clId, targetIdea.id, { mapping_pair_ids: newPairIds });
                }
            
            } else if (suggestion.type === 'conclusion_cta') {
                const defaultAnnotation = "I am available for an interview at your earliest convenience and look forward to hearing from you.";
                const annotation = customText.trim() || defaultAnnotation;
                const res = await addCoverLetterIdea(clId, "Call to Action", [], annotation);
                newIdea = res.data;
                targetParagraph = coverLetter.paragraphs.find(p => p.purpose.toLowerCase().includes('conclusion'));
            }
            
            // --- NEW: Auto-add to outline logic ---
            if (newIdea && addToOutline && targetParagraph) {
                await updateCoverLetterParagraph(clId, targetParagraph.id, {
                    idea_ids: [...targetParagraph.idea_ids, newIdea.id]
                });
            }
            // --- END NEW ---
            
            await onSuggestionsAccepted();

        } catch (err) {
            console.error("Failed to accept suggestion:", err);
            alert(`Error: ${err.message || 'Failed to perform action.'}`);
        } finally {
            setIsSubmitting(false);
            setAcceptingId(null);
        }
    };
    
    // Handler for creating a blank idea (unchanged, does not auto-add to outline)
    const handleCreateBlankIdea = async (e) => {
        e.preventDefault();
        if (!newIdeaTitle.trim()) return;

        setIsCreatingBlank(true);
        setIsSubmitting(true);
        try {
            await addCoverLetterIdea(coverLetter.id, newIdeaTitle, [], null);
            setNewIdeaTitle(''); // Clear input
            await onSuggestionsAccepted(); // Reload
        } catch (err) {
            console.error("Failed to create blank idea:", err);
            alert("Failed to create blank idea.");
        } finally {
            setIsCreatingBlank(false);
            setIsSubmitting(false);
        }
    };

    if (!isOpen) return null;

    return (
        <>
            <div className="modal-backdrop fade show"></div>
            <div className="modal fade show" style={{ display: 'block' }} tabIndex="-1">
                <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                    <div className="modal-content">
                        <div className="modal-header">
                            <h5 className="modal-title">
                                <i className="bi bi-magic me-2"></i>
                                Cover Letter Assistant
                            </h5>
                            <button
                                type="button"
                                className="btn-close"
                                onClick={onClose}
                                disabled={isSubmitting}
                            ></button>
                        </div>
                        
                        <div 
                            className="modal-body bg-light" 
                            style={{ 
                                maxHeight: '65vh', 
                                overflowY: 'auto' 
                            }}
                        >
                          
                            
                            {/* <hr /> */}
                        
                            {isSubmitting && !isCreatingBlank && (
                                <div className="text-center">
                                    <div className="spinner-border text-primary" role="status">
                                        <span className="visually-hidden">Loading...</span>
                                    </div>
                                    <p>Updating cover letter...</p>
                                </div>
                            )}

                            {!isSubmitting && suggestions.length === 0 && (
                                <div className="text-center p-4">
                                    <h5 className="text-success">All Set!</h5>
                                    <p className="text-muted">
                                        The assistant has no new automated suggestions.
                                    </p>
                                </div>
                            )}

                            {!isSubmitting && suggestions.map(sugg => (
                                <SuggestionCard
                                    key={sugg.id}
                                    suggestion={sugg}
                                    onAccept={handleAcceptSuggestion}
                                    isAccepting={acceptingId === sugg.id}
                                />
                            ))}
                        </div>
                        <div className="modal-footer">
                            <button
                                type="button"
                                className="btn btn-secondary"
                                onClick={onClose}
                                disabled={isSubmitting}
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default CL_SuggestionModal;