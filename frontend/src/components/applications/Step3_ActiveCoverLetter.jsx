// frontend/src/components/applications/Step3_ActiveCoverLetter.jsx
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
    generateCoverLetterPrompt,
    createCoverLetter,
    fetchCoverLetterDetails,
    addCoverLetterIdea,
    updateCoverLetterIdea,
    deleteCoverLetterIdea,
    addCoverLetterParagraph,
    updateCoverLetterParagraph,
    deleteCoverLetterParagraph,
    autofillCoverLetter
} from '../../api/applicationClient.js';

import PromptModal from './PromptModal.jsx';
import CVItemPreviewModal from './CVItemPreviewModal.jsx';
// We no longer import IntelligentTextAreaModal
import ParagraphStudio from './ParagraphStudio.jsx'; // The new "Chapter" component

import {
    Wand2,
    FileText,
    ArrowRight,
    ArrowLeft,
    Loader2,
    BrainCircuit,
    Sparkles,
    BookUser
} from 'lucide-react';

const Step3_ActiveCoverLetter = ({
    application,
    mapping,
    onPrev,
    onNext,
    job,
    onCoverLetterCreated,
    fullCV
}) => {
    const [coverLetter, setCoverLetter] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [error, setError] = useState(null);
    const [strategy, setStrategy] = useState('standard');

    // Modal States
    const [clPromptJson, setClPromptJson] = useState('');
    const [isPromptModalOpen, setIsPromptModalOpen] = useState(false);
    const [previewItem, setPreviewItem] = useState(null);

    // Create maps for efficient data lookup
    const { ideaMap, pairMap } = useMemo(() => {
        const iMap = new Map(coverLetter?.ideas.map(i => [i.id, i]));
        const pMap = new Map(mapping?.pairs.map(p => [p.id, p]));
        return { ideaMap: iMap, pairMap: pMap };
    }, [coverLetter?.ideas, mapping?.pairs]);

    // --- API HANDLERS ---
    
    const handleAutofill = useCallback(async (coverId, newStrategy, mode = 'reset') => {
        setIsSubmitting(true);
        setError(null);
        try {
            const res = await autofillCoverLetter(coverId, newStrategy, mode);
            setCoverLetter(res.data);
            setStrategy(newStrategy);
            return res.data;
        } catch (err) {
            setError("Failed to generate outline. " + (err.response?.data?.detail || err.message));
            console.error(err);
        } finally {
            setIsSubmitting(false);
        }
    }, []);

    const loadCoverLetter = useCallback(async () => {
        setIsLoading(true);
        try {
            let clData;
            if (application.cover_letter_id) {
                clData = (await fetchCoverLetterDetails(application.cover_letter_id)).data;
            } else {
                if (!application.job_id || !application.base_cv_id || !application.mapping_id) {
                    setError("Missing application data to create cover letter.");
                    setIsLoading(false); // Changed from True to False
                    return;
                }
                const res = await createCoverLetter(application.job_id, application.base_cv_id, application.mapping_id);
                clData = res.data;
                if (onCoverLetterCreated) {
                    onCoverLetterCreated(clData.id);
                }
            }
            
            if (clData.paragraphs.length === 0) {
                const filledData = await handleAutofill(clData.id, 'standard');
                setCoverLetter(filledData || clData); // Set filled data or at least the created CL
            } else {
                setCoverLetter(clData);
            }
        } catch (err) {
            setError("Failed to load cover letter data.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    }, [application.cover_letter_id, application.job_id, application.base_cv_id, application.mapping_id, onCoverLetterCreated, handleAutofill]);

    useEffect(() => {
        loadCoverLetter();
    }, [loadCoverLetter]);

    const handleStrategyChange = (e) => {
        const newStrategy = e.target.value;
        if (!coverLetter?.id) return;
        handleAutofill(coverLetter.id, newStrategy, 'reset');
    };

    // --- CRUD Handlers (passed down to children) ---
    // These handlers now contain the "Promotion" logic
    
    const handleUpdateIdea = async (ideaId, updates) => {
        if (isSubmitting) return;
        setIsSubmitting(true);
        
        const optimisticIdea = { ...ideaMap.get(ideaId), ...updates, owner: 'user' };
        setCoverLetter(prev => ({
            ...prev,
            ideas: prev.ideas.map(i => i.id === ideaId ? optimisticIdea : i)
        }));

        try {
            await updateCoverLetterIdea(coverLetter.id, ideaId, updates);
        } catch (err) {
            setError("Failed to save argument.");
            await loadCoverLetter(); // Rollback
        } finally {
            setIsSubmitting(false);
        }
    };
    
    const handleUpdateParagraph = async (paraId, updates) => {
        // This is for local saves (draft text), so we don't set global isSubmitting
        try {
            // Note: We DO NOT send owner: "user" here.
            await updateCoverLetterParagraph(coverLetter.id, paraId, updates);
            setCoverLetter(prev => ({
                ...prev,
                paragraphs: prev.paragraphs.map(p => p.id === paraId ? { ...p, ...updates } : p)
            }));
        } catch (err) {
            setError("Failed to save draft.");
            await loadCoverLetter(); // Rollback
        }
    };

    const handleAddArgument = async (paraId, classification) => {
        if (isSubmitting) return;
        setIsSubmitting(true);
        try {
            const newIdeaRes = await addCoverLetterIdea(
                coverLetter.id,
                "New Custom Argument",
                [],
                "",
                classification
            );
            const newIdea = newIdeaRes.data;
            const para = coverLetter.paragraphs.find(p => p.id === paraId);
            const newIdeaIds = [...para.idea_ids, newIdea.id];
            await updateCoverLetterParagraph(coverLetter.id, paraId, { idea_ids: newIdeaIds });
            await loadCoverLetter();
        } catch (err) {
            setError("Failed to add argument.");
            await loadCoverLetter();
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleDeleteIdea = async (idea, para) => {
        if (!window.confirm(`Are you sure you want to delete this argument: "${idea.title}"?`)) return;
        if (isSubmitting) return;
        setIsSubmitting(true);
        try {
            const newIdeaIds = para.idea_ids.filter(id => id !== idea.id);
            await updateCoverLetterParagraph(coverLetter.id, para.id, { idea_ids: newIdeaIds });
            await deleteCoverLetterIdea(coverLetter.id, idea.id);
            await loadCoverLetter();
        } catch (err) {
            setError("Failed to delete argument.");
            await loadCoverLetter();
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleAddCustomParagraph = async () => {
        const purpose = prompt("What is the purpose of this new section?", "My Custom Section");
        if (!purpose || purpose.trim() === "") return;
        
        if (isSubmitting) return;
        setIsSubmitting(true);
        try {
            await addCoverLetterParagraph(coverLetter.id, purpose, []);
            await loadCoverLetter();
        } catch (err) {
            setError("Failed to add custom paragraph.");
        } finally {
            setIsSubmitting(false);
        }
    };

    const handleCopyAsText = () => {
        const text = coverLetter.paragraphs
            .sort((a, b) => a.order - b.order)
            .map(p => `--- ${p.purpose} ---\n${p.draft_text || "[No text written for this section]"}`)
            .join('\n\n');
        
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed'; ta.style.top = '-9999px'; ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); } catch (err) { console.error('Failed to copy text', err); }
        document.body.removeChild(ta);
    };

    const handleGeneratePrompt = async () => {
        if (isSubmitting) return;
        setIsSubmitting(true);
        try {
            const res = await generateCoverLetterPrompt(mapping.id);
            setClPromptJson(JSON.stringify(res.data, null, 2));
            setIsPromptModalOpen(true);
        } catch (err) {
            setError("Failed to generate prompt.");
        } finally {
            setIsSubmitting(false);
        }
    };

    // --- RENDER ---
    if (isLoading) {
        return <div className="p-5 text-center"><Loader2 className="animate-spin h-8 w-8 text-primary" /></div>;
    }

    if (error) {
        return <div className="alert alert-danger" role="alert">{error}</div>;
    }

    if (!coverLetter) {
        return <div className="alert alert-info">No cover letter data.</div>;
    }
    
    const sortedParagraphs = [...coverLetter.paragraphs].sort((a, b) => a.order - b.order);

    return (
        <div className="container-fluid" style={{ maxWidth: '1000px', margin: '0 auto' }}>
            
            <div className="d-flex justify-content-between align-items-center mb-4 sticky-top bg-light py-3 border-bottom" style={{top: 0, zIndex: 10}}>
                <div>
                    <h4 className="fw-bold text-dark mb-1 d-flex align-items-center gap-2">
                        <BrainCircuit size={24} className="text-primary"/> Cover Letter Studio
                    </h4>
                    <p className="text-muted small mb-0">Build your strategy, then write your prose.</p>
                </div>
                <div className="d-flex align-items-center gap-2">
                    <button 
                        className="btn btn-outline-secondary d-flex align-items-center gap-2"
                        onClick={handleCopyAsText}
                        title="Copy written text to clipboard"
                    >
                        <FileText size={16}/> Copy Text
                    </button>
                    <button 
                        className="btn btn-primary d-flex align-items-center gap-2"
                        onClick={handleGeneratePrompt}
                        disabled={isSubmitting}
                        title="Generate AI prompt from strategy"
                    >
                        {isSubmitting ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>} Generate Prompt
                    </button>
                </div>
            </div>

            <div className="mb-4 p-3 bg-white border rounded-3 shadow-sm">
                <div className="d-flex align-items-center gap-3">
                    <div className="flex-shrink-0">
                        <Sparkles size={20} className="text-primary"/>
                    </div>
                    <div className="flex-grow-1">
                        <label htmlFor="strategy-select" className="form-label fw-bold mb-1">Outline Strategy</label>
                        <select 
                            id="strategy-select" 
                            className="form-select" 
                            value={strategy} 
                            onChange={handleStrategyChange}
                            disabled={isSubmitting}
                        >
                            <option value="standard">Standard (Pro {"->"} Personal {"->"} Company)</option>
                            <option value="mission_driven">Mission-Driven (Company {"->"} Pro {"->"} Personal)</option>
                            <option value="specialist">Specialist (Pro {"->"} Personal)</option>
                        </select>
                    </div>
                    {isSubmitting && <Loader2 className="animate-spin text-muted" />}
                </div>
            </div>

            <div className="d-flex flex-column gap-4">
                {sortedParagraphs.map(para => (
                    <ParagraphStudio
                        key={para.id}
                        paragraph={para}
                        jobFeatures={job?.features || []}
                        fullCV={fullCV}
                        ideaMap={ideaMap}
                        pairMap={pairMap}
                        isSubmitting={isSubmitting}
                        // API Handlers
                        onAddArgument={handleAddArgument}
                        onDeleteIdea={handleDeleteIdea}
                        onUpdateIdea={handleUpdateIdea}
                        onUpdateParagraph={handleUpdateParagraph}
                        // UI Handlers
                        onShowPreview={setPreviewItem}
                    />
                ))}
                
                <div className="text-center mt-3">
                    <button
                        className="btn btn-outline-secondary d-flex align-items-center gap-2 mx-auto"
                        onClick={handleAddCustomParagraph}
                        disabled={isSubmitting}
                    >
                        <BookUser size={16} /> Add Custom Section
                    </button>
                </div>
            </div>

            <div className="d-flex justify-content-between border-top pt-3 mt-5">
                <button className="btn btn-outline-secondary d-flex align-items-center gap-2 px-4" onClick={onPrev}><ArrowLeft size={16}/> Back</button>
                <button className="btn btn-outline-primary d-flex align-items-center gap-2 px-4" onClick={onNext}>Next <ArrowRight size={16}/></button>
            </div>

            <PromptModal isOpen={isPromptModalOpen} jsonString={clPromptJson} onClose={() => setIsPromptModalOpen(false)} />
            
            {previewItem && (
                <CVItemPreviewModal 
                    isOpen={!!previewItem}
                    onClose={() => setPreviewItem(null)}
                    itemToPreview={previewItem}
                    allSkills={fullCV?.skills} 
                    allAchievements={fullCV?.achievements} 
                    allExperiences={fullCV?.experiences} 
                    allEducation={fullCV?.education} 
                    allHobbies={fullCV?.hobbies}
                />
            )}
            
            {/* The IntelligentTextAreaModal is GONE. All editing is inline. */}
        </div>
    );
};

export default Step3_ActiveCoverLetter;