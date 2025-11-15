// frontend/src/components/applications/Step2_GenerateCV.jsx
import React, { useState, useMemo, useEffect, useRef } from 'react';
import { 
    FileText, Wand2, Plus, Check, X, ArrowRight, ArrowLeft, 
    Loader2, Sparkles, Briefcase, GraduationCap, FolderGit2, Heart, Cpu,
    FileCheck, Link2, ChevronDown, ChevronUp, Search
} from 'lucide-react';
import { 
    generateCvPrompt,
    inferMappingPairs,  
    addMappingPair      
} from '../../api/applicationClient';
import PromptModal from './PromptModal'; 
import CVItemDisplayCard from './CVItemDisplayCard'; 
import { getCVDisplayName } from '../../utils/cvHelpers'; 

// --- 1. MODERN "AI GHOST" COMPONENT ---
const AISuggestionCard = ({ item, itemType, suggestion, onAccept, onIgnore, isAccepting }) => {
    let itemTitle = item.title || item.name || item.degree || 'Unknown Item';
    if (itemType === 'experiences') itemTitle = `${item.title} @ ${item.company}`;
    if (itemType === 'education') itemTitle = `${item.degree} @ ${item.institution}`;

    return (
        <div className="card border-0 shadow-sm mb-3 bg-primary bg-opacity-10 border-start border-4 border-primary overflow-hidden">
            <div className="card-body p-3">
                <div className="d-flex justify-content-between align-items-start mb-2">
                    <div className="d-flex align-items-center gap-2 text-primary fw-bold small text-uppercase">
                        <Sparkles size={14} /> AI Recommendation
                    </div>
                    <div className="d-flex gap-1">
                        <button 
                            type="button"
                            className="btn btn-sm btn-white border text-muted hover-bg-light p-1" 
                            onClick={() => onIgnore(suggestion.id)}
                            disabled={isAccepting}
                            title="Ignore"
                        >
                            <X size={14}/>
                        </button>
                        <button 
                            type="button"
                            className="btn btn-sm btn-primary shadow-sm p-1 px-2 d-flex align-items-center gap-1" 
                            onClick={() => onAccept(suggestion)}
                            disabled={isAccepting}
                        >
                            {isAccepting ? <Loader2 size={14} className="animate-spin"/> : <Plus size={14}/>}
                            Add
                        </button>
                    </div>
                </div>
                
                <h6 className="fw-bold text-dark mb-1">{itemTitle}</h6>
                <p className="small text-muted mb-2">
                    Matches: <span className="fw-medium text-dark">{suggestion.feature_text}</span>
                </p>
                <div className="bg-white bg-opacity-50 p-2 rounded border border-primary border-opacity-10">
                    <p className="small fst-italic mb-0 text-primary text-opacity-75">
                        "{suggestion.annotation || "Good match."}"
                    </p>
                </div>
            </div>
        </div>
    );
};

// --- 2. INLINE GHOST ITEM CARD (With Custom Searchable Dropdown) ---
const GhostItemCard = ({ item, itemType, jobFeatures, onPromote }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    
    // Form State
    const [selectedReqId, setSelectedReqId] = useState("");
    const [annotation, setAnnotation] = useState("");
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Dropdown State
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState("");

    let itemTitle = item.title || item.name || item.degree || 'Unknown Item';
    if (itemType === 'experiences') itemTitle = `${item.title} @ ${item.company}`;
    if (itemType === 'education') itemTitle = `${item.degree} @ ${item.institution}`;

    // Helper to get selected Requirement Text
    const selectedReqText = useMemo(() => {
        if (!selectedReqId) return "Select a requirement...";
        const req = jobFeatures.find(f => f.id === selectedReqId);
        return req ? req.description : "Unknown Requirement";
    }, [selectedReqId, jobFeatures]);

    // Filter requirements for dropdown
    const filteredFeatures = useMemo(() => {
        if (!searchTerm) return jobFeatures || [];
        return (jobFeatures || []).filter(f => 
            f.description.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }, [jobFeatures, searchTerm]);

    const handleSubmit = async () => {
        if (!selectedReqId) return;
        setIsSubmitting(true);
        try {
            await onPromote(selectedReqId, item.id, itemType, annotation);
            setIsExpanded(false);
            setAnnotation("");
            setSelectedReqId("");
            setSearchTerm("");
        } finally {
            setIsSubmitting(false);
        }
    };

    return (
        <div className={`card border-2 transition-all mb-2 ${isExpanded ? 'border-primary shadow-sm bg-white' : 'border-dashed border-light bg-light bg-opacity-25 hover-shadow-sm'}`}>
            <div className="card-body p-2">
                {/* Card Header / Toggle */}
                <div className="d-flex justify-content-between align-items-center cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
                    <span className={`small fw-medium ps-2 ${isExpanded ? 'text-primary fw-bold' : 'text-muted'}`}>{itemTitle}</span>
                    <button 
                        type="button"
                        className={`btn btn-sm border shadow-sm fw-medium d-flex align-items-center gap-1 transition-all ${isExpanded ? 'btn-light text-muted' : 'btn-white text-primary'}`}
                        onClick={(e) => { e.stopPropagation(); setIsExpanded(!isExpanded); }}
                    >
                        {isExpanded ? <X size={14}/> : <Plus size={12}/>} 
                        {isExpanded ? "Cancel" : "Map"}
                    </button>
                </div>

                {/* Expanded Form */}
                {isExpanded && (
                    <div className="mt-3 ps-2 pe-2 pb-1 animate-fade-in">
                        
                        {/* Custom Searchable Dropdown */}
                        <div className="mb-3 position-relative">
                            <label className="form-label small text-muted fw-bold text-uppercase mb-1">Link to Requirement</label>
                            
                            {/* Trigger Button */}
                            <button 
                                type="button"
                                className="form-control form-control-sm d-flex justify-content-between align-items-center bg-light border-0 shadow-sm text-start"
                                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                                style={{ cursor: 'pointer', minHeight: '38px' }}
                            >
                                <span className={`text-truncate ${!selectedReqId ? 'text-muted' : 'text-dark'}`} style={{maxWidth: '90%'}}>
                                    {selectedReqText}
                                </span>
                                <ChevronDown size={14} className="text-muted"/>
                            </button>

                            {/* Dropdown Menu */}
                            {isDropdownOpen && (
                                <>
                                    <div className="fixed-top w-100 h-100" style={{zIndex: 1040}} onClick={() => setIsDropdownOpen(false)}/>
                                    <div className="position-absolute w-100 bg-white shadow-lg rounded-3 border mt-1 overflow-hidden animate-fade-in" style={{zIndex: 1050, top: '100%'}}>
                                        <div className="p-2 border-bottom bg-light">
                                            <div className="input-group input-group-sm">
                                                <span className="input-group-text bg-white border-0"><Search size={12} className="text-muted"/></span>
                                                <input 
                                                    type="text" 
                                                    className="form-control border-0 shadow-none bg-white" 
                                                    placeholder="Search requirements..."
                                                    value={searchTerm}
                                                    onChange={(e) => setSearchTerm(e.target.value)}
                                                    autoFocus
                                                />
                                            </div>
                                        </div>
                                        <div className="list-group list-group-flush custom-scroll" style={{maxHeight: '200px', overflowY: 'auto'}}>
                                            {filteredFeatures.map(f => (
                                                <button
                                                    key={f.id}
                                                    type="button"
                                                    className={`list-group-item list-group-item-action border-0 small text-start py-2 ${selectedReqId === f.id ? 'bg-primary bg-opacity-10 text-primary fw-medium' : ''}`}
                                                    onClick={() => {
                                                        setSelectedReqId(f.id);
                                                        setIsDropdownOpen(false);
                                                    }}
                                                >
                                                    {f.description}
                                                </button>
                                            ))}
                                            {filteredFeatures.length === 0 && (
                                                <div className="p-3 text-center text-muted small fst-italic">No matching requirements found.</div>
                                            )}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>

                        {/* Annotation */}
                        <div className="mb-3">
                            <label className="form-label small text-muted fw-bold text-uppercase mb-1">Reasoning (Optional)</label>
                            <textarea 
                                className="form-control form-control-sm bg-light border-0 shadow-sm"
                                rows="2"
                                placeholder="Why is this relevant?"
                                value={annotation}
                                onChange={(e) => setAnnotation(e.target.value)}
                            />
                        </div>

                        {/* Action Buttons */}
                        <div className="d-flex justify-content-end">
                            <button 
                                type="button"
                                className="btn btn-sm btn-primary d-flex align-items-center gap-2 shadow-sm"
                                disabled={!selectedReqId || isSubmitting}
                                onClick={handleSubmit}
                            >
                                {isSubmitting ? <Loader2 size={14} className="animate-spin"/> : <Link2 size={14}/>}
                                Confirm & Add
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

// --- MAIN COMPONENT ---
const Step2_GenerateCV = ({ job, cv, mapping, onPrev, onNext, onMappingChanged }) => {
    const [cvPromptJson, setCvPromptJson] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isLoadingPrompt, setIsLoadingPrompt] = useState(false);

    // AI State
    const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(true);
    const [allSuggestions, setAllSuggestions] = useState([]);
    const [suggestionError, setSuggestionError] = useState(null);
    const [isAccepting, setIsAccepting] = useState(null);

    // --- FETCH SUGGESTIONS ---
    useEffect(() => {
        setIsLoadingSuggestions(true);
        setSuggestionError(null);
        inferMappingPairs(mapping.id, "eager_mode")
            .then(res => {
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

    // --- TRIAGE LOGIC ---
    const {
        mappedExperiences, mappedProjects, mappedEducation, mappedHobbies,
        suggestedExperiences, suggestedProjects, suggestedEducation, suggestedHobbies,
        otherUnmappedExperiences, otherUnmappedProjects, otherUnmappedEducation, otherUnmappedHobbies,
        aggregatedSkillIds, groupedSkills
    } = useMemo(() => {
        const mappedItemIds = new Set(mapping.pairs.map(p => p.context_item_id));
        const suggestionMap = new Map(allSuggestions.map(s => [s.context_item_id, s]));

        const triageList = (cvList) => {
            const mapped = [], suggested = [], otherUnmapped = [];
            for (const item of cvList) {
                if (mappedItemIds.has(item.id)) {
                    mapped.push(item);
                } else if (suggestionMap.has(item.id)) {
                    suggested.push({ ...item, suggestion: suggestionMap.get(item.id) });
                } else {
                    otherUnmapped.push(item);
                }
            }
            return { mapped, suggested, otherUnmapped };
        };

        const { mapped: mappedExperiences, suggested: suggestedExperiences, otherUnmapped: otherUnmappedExperiences } = triageList(cv.experiences);
        const { mapped: mappedProjects, suggested: suggestedProjects, otherUnmapped: otherUnmappedProjects } = triageList(cv.projects);
        const { mapped: mappedEducation, suggested: suggestedEducation, otherUnmapped: otherUnmappedEducation } = triageList(cv.education);
        const { mapped: mappedHobbies, suggested: suggestedHobbies, otherUnmapped: otherUnmappedHobbies } = triageList(cv.hobbies);

        // Skill logic
        let skillIds = new Set();
        const allMappedItems = [...mappedExperiences, ...mappedProjects, ...mappedEducation, ...mappedHobbies];
        
        allMappedItems.forEach(item => {
            (item.skill_ids || []).forEach(id => skillIds.add(id));
            (item.achievement_ids || []).forEach(achId => {
                const ach = cv.achievements.find(a => a.id === achId);
                if (ach) (ach.skill_ids || []).forEach(id => skillIds.add(id));
            });
        });

        const groupedSkills = { technical: [], soft: [], language: [], other: [] };
        cv.skills.forEach(skill => {
            if(skill) (groupedSkills[skill.category] || groupedSkills.other).push(skill);
        });
        
        return { 
            mappedExperiences, mappedProjects, mappedEducation, mappedHobbies, 
            suggestedExperiences, suggestedProjects, suggestedEducation, suggestedHobbies,
            otherUnmappedExperiences, otherUnmappedProjects, otherUnmappedEducation, otherUnmappedHobbies,
            aggregatedSkillIds: skillIds, groupedSkills
        };
    }, [cv, mapping, allSuggestions]);

    const [selectedSkillIds, setSelectedSkillIds] = useState(new Set(aggregatedSkillIds));

    useEffect(() => {
        setSelectedSkillIds(new Set(aggregatedSkillIds));
    }, [aggregatedSkillIds]);

    const handleToggleSkill = (skillId) => {
        setSelectedSkillIds(prev => {
            const newSet = new Set(prev);
            newSet.has(skillId) ? newSet.delete(skillId) : newSet.add(skillId);
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
        } finally {
            setIsLoadingPrompt(false);
        }
    };

    // --- HANDLERS ---
    const handleAcceptSuggestion = async (suggestion) => {
        setIsAccepting(suggestion.id); 
        try {
            await addMappingPair(
                mapping.id, suggestion.feature_id, suggestion.context_item_id, 
                suggestion.context_item_type, suggestion.annotation, 
                suggestion.feature_text, suggestion.context_item_text
            );
            await onMappingChanged(); 
        } catch (err) {
            alert(`Failed to accept: ${err.message}`);
        } finally {
            setIsAccepting(null);
        }
    };

    const handleIgnoreSuggestion = (suggestionId) => {
        setAllSuggestions(prev => prev.filter(s => s.id !== suggestionId));
    };

    // New Handler for Inline Promotion (Replaces Modal Logic)
    const handleInlinePromote = async (reqId, itemId, type, note) => {
        try {
            await addMappingPair(
                mapping.id, reqId, itemId, type, note || "Manually promoted in Step 2"
            );
            await onMappingChanged();
        } catch (err) {
            alert(`Failed to promote: ${err.message}`);
            throw err; // Rethrow to stop loading state in child
        }
    };

    return (
        // --- THIS IS THE MODIFIED LINE ---
        <div className="animate-fade-in" style={{ maxWidth: '960px', margin: '0 auto' }}>
            <style>{`
                .border-dashed { border-style: dashed !important; }
                .hover-shadow-sm:hover { box-shadow: 0 .125rem .25rem rgba(0,0,0,.075)!important; }
                .cursor-pointer { cursor: pointer; }
                .custom-scroll::-webkit-scrollbar { width: 6px; }
                .custom-scroll::-webkit-scrollbar-track { background: transparent; }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: #cbd5e1; border-radius: 10px; }
            `}</style>

            {/* Header */}
            <div className="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <h5 className="fw-bold text-dark mb-1 d-flex align-items-center gap-2">
                        <FileCheck size={24} className="text-primary"/> Review Tailored CV
                    </h5>
                    <p className="text-muted small mb-0">
                        Review the content selected for this application. Accept suggestions to enrich your CV.
                    </p>
                </div>
                <button 
                    type="button"
                    className="btn btn-primary shadow-sm d-flex align-items-center gap-2" 
                    onClick={handleGeneratePrompt}
                    disabled={isLoadingPrompt || isLoadingSuggestions}
                >
                    {isLoadingPrompt ? <Loader2 size={16} className="animate-spin"/> : <Wand2 size={16}/>}
                    Generate AI Prompt
                </button>
            </div>
            
            {isLoadingSuggestions && (
                <div className="alert alert-light border d-flex align-items-center justify-content-center gap-2 text-muted mb-4">
                    <Loader2 size={16} className="animate-spin"/> Checking for extra suggestions...
                </div>
            )}
            {suggestionError && <div className="alert alert-danger small mb-4">{suggestionError}</div>}
            
            {/* Content Preview Area */}
            <div className="border rounded-4 shadow-sm bg-white overflow-hidden">
                <div className="bg-light border-bottom p-3 text-center">
                    <h3 className="h5 fw-bold text-dark mb-1">{getCVDisplayName(cv)}</h3>
                    {cv.summary && <p className="small text-muted fst-italic mb-0">{cv.summary}</p>}
                </div>

                <div className="p-4" style={{ maxHeight: '65vh', overflowY: 'auto' }}>
                    
                    {/* --- Section Renderer Helper --- */}
                    {[
                        { title: 'Professional Experience', icon: Briefcase, mapped: mappedExperiences, suggested: suggestedExperiences, ghost: otherUnmappedExperiences, type: 'experiences' },
                        { title: 'Education', icon: GraduationCap, mapped: mappedEducation, suggested: suggestedEducation, ghost: otherUnmappedEducation, type: 'education' },
                        { title: 'Projects', icon: FolderGit2, mapped: mappedProjects, suggested: suggestedProjects, ghost: otherUnmappedProjects, type: 'projects' },
                        { title: 'Hobbies & Interests', icon: Heart, mapped: mappedHobbies, suggested: suggestedHobbies, ghost: otherUnmappedHobbies, type: 'hobbies' }
                    ].map(section => (
                        (section.mapped.length > 0 || section.suggested.length > 0 || section.ghost.length > 0) && (
                            <div key={section.type} className="mb-5">
                                <h6 className="fw-bold text-uppercase text-muted small mb-3 d-flex align-items-center gap-2 border-bottom pb-2">
                                    <section.icon size={14}/> {section.title}
                                </h6>
                                
                                {/* Mapped Items */}
                                <div className="d-flex flex-column gap-3 mb-3">
                                    {section.mapped.map(item => (
                                        <CVItemDisplayCard 
                                            key={item.id} 
                                            item={item} 
                                            itemType={section.type} 
                                            allSkills={cv.skills} 
                                            allAchievements={cv.achievements} 
                                            allExperiences={cv.experiences} 
                                            allEducation={cv.education} 
                                            allHobbies={cv.hobbies} 
                                        />
                                    ))}
                                </div>

                                {/* AI Suggestions */}
                                {section.suggested.length > 0 && (
                                    <div className="ps-3 border-start border-2 border-primary border-opacity-25 mb-3">
                                        <div className="small text-primary fw-bold mb-2">AI Suggestions</div>
                                        {section.suggested.map(item => (
                                            <AISuggestionCard 
                                                key={item.suggestion.id} 
                                                item={item} 
                                                itemType={section.type} 
                                                suggestion={item.suggestion}
                                                onAccept={handleAcceptSuggestion}
                                                onIgnore={handleIgnoreSuggestion}
                                                isAccepting={isAccepting === item.suggestion.id}
                                            />
                                        ))}
                                    </div>
                                )}

                                {/* Ghost Items (Inline Expandable) */}
                                {section.ghost.length > 0 && (
                                    <div className="ps-3 border-start border-2 border-secondary border-opacity-10">
                                        <div className="small text-muted mb-2">Also Available</div>
                                        {section.ghost.map(item => (
                                            <GhostItemCard 
                                                key={item.id} 
                                                item={item} 
                                                itemType={section.type} 
                                                jobFeatures={job.features} 
                                                onPromote={handleInlinePromote} 
                                            />
                                        ))}
                                    </div>
                                )}
                            </div>
                        )
                    ))}

                    {/* --- Skills Section --- */}
                    <div className="mb-4">
                        <h6 className="fw-bold text-uppercase text-muted small mb-3 d-flex align-items-center gap-2 border-bottom pb-2">
                            <Cpu size={14}/> Skills
                        </h6>
                        <p className="small text-muted mb-3">
                            Green skills are automatically selected based on your mapping. Click to toggle manual selection (Blue).
                        </p>
                        
                        {['technical', 'soft', 'language', 'other'].map(category => (
                            groupedSkills[category].length > 0 && (
                                <div key={category} className="mb-3">
                                    <strong className="text-capitalize d-block small text-dark mb-2">{category}</strong>
                                    <div className="d-flex flex-wrap gap-2">
                                        {groupedSkills[category].map(skill => {
                                            if (!skill) return null;
                                            const isSelected = selectedSkillIds.has(skill.id);
                                            const isAutoSelected = aggregatedSkillIds.has(skill.id);
                                            
                                            return (
                                                <div key={skill.id} className="position-relative">
                                                    <input
                                                        type="checkbox"
                                                        className="btn-check"
                                                        id={`skill-check-${skill.id}`}
                                                        checked={isSelected}
                                                        onChange={() => handleToggleSkill(skill.id)}
                                                        autoComplete="off"
                                                    />
                                                    <label 
                                                        className={`btn btn-sm rounded-pill border small px-3 transition-all ${
                                                            isSelected 
                                                                ? (isAutoSelected ? 'btn-success text-white border-success' : 'btn-primary text-white border-primary') 
                                                                : 'btn-white text-muted border-light hover-border-secondary'
                                                        }`}
                                                        htmlFor={`skill-check-${skill.id}`}
                                                        style={{fontSize: '0.8rem'}}
                                                    >
                                                        {skill.name}
                                                    </label>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )
                        ))}
                    </div>
                </div>
            </div>

            <PromptModal
                isOpen={isModalOpen}
                jsonString={cvPromptJson}
                onClose={() => setIsModalOpen(false)}
            />
            
            {/* This button bar will now be constrained by the parent div */}
            <div className="d-flex justify-content-between mt-4 pt-3 border-top">
                <button type="button" className="btn btn-outline-secondary d-flex align-items-center gap-2 px-4" onClick={onPrev}>
                    <ArrowLeft size={16}/> Back
                </button>
                <button type="button" className="btn btn-outline-primary d-flex align-items-center gap-2 px-4" onClick={onNext}>
                    Next <ArrowRight size={16}/>
                </button>
            </div>
        </div>
    );
};

export default Step2_GenerateCV;