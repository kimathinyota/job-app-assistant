// frontend/src/components/applications/TailoredCVManager.jsx
import React, { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    Save, RefreshCw, ArrowLeft, Download, Loader2, 
    MoveUp, MoveDown, Trash2, Plus, Type
} from 'lucide-react';

import { 
    fetchApplicationDetails, 
    getOrCreateDerivedCV, 
    updateCV 
} from '../../api/applicationClient'; 

import { API_BASE_URL } from '../../api/client';

// Constants
const MIN_FONT_SIZE = 9;
const MAX_FONT_SIZE = 16;

// --- HELPER: INVISIBLE INPUT ---
const Editable = ({ value, onChange, placeholder, className="", style={}, multiline=false, bold=false, italic=false, align="left", type="text" }) => {
    const textareaRef = useRef(null);

    // AUTO-RESIZE LOGIC: Runs on mount and whenever value changes
    useLayoutEffect(() => {
        if (multiline && textareaRef.current) {
            // Reset height to auto to correctly calculate new scrollHeight (shrink if needed)
            textareaRef.current.style.height = 'auto'; 
            // Set to scrollHeight to expand
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
        }
    }, [value, multiline]);

    const commonStyles = {
        background: 'transparent',
        border: 'none',
        outline: 'none',
        padding: 0,
        margin: 0,
        fontFamily: 'inherit',
        fontSize: 'inherit',
        color: 'inherit',
        fontWeight: bold ? 'bold' : 'inherit',
        fontStyle: italic ? 'italic' : 'inherit',
        textAlign: align,
        width: '100%',
        resize: 'none',
        ...style
    };

    if (multiline) {
        return (
            <textarea
                ref={textareaRef}
                value={value || ""}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                className={`editable-hover ${className}`}
                style={{
                    ...commonStyles, 
                    overflow: 'hidden', 
                    whiteSpace: 'pre-wrap', 
                    wordBreak: 'break-word',
                    display: 'block',
                    minHeight: '1.5em'
                }}
                rows={1}
            />
        );
    }

    return (
        <input
            type={type}
            value={value || ""}
            onChange={(e) => onChange(e.target.value)}
            placeholder={placeholder}
            className={`editable-hover ${className}`}
            style={commonStyles}
        />
    );
};

// --- HELPER: ITEM WRAPPER ---
const ItemWrapper = ({ index, total, onMove, onDelete, children }) => (
    <div className="position-relative group mb-4">
        <div className="position-absolute top-0 start-0 h-100 d-flex flex-column justify-content-start align-items-center opacity-0 group-hover-opacity transition-opacity" 
             style={{ transform: 'translateX(-120%)', width: '24px', zIndex: 10 }}>
            <div className="btn-group-vertical shadow-sm bg-white rounded border">
                {index > 0 && <button onClick={() => onMove(index, -1)} className="btn btn-xs btn-light text-muted p-1" title="Move Up"><MoveUp size={12}/></button>}
                {index < total - 1 && <button onClick={() => onMove(index, 1)} className="btn btn-xs btn-light text-muted p-1" title="Move Down"><MoveDown size={12}/></button>}
                <button onClick={() => onDelete(index)} className="btn btn-xs btn-light text-danger p-1" title="Delete"><Trash2 size={12}/></button>
            </div>
        </div>
        {children}
    </div>
);

// --- HELPER: SECTION WRAPPER ---
const SectionWrapper = ({ sectionKey, index, total, onMove, children }) => (
    <div className="position-relative group section-hover mb-2">
        <div className="position-absolute top-0 end-0 mt-2 opacity-0 group-hover-opacity transition-opacity d-flex gap-1" 
             style={{ transform: 'translateX(110%)', width: '30px' }}>
            {index > 0 && <button onClick={() => onMove(index, -1)} className="btn btn-sm btn-light border shadow-sm text-muted p-1"><MoveUp size={16}/></button>}
            {index < total - 1 && <button onClick={() => onMove(index, 1)} className="btn btn-sm btn-light border shadow-sm text-muted p-1"><MoveDown size={16}/></button>}
        </div>
        {children}
    </div>
);

const TailoredCVManager = () => {
    const { applicationId } = useParams();
    const navigate = useNavigate();

    const [derivedCV, setDerivedCV] = useState(null);
    const [fontSize, setFontSize] = useState(12);
    
    const [sectionOrder, setSectionOrder] = useState([
        'summary', 
        'education', 
        'skills', 
        'projects', 
        'experiences', 
        'hobbies'
    ]);
    
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);
    
    // Guard against double firing in strict mode
    const hasLoaded = useRef(false);

    // --- DYNAMIC STYLES ---
    const paperStyle = {
        minHeight: '297mm', 
        width: '100%',
        maxWidth: '210mm', 
        margin: '0 auto',
        padding: '15mm 20mm',
        backgroundColor: 'white',
        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        fontFamily: '"Times New Roman", Times, serif',
        color: '#000',
        lineHeight: '1.5',
        fontSize: `${fontSize}pt`,
        transition: 'font-size 0.2s ease'
    };

    const sectionHeaderStyle = {
        borderBottom: '1px solid #000',
        paddingBottom: '4px',
        marginBottom: '12px',
        marginTop: '20px',
        textTransform: 'uppercase',
        fontWeight: 'bold',
        fontSize: '1.2em',
        letterSpacing: '0.05em'
    };

    // --- HYDRATION ---
    const hydrateCV = (cvData) => {
        const achMap = new Map(cvData.achievements?.map(a => [a.id, a]) || []);
        const hydrateItem = (item) => {
            if (!item.achievement_ids) return { ...item, achievements: [] };
            const attachedAchievements = item.achievement_ids
                .map(id => achMap.get(id))
                .filter(Boolean);
            return { ...item, achievements: attachedAchievements };
        };
        const processList = (listName, idListField) => {
            if (!cvData[listName] || !cvData[idListField]) return [];
            return cvData[listName]
                .filter(item => cvData[idListField].includes(item.id))
                .map(hydrateItem);
        };
        return {
            ...cvData,
            experiences: processList('experiences', 'selected_experience_ids'),
            education: processList('education', 'selected_education_ids'),
            projects: processList('projects', 'selected_project_ids'),
            hobbies: processList('hobbies', 'selected_hobby_ids'),
            skills: cvData.skills?.filter(s => cvData.selected_skill_ids?.includes(s.id)) || []
        };
    };

    // --- LOAD ---
    useEffect(() => {
        // Prevent double execution
        if (hasLoaded.current) return;
        hasLoaded.current = true;

        const load = async () => {
            try {
                const cvRes = await getOrCreateDerivedCV(applicationId); 
                const hydrated = hydrateCV(cvRes);
                setDerivedCV(hydrated);
            } catch (err) {
                console.error(err);
                hasLoaded.current = false; // allow retry
                alert("Failed to load CV.");
            } finally {
                setIsLoading(false);
            }
        };
        load();
    }, [applicationId]);

    // --- FONT ACTIONS ---
    const changeFontSize = (delta) => {
        setFontSize(prev => {
            const newSize = prev + delta;
            if (newSize < MIN_FONT_SIZE) return MIN_FONT_SIZE;
            if (newSize > MAX_FONT_SIZE) return MAX_FONT_SIZE;
            return newSize;
        });
    };

    // --- UPDATERS ---
    const updateRoot = (field, val) => setDerivedCV(prev => ({...prev, [field]: val}));
    const updateContact = (key, val) => setDerivedCV(prev => ({...prev, contact_info: {...prev.contact_info, [key]: val}}));
    
    const updateItem = (list, idx, field, val) => {
        setDerivedCV(prev => {
            const copy = [...prev[list]];
            copy[idx] = { ...copy[idx], [field]: val };
            return { ...prev, [list]: copy };
        });
    };

    const updateAchievementText = (listName, parentIdx, achIdx, newVal) => {
        setDerivedCV(prev => {
            const parentList = [...prev[listName]];
            const parent = { ...parentList[parentIdx] };
            const achs = [...parent.achievements];
            const targetAch = { ...achs[achIdx], text: newVal };
            achs[achIdx] = targetAch;
            parent.achievements = achs;
            parentList[parentIdx] = parent;
            
            // Note: Global achievements list isn't strictly necessary for rendering derived view
            // as we rely on the hydrated parent objects, but keeping it synced is good practice.
            return { ...prev, [listName]: parentList };
        });
    };

    const addAchievement = (listName, parentIdx) => {
        const newId = `new_${Date.now()}`;
        const newAch = { id: newId, text: "" };
        setDerivedCV(prev => {
            const parentList = [...prev[listName]];
            const parent = { ...parentList[parentIdx] };
            const achs = [...(parent.achievements || []), newAch];
            parent.achievement_ids = [...(parent.achievement_ids || []), newId];
            parent.achievements = achs;
            parentList[parentIdx] = parent;
            return { ...prev, [listName]: parentList, achievements: [...prev.achievements, newAch] };
        });
    };

    const deleteAchievement = (listName, parentIdx, achIdx) => {
        setDerivedCV(prev => {
            const parentList = [...prev[listName]];
            const parent = { ...parentList[parentIdx] };
            const targetId = parent.achievements[achIdx].id;
            parent.achievements = parent.achievements.filter((_, i) => i !== achIdx);
            parent.achievement_ids = parent.achievement_ids.filter(id => id !== targetId);
            parentList[parentIdx] = parent;
            return { ...prev, [listName]: parentList };
        });
    };

    const moveItem = (list, idx, dir) => {
        setDerivedCV(prev => {
            const copy = [...prev[list]];
            const [item] = copy.splice(idx, 1);
            copy.splice(idx + dir, 0, item);
            return { ...prev, [list]: copy };
        });
    };

    const deleteItem = (list, idx) => {
        if(!window.confirm("Remove this item?")) return;
        setDerivedCV(prev => {
            const copy = [...prev[list]];
            copy.splice(idx, 1);
            return { ...prev, [list]: copy };
        });
    };

    const moveSection = (index, direction) => {
        const newOrder = [...sectionOrder];
        const [moved] = newOrder.splice(index, 1);
        newOrder.splice(index + direction, 0, moved);
        setSectionOrder(newOrder);
    };

    // --- API ACTIONS ---
    const handleSave = async () => {
        setIsSaving(true);
        try { 
            await updateCV(derivedCV.id, derivedCV); 
        } 
        catch (e) { console.error(e); alert("Save failed"); } 
        finally { setIsSaving(false); }
    };

    const handleSync = async () => {
        if(!window.confirm("Resync will overwrite edits. Continue?")) return;
        setIsSyncing(true);
        try {
            const res = await getOrCreateDerivedCV(applicationId, { force_refresh: true });
            setDerivedCV(hydrateCV(res));
        } catch (e) { alert("Sync failed"); } 
        finally { setIsSyncing(false); }
    };

    const handleExport = () => {
        const url = `${API_BASE_URL}/cv/${derivedCV.id}/export/pdf?sections=${sectionOrder.join(',')}&font_size=${fontSize}`;
        window.open(url, '_blank');
    };

    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin"/></div>;
    if (!derivedCV) return <div>Error loading.</div>;

    // --- RENDER SECTIONS ---
    const renderSection = (key, idx) => {
        switch(key) {
            case 'summary':
                return derivedCV.summary ? (
                    <div className="mb-4">
                        <div style={sectionHeaderStyle}>Professional Summary</div>
                        <Editable value={derivedCV.summary} onChange={v => updateRoot('summary', v)} multiline/>
                    </div>
                ) : null;

            case 'education':
                return derivedCV.education?.length > 0 ? (
                    <div className="mb-4">
                        <div style={sectionHeaderStyle}>Education</div>
                        {derivedCV.education.map((edu, idx) => (
                            <ItemWrapper key={edu.id} index={idx} total={derivedCV.education.length} onMove={(i,d) => moveItem('education',i,d)} onDelete={(i) => deleteItem('education',i)}>
                                <div className="d-flex justify-content-between">
                                    <div className="d-flex gap-3 flex-grow-1">
                                        <Editable value={edu.institution} onChange={v => updateItem('education', idx, 'institution', v)} bold style={{flex:1}}/>
                                        <Editable value={edu.degree} onChange={v => updateItem('education', idx, 'degree', v)} bold style={{flex:1, textAlign:'right'}}/>
                                    </div>
                                </div>
                                <div className="d-flex justify-content-between fst-italic small">
                                    <Editable value={edu.field} onChange={v => updateItem('education', idx, 'field', v)}/>
                                    <div className="d-flex gap-1" style={{minWidth:'160px', justifyContent:'flex-end'}}>
                                        <Editable type="date" value={edu.start_date} onChange={v => updateItem('education', idx, 'start_date', v)} align="right" style={{width:'auto'}}/>
                                        <span>-</span>
                                        <Editable type="date" value={edu.end_date} onChange={v => updateItem('education', idx, 'end_date', v)} align="right" style={{width:'auto'}}/>
                                    </div>
                                </div>
                                {edu.achievements?.length > 0 && (
                                    <ul className="mb-0 ps-3 mt-1 small">
                                        {edu.achievements.map((ach, aIdx) => (
                                            <li key={aIdx} className="mb-1 position-relative group">
                                                <div className="d-flex align-items-start gap-2 w-100">
                                                    <Editable value={ach.text} onChange={v => updateAchievementText('education', idx, aIdx, v)} multiline className="w-100"/>
                                                    <button onClick={() => deleteAchievement('education', idx, aIdx)} className="btn btn-link p-0 text-danger opacity-0 group-hover-opacity flex-shrink-0"><Trash2 size={12}/></button>
                                                </div>
                                            </li>
                                        ))}
                                        <li className="list-unstyled mt-1 opacity-50 hover-opacity-100 cursor-pointer text-primary small" onClick={() => addAchievement('education', idx)}>
                                            <Plus size={12} className="me-1"/> Add detail
                                        </li>
                                    </ul>
                                )}
                            </ItemWrapper>
                        ))}
                    </div>
                ) : null;

            case 'skills':
                return derivedCV.skills?.length > 0 ? (
                    <div className="mb-4">
                        <div style={sectionHeaderStyle}>Technical Skills</div>
                        {['technical', 'soft', 'language', 'tool'].map(cat => {
                            const skills = derivedCV.skills.filter(s => s.category === cat);
                            if(!skills.length) return null;
                            return (
                                <div key={cat} className="d-flex mb-1 small">
                                    <strong className="text-capitalize" style={{minWidth: '140px'}}>{cat}:</strong>
                                    <span>{skills.map(s => s.name).join(", ")}</span>
                                </div>
                            )
                        })}
                    </div>
                ) : null;

            case 'projects':
                return derivedCV.projects?.length > 0 ? (
                    <div className="mb-4">
                        <div style={sectionHeaderStyle}>Academic & Research Projects</div>
                        {derivedCV.projects.map((proj, idx) => (
                            <ItemWrapper key={proj.id} index={idx} total={derivedCV.projects.length} onMove={(i,d) => moveItem('projects',i,d)} onDelete={(i) => deleteItem('projects',i)}>
                                <div className="d-flex justify-content-between mb-1">
                                    <div className="d-flex w-100 align-items-baseline">
                                        <Editable value={proj.title} onChange={v => updateItem('projects', idx, 'title', v)} bold style={{width:'auto'}}/>
                                        <span className="mx-2">|</span>
                                        <Editable value={proj.description} onChange={v => updateItem('projects', idx, 'description', v)} italic/>
                                    </div>
                                    <div className="d-flex gap-1 ms-3" style={{minWidth:'100px', justifyContent:'flex-end'}}>
                                         <Editable value={proj.dates} onChange={v => updateItem('projects', idx, 'dates', v)} align="right" placeholder="Dates"/>
                                    </div>
                                </div>
                                <ul className="mb-0 ps-3 small">
                                    {proj.achievements?.map((ach, aIdx) => (
                                        <li key={aIdx} className="mb-1 position-relative group">
                                            <div className="d-flex align-items-start gap-2 w-100">
                                                <Editable value={ach.text} onChange={v => updateAchievementText('projects', idx, aIdx, v)} multiline className="w-100"/>
                                                <button onClick={() => deleteAchievement('projects', idx, aIdx)} className="btn btn-link p-0 text-danger opacity-0 group-hover-opacity flex-shrink-0"><Trash2 size={12}/></button>
                                            </div>
                                        </li>
                                    ))}
                                    <li className="list-unstyled mt-1 opacity-50 hover-opacity-100 cursor-pointer text-primary small" onClick={() => addAchievement('projects', idx)}>
                                        <Plus size={12} className="me-1"/> Add detail
                                    </li>
                                </ul>
                            </ItemWrapper>
                        ))}
                    </div>
                ) : null;

            case 'experiences':
                return derivedCV.experiences?.length > 0 ? (
                    <div className="mb-4">
                        <div style={sectionHeaderStyle}>Experience</div>
                        {derivedCV.experiences.map((exp, idx) => (
                            <ItemWrapper key={exp.id} index={idx} total={derivedCV.experiences.length} onMove={(i,d) => moveItem('experiences',i,d)} onDelete={(i) => deleteItem('experiences',i)}>
                                <div className="d-flex justify-content-between">
                                    <Editable value={exp.title} onChange={v => updateItem('experiences', idx, 'title', v)} bold/>
                                    <Editable value={exp.company} onChange={v => updateItem('experiences', idx, 'company', v)} bold align="right"/>
                                </div>
                                <div className="d-flex justify-content-between fst-italic small mb-1">
                                    <Editable value={exp.location} onChange={v => updateItem('experiences', idx, 'location', v)} placeholder="Location"/>
                                    <div className="d-flex gap-1" style={{minWidth:'160px', justifyContent:'flex-end'}}>
                                        <Editable type="date" value={exp.start_date} onChange={v => updateItem('experiences', idx, 'start_date', v)} align="right" style={{width:'auto'}} placeholder="Start"/>
                                        <span>-</span>
                                        <Editable type="date" value={exp.end_date} onChange={v => updateItem('experiences', idx, 'end_date', v)} align="right" style={{width:'auto'}} placeholder="End"/>
                                    </div>
                                </div>
                                <ul className="mb-0 ps-3 small">
                                    {exp.achievements?.map((ach, aIdx) => (
                                        <li key={aIdx} className="mb-1 position-relative group">
                                            <div className="d-flex align-items-start gap-2 w-100">
                                                <Editable value={ach.text} onChange={v => updateAchievementText('experiences', idx, aIdx, v)} multiline className="w-100"/>
                                                <button onClick={() => deleteAchievement('experiences', idx, aIdx)} className="btn btn-link p-0 text-danger opacity-0 group-hover-opacity flex-shrink-0"><Trash2 size={12}/></button>
                                            </div>
                                        </li>
                                    ))}
                                    <li className="list-unstyled mt-1 opacity-50 hover-opacity-100 cursor-pointer text-primary small" onClick={() => addAchievement('experiences', idx)}>
                                        <Plus size={12} className="me-1"/> Add detail
                                    </li>
                                </ul>
                            </ItemWrapper>
                        ))}
                    </div>
                ) : null;

            case 'hobbies':
                return derivedCV.hobbies?.length > 0 ? (
                    <div className="mb-4">
                        <div style={sectionHeaderStyle}>Interests & Hobbies</div>
                        <ul className="mb-0 ps-3 small">
                            {derivedCV.hobbies.map((hobby, idx) => (
                                <ItemWrapper key={hobby.id} index={idx} total={derivedCV.hobbies.length} onMove={(i,d) => moveItem('hobbies',i,d)} onDelete={(i) => deleteItem('hobbies',i)}>
                                    <li className="d-flex gap-2">
                                        <Editable value={hobby.name} onChange={v => updateItem('hobbies', idx, 'name', v)} bold style={{width:'auto'}}/>
                                        <span>:</span>
                                        <Editable value={hobby.description} onChange={v => updateItem('hobbies', idx, 'description', v)}/>
                                    </li>
                                </ItemWrapper>
                            ))}
                        </ul>
                    </div>
                ) : null;
            
            default: return null;
        }
    };

    return (
        <div className="container-fluid bg-light min-vh-100 pb-5">
            <style>{`
                .editable-hover:hover { background-color: rgba(0,0,0,0.03); cursor: text; border-radius: 2px; }
                .group:hover .group-hover-opacity { opacity: 1 !important; }
                .section-hover:hover .group-hover-opacity { opacity: 1 !important; }
            `}</style>

            {/* HEADER */}
            <div className="sticky-top bg-white border-bottom shadow-sm px-4 py-2 mb-4 d-flex justify-content-between align-items-center z-3">
                <div className="d-flex align-items-center gap-3">
                    <button onClick={() => navigate(-1)} className="btn btn-link text-dark p-0"><ArrowLeft size={20}/></button>
                    <div>
                        <h5 className="fw-bold mb-0">Tailored CV Editor</h5>
                        <small className="text-muted">{derivedCV.name}</small>
                    </div>
                </div>
                <div className="d-flex gap-2 align-items-center">
                    {/* FONT CONTROLS */}
                    <div className="btn-group btn-group-sm me-2 shadow-sm border rounded">
                        <button className="btn btn-white border-end" onClick={() => changeFontSize(-1)} title="Decrease Font">
                            <Type size={14} className="text-muted"/> -
                        </button>
                        <span className="btn btn-white disabled border-end fw-bold text-dark px-3" style={{width: '3.5rem'}}>
                            {fontSize}pt
                        </span>
                        <button className="btn btn-white" onClick={() => changeFontSize(1)} title="Increase Font">
                            <Type size={14} className="text-dark"/> +
                        </button>
                    </div>

                    <button className="btn btn-sm btn-outline-secondary d-flex gap-2 align-items-center" onClick={handleSync} disabled={isSyncing}>
                        <RefreshCw size={14} className={isSyncing ? "animate-spin":""}/> Sync
                    </button>
                    <button className="btn btn-sm btn-outline-primary d-flex gap-2 align-items-center" onClick={handleSave} disabled={isSaving}>
                        {isSaving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>} Save Draft
                    </button>
                    <button className="btn btn-sm btn-dark d-flex gap-2 align-items-center" onClick={handleExport}>
                        <Download size={14}/> Download PDF
                    </button>
                </div>
            </div>

            {/* DOCUMENT */}
            <div style={paperStyle}>
                
                {/* 1. Header Info (Fixed Position) */}
                <div className="text-center mb-3">
                    <div className="d-flex justify-content-center gap-2 mb-1">
                        <Editable value={derivedCV.first_name} onChange={v => updateRoot('first_name', v)} style={{fontSize: '2.2em', textAlign: 'right', width:'auto'}} bold placeholder="First Name"/>
                        <Editable value={derivedCV.last_name} onChange={v => updateRoot('last_name', v)} style={{fontSize: '2.2em', textAlign: 'left', width:'auto'}} bold placeholder="Last Name"/>
                    </div>
                    <div className="d-flex justify-content-center gap-2 small text-muted">
                        {['email', 'phone', 'location', 'linkedin'].map(k => derivedCV.contact_info?.[k] && (
                            <React.Fragment key={k}>
                                <Editable value={derivedCV.contact_info[k]} onChange={v => updateContact(k, v)} style={{width:'auto', textAlign:'center'}}/>
                                <span className="mx-1">|</span>
                            </React.Fragment>
                        ))}
                    </div>
                </div>

                {/* DYNAMIC SECTIONS */}
                {sectionOrder.map((sectionKey, idx) => (
                    <SectionWrapper 
                        key={sectionKey} 
                        sectionKey={sectionKey} 
                        index={idx} 
                        total={sectionOrder.length} 
                        onMove={moveSection}
                    >
                        {renderSection(sectionKey, idx)}
                    </SectionWrapper>
                ))}

            </div>
        </div>
    );
};

export default TailoredCVManager;