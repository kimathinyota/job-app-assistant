import React, { useState, useEffect, useRef, useLayoutEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    Save, RefreshCw, ArrowLeft, Loader2, 
    MoveUp, MoveDown, Trash2, Plus, Type, CheckCircle, UploadCloud,
    FileText, FileType as FileTypeIcon, FileCode, Archive, Download, Settings,
    Phone, Mail, Globe, MapPin, Linkedin, HelpCircle, Search, ChevronRight, 
    ChevronDown, Link as LinkIcon, Award, Briefcase, GraduationCap, FolderGit2, Wrench, Heart
} from 'lucide-react';

import { getOrCreateDerivedCV, updateCV as updateDerivedCV, fetchJobDetails } from '../../api/applicationClient'; 
import { fetchCVDetails, updateBaseCV, exportCV } from '../../api/cvClient';
import { apiClient } from '../../api/client';

const MIN_FONT_SIZE = 9;
const MAX_FONT_SIZE = 14;

const DEFAULT_TITLES = {
    summary: 'Professional Summary',
    experiences: 'Experience',
    education: 'Education',
    projects: 'Academic & Research Projects',
    skills: 'Technical Skills',
    hobbies: 'Interests & Hobbies'
};

// HELPER: FORMAT DATE
const formatDateDisplay = (start, end) => {
    const parse = (d) => {
        if (!d || ['present', 'now', 'current'].includes(d.toLowerCase())) return null;
        return new Date(d).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
    };
    const s = parse(start);
    const e = parse(end);
    if (!s && !e) return "";
    if (!s && e) return e;
    if (s && !e) return `${s} - Present`;
    if (s === e) return s;
    return `${s} - ${e}`;
};

// HELPER: EDITABLE DATE
const EditableDate = ({ value, onChange, placeholder, align = "right" }) => {
    const [isEditing, setIsEditing] = useState(false);
    if (isEditing) {
        return (
            <input
                type="month" value={value ? value.substring(0, 7) : ""}
                onChange={(e) => onChange(e.target.value)} onBlur={() => setIsEditing(false)}
                autoFocus className="editable-hover"
                style={{ background: 'transparent', border: 'none', outline: 'none', fontSize: 'inherit', fontFamily: 'inherit', textAlign: align, width: '110px' }}
            />
        );
    }
    return (
        <span 
            className={`editable-hover px-1 cursor-text ${!value ? 'text-muted' : ''}`} 
            onClick={() => setIsEditing(true)}
            style={{ textAlign: align, display: 'inline-block', minWidth: '40px', opacity: !value ? 0.3 : 1 }}
        >
            {value ? formatDateDisplay(value, value) : placeholder}
        </span>
    );
};

// HELPER: INVISIBLE INPUT
const Editable = ({ value, onChange, placeholder, className="", style={}, multiline=false, bold=false, italic=false, align="left", type="text", resizeTrigger }) => {
    const textareaRef = useRef(null);
    useLayoutEffect(() => {
        if (multiline && textareaRef.current) {
            textareaRef.current.style.height = '0px'; 
            textareaRef.current.style.height = `${textareaRef.current.scrollHeight + 2}px`; 
        }
    }, [value, multiline, resizeTrigger]);

    const commonStyles = {
        background: 'transparent', border: 'none', outline: 'none', padding: 0, margin: 0,
        fontFamily: 'inherit', fontSize: 'inherit', color: 'inherit',
        fontWeight: bold ? 'bold' : 'inherit', fontStyle: italic ? 'italic' : 'inherit',
        textAlign: align, width: '100%', resize: 'none', ...style
    };

    if (multiline) {
        return (
            <textarea
                ref={textareaRef} value={value || ""} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
                className={`editable-hover ${className} ${!value ? 'opacity-50' : ''}`}
                style={{ ...commonStyles, overflow: 'hidden', whiteSpace: 'pre-wrap', wordBreak: 'break-word', display: 'block', minHeight: '1.5em' }}
                rows={1}
            />
        );
    }
    return <input type={type} value={value || ""} onChange={(e) => onChange(e.target.value)} placeholder={placeholder} className={`editable-hover ${className} ${!value ? 'opacity-50' : ''}`} style={commonStyles} />;
};

// HELPER: EDITABLE DEGREE & FIELD SPLIT INPUT
const EditableDegreeField = ({ degree, field, onDegreeChange, onFieldChange }) => {
    const [isEditing, setIsEditing] = useState(false);
    const containerRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (containerRef.current && !containerRef.current.contains(event.target)) {
                setIsEditing(false);
            }
        };
        if (isEditing) document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [isEditing]);

    if (isEditing) {
        return (
            <div ref={containerRef} className="d-flex align-items-baseline gap-1 bg-light px-2 py-1 rounded shadow-sm border border-primary w-100" style={{ zIndex: 20 }}>
                <input
                    autoFocus type="text" value={degree || ""} onChange={(e) => onDegreeChange(e.target.value)}
                    placeholder="Degree (e.g. BSc)"
                    className="border-0 shadow-none fw-bold bg-transparent p-0 m-0 text-end text-primary flex-grow-1"
                    style={{ fontSize: 'inherit', fontFamily: 'inherit', outline: 'none', width: '45%' }}
                />
                <span className="fw-bold text-dark">,</span>
                <input
                    type="text" value={field || ""} onChange={(e) => onFieldChange(e.target.value)}
                    placeholder="Field of Study"
                    className="border-0 shadow-none fw-bold bg-transparent p-0 m-0 text-primary flex-grow-1"
                    style={{ fontSize: 'inherit', fontFamily: 'inherit', outline: 'none', width: '50%' }}
                />
            </div>
        );
    }

    const displayValue = [degree, field].filter(Boolean).join(", ");

    return (
        <span 
            className={`editable-hover cursor-text fw-bold ${!displayValue ? 'opacity-50 text-muted' : ''}`} 
            onClick={() => setIsEditing(true)}
            style={{ display: 'inline-block', minWidth: '150px', width: 'auto', flexGrow: 1 }}
        >
            {displayValue || "Degree, Field of Study"}
        </span>
    );
};

// HELPER: ITEM WRAPPER
const ItemWrapper = ({ index, total, onMove, onDelete, children }) => (
    <div className="position-relative group" style={{ marginBottom: '10pt' }}>
        <div className="position-absolute top-0 start-0 h-100 d-flex flex-column justify-content-start align-items-center opacity-0 group-hover-opacity" style={{ transform: 'translateX(-120%)', width: '24px', zIndex: 10 }}>
            <div className="btn-group-vertical shadow-sm bg-white rounded border">
                {index > 0 && <button onClick={() => onMove(index, -1)} className="btn btn-xs btn-light text-muted p-1" title="Move Up"><MoveUp size={12}/></button>}
                {index < total - 1 && <button onClick={() => onMove(index, 1)} className="btn btn-xs btn-light text-muted p-1" title="Move Down"><MoveDown size={12}/></button>}
                <button onClick={() => onDelete(index)} className="btn btn-xs btn-light text-danger p-1" title="Delete"><Trash2 size={12}/></button>
            </div>
        </div>
        {children}
    </div>
);

// HELPER: SECTION WRAPPER
const SectionWrapper = ({ index, total, onMove, children }) => (
    <div className="position-relative group section-hover" style={{ marginBottom: '16pt' }}>
        <div className="position-absolute top-0 end-0 mt-2 opacity-0 group-hover-opacity d-flex gap-1" style={{ transform: 'translateX(110%)', width: '30px', zIndex: 10 }}>
            {index > 0 && <button onClick={() => onMove(index, -1)} className="btn btn-sm btn-light border shadow-sm text-muted p-1"><MoveUp size={16}/></button>}
            {index < total - 1 && <button onClick={() => onMove(index, 1)} className="btn btn-sm btn-light border shadow-sm text-muted p-1"><MoveDown size={16}/></button>}
        </div>
        {children}
    </div>
);

const QuickCVEditor = () => {
    const { applicationId, cvId } = useParams();
    const navigate = useNavigate();

    const [cvData, setCvData] = useState(null);
    const [fontSize, setFontSize] = useState(11); 
    const [fontFamily, setFontFamily] = useState('Arial');
    const [isDerived, setIsDerived] = useState(false);
    
    // Config States
    const [sectionOrder, setSectionOrder] = useState(['summary', 'education', 'skills', 'projects', 'experiences', 'hobbies']);
    const [sectionTitles, setSectionTitles] = useState(DEFAULT_TITLES);
    const [skillCategories, setSkillCategories] = useState(['technical', 'soft', 'language', 'tool']);
    
    // UI States
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [isSyncing, setIsSyncing] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    
    // Modals
    const [isSkillModalOpen, setIsSkillModalOpen] = useState(false);
    const [addingCategoryAt, setAddingCategoryAt] = useState(null);
    const [newCategoryName, setNewCategoryName] = useState("");
    const [expandedGhostCats, setExpandedGhostCats] = useState({}); 
    
    const [isContactModalOpen, setIsContactModalOpen] = useState(false);
    const [contactFormData, setContactFormData] = useState([]);

    // Ghost Item Modal & Triage Data
    const [restoreModalData, setRestoreModalData] = useState(null);
    const [jobFeatures, setJobFeatures] = useState([]);
    
    const [mappingWizardStep, setMappingWizardStep] = useState('FEATURE'); 
    const [mappingSearch, setMappingSearch] = useState(""); 
    const [mappingFilterType, setMappingFilterType] = useState("ALL"); 
    const [selectedFeatureForMapping, setSelectedFeatureForMapping] = useState(null);
    const [pendingManualMappings, setPendingManualMappings] = useState([]); 
    
    const hasLoaded = useRef(false);

    const paperStyle = {
        minHeight: '297mm', width: '100%', maxWidth: '210mm', margin: '0 auto', padding: '12.7mm',
        backgroundColor: 'white', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
        fontFamily: fontFamily, color: '#000', lineHeight: '1.2', fontSize: `${fontSize}pt`
    };

    const sectionHeaderStyle = {
        borderBottom: '1px solid #000', paddingBottom: '2pt', marginBottom: '8pt', marginTop: '16pt',
        textTransform: 'uppercase', fontWeight: 'bold', fontSize: '1.15em', letterSpacing: '0.02em'
    };

    // HYDRATION
    const hydrateCV = (data) => {
        const achMap = new Map(data.achievements?.map(a => [a.id, a]) || []);
        
        const hydrateItem = (item) => {
            let populatedItem = { ...item, achievements: [] };
            if (item.achievement_ids) populatedItem.achievements = item.achievement_ids.map(id => achMap.get(id)).filter(Boolean);
            return populatedItem;
        };

        const processList = (listName, idListField, mapFunc = hydrateItem) => {
            if (!data[listName]) return { active: [], inactive: [] };
            if (data[idListField]) {
                const active = data[idListField].map(id => data[listName].find(i => i.id === id)).filter(Boolean).map(mapFunc);
                const inactive = data[listName].filter(item => !data[idListField].includes(item.id)).map(mapFunc);
                return { active, inactive };
            }
            return { active: data[listName].map(mapFunc), inactive: [] };
        };

        const expData = processList('experiences', 'selected_experience_ids');
        const eduData = processList('education', 'selected_education_ids'); 
        const projData = processList('projects', 'selected_project_ids');
        const hobData = processList('hobbies', 'selected_hobby_ids');

        const activeSkills = [];
        const inactiveSkills = [];
        (data.skills || []).forEach(s => {
            if (!data.selected_skill_ids || data.selected_skill_ids.includes(s.id)) activeSkills.push(s);
            else inactiveSkills.push(s);
        });

        return {
            ...data,
            experiences: expData.active, inactive_experiences: expData.inactive,
            education: eduData.active, inactive_education: eduData.inactive,
            projects: projData.active, inactive_projects: projData.inactive,
            hobbies: hobData.active, inactive_hobbies: hobData.inactive,
            skills: activeSkills, inactive_skills: inactiveSkills,
            mapping_reasons: data.mapping_reasons || {}
        };
    };

    // LOAD
    useEffect(() => {
        if (hasLoaded.current) return;
        hasLoaded.current = true;

        const load = async () => {
            try {
                let res;
                if (applicationId) {
                    res = await getOrCreateDerivedCV(applicationId);
                    setIsDerived(true);
                    
                    if (res.job_id) {
                        try {
                            const jobRes = await fetchJobDetails(res.job_id);
                            if (jobRes && jobRes.data && jobRes.data.features) {
                                setJobFeatures(jobRes.data.features);
                            }
                        } catch(e) { console.warn("Could not fetch features for mapping triage."); }
                    }
                } else if (cvId) {
                    res = await fetchCVDetails(cvId);
                    setIsDerived(!!res.base_cv_id); 
                }
                
                if (res.section_order?.length) setSectionOrder(res.section_order);
                if (res.section_titles) setSectionTitles({...DEFAULT_TITLES, ...res.section_titles});
                
                const hydrated = hydrateCV(res);
                setCvData(hydrated);

                if (hydrated.skills?.length || hydrated.inactive_skills?.length) {
                    const uniqueCats = [...new Set([...(hydrated.skills || []), ...(hydrated.inactive_skills || [])].map(s => s.category || 'other'))];
                    setSkillCategories(prev => {
                        const existing = res.skill_category_order?.length ? res.skill_category_order : prev;
                        const missing = uniqueCats.filter(c => !existing.includes(c));
                        return [...existing, ...missing];
                    });
                }
            } catch (err) { alert("Failed to load CV."); } 
            finally { setIsLoading(false); }
        };
        load();
    }, [applicationId, cvId]);

    const changeFontSize = (delta) => setFontSize(prev => Math.min(Math.max(prev + delta, MIN_FONT_SIZE), MAX_FONT_SIZE));

    // UPDATERS
    const updateRoot = (field, val) => setCvData(prev => ({...prev, [field]: val}));
    const updateContact = (key, val) => setCvData(prev => ({...prev, contact_info: {...prev.contact_info, [key]: val}}));
    
    const updateItem = (list, idx, field, val) => {
        setCvData(prev => {
            const copy = [...prev[list]];
            copy[idx] = { ...copy[idx], [field]: val };
            return { ...prev, [list]: copy };
        });
    };

    const updateAchievementText = (listName, parentIdx, achIdx, newVal) => {
        setCvData(prev => {
            const parentList = [...prev[listName]];
            const parent = { ...parentList[parentIdx] };
            const achs = [...parent.achievements];
            achs[achIdx] = { ...achs[achIdx], text: newVal };
            parent.achievements = achs;
            parentList[parentIdx] = parent;
            return { ...prev, [listName]: parentList };
        });
    };

    const addAchievement = (listName, parentIdx) => {
        const newId = `new_${Date.now()}`;
        setCvData(prev => {
            const parentList = [...prev[listName]];
            const parent = { ...parentList[parentIdx] };
            const achs = [...(parent.achievements || []), { id: newId, text: "" }];
            parent.achievement_ids = [...(parent.achievement_ids || []), newId];
            parent.achievements = achs;
            parentList[parentIdx] = parent;
            return { ...prev, [listName]: parentList, achievements: [...(prev.achievements || []), { id: newId, text: "" }] };
        });
    };

    const deleteAchievement = (listName, parentIdx, achIdx) => {
        setCvData(prev => {
            const parentList = [...prev[listName]];
            const parent = { ...parentList[parentIdx] };
            const targetId = parent.achievements[achIdx].id;
            parent.achievements = parent.achievements.filter((_, i) => i !== achIdx);
            parent.achievement_ids = parent.achievement_ids.filter(id => id !== targetId);
            parentList[parentIdx] = parent;
            return { ...prev, [listName]: parentList };
        });
    };

    const moveAchievement = (listName, parentIdx, achIdx, dir) => {
        setCvData(prev => {
            const copyList = [...prev[listName]];
            const parent = { ...copyList[parentIdx] };
            const achs = [...parent.achievements];
            if (achIdx + dir < 0 || achIdx + dir >= achs.length) return prev;
            const temp = achs[achIdx];
            achs[achIdx] = achs[achIdx + dir];
            achs[achIdx + dir] = temp;
            parent.achievements = achs;
            parent.achievement_ids = achs.map(a => a.id);
            copyList[parentIdx] = parent;
            return { ...prev, [listName]: copyList };
        });
    };

    const moveItem = (list, idx, dir) => {
        setCvData(prev => {
            const copy = [...prev[list]];
            const [item] = copy.splice(idx, 1);
            copy.splice(idx + dir, 0, item);
            return { ...prev, [list]: copy };
        });
    };

    const deleteItem = (list, idx) => {
        if(!window.confirm("Remove this item?")) return;
        setCvData(prev => {
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

    // GHOST CHILDREN RESTORE WIZARD
    const handleRestoreClick = (listName, index, item) => {
        setRestoreModalData({ listName, index, item });
        setMappingSearch("");
        setMappingFilterType("ALL");
        setMappingWizardStep('FEATURE'); // ALWAYS START ON MAPPING
    };

    const selectFeatureForMapping = (feature) => {
        setSelectedFeatureForMapping(feature);
        setMappingWizardStep('SEGMENT');
    };

    const finalizeRestore = (reasonType, segmentData = null) => {
        const { listName, index, item } = restoreModalData;
        
        if (reasonType === 'MAPPED' && segmentData && selectedFeatureForMapping) {
            const newMappingPair = {
                feature_id: selectedFeatureForMapping.id,
                context_item_id: item.id,
                context_item_type: listName,
                feature_text: selectedFeatureForMapping.description,
                context_item_text: segmentData.text,
                strength: 1.0,
                status: "user_manual",
                meta: {
                    best_match: {
                        segment_text: segmentData.text,
                        segment_type: segmentData.type,
                        score: 1.0,
                        lineage: [
                            { id: item.id, type: listName, name: item.title || item.name || item.degree || 'Item' },
                            ...(segmentData.type === 'achievement' ? [{ id: segmentData.id, type: 'achievement', name: 'Achievement' }] : [])
                        ]
                    },
                    supporting_matches: [],
                    rejected_matches: [],
                    summary_note: "Manually mapped during Quick Edit."
                }
            };
            setPendingManualMappings(prev => [...prev, newMappingPair]);
        }

        setCvData(prev => {
            const newInactive = [...prev[`inactive_${listName}`]];
            newInactive.splice(index, 1);
            const newActive = [...prev[listName], item];
            
            let finalReason = reasonType;
            if (reasonType === 'MAPPED' && selectedFeatureForMapping) finalReason = `MAPPED:${selectedFeatureForMapping.id}`;

            return {
                ...prev,
                [`inactive_${listName}`]: newInactive,
                [listName]: newActive,
                mapping_reasons: { ...(prev.mapping_reasons || {}), [item.id]: finalReason }
            };
        });

        setRestoreModalData(null);
        setSelectedFeatureForMapping(null);
        setMappingWizardStep('FEATURE');
    };

    const getSegmentsForItem = (item, listName) => {
        const segments = [];
        if (listName === 'skills') {
            segments.push({ id: item.id, type: 'skill', text: item.name, label: 'Skill Name' });
        } else {
            if (item.description) segments.push({ id: item.id, type: 'description', text: item.description, label: 'Description/Summary' });
            if (item.title) segments.push({ id: item.id, type: 'title', text: item.title, label: 'Title' });
            if (item.degree || item.field) segments.push({ id: item.id, type: 'degree', text: [item.degree, item.field].filter(Boolean).join(", "), label: 'Degree' });
            
            if (item.achievements && item.achievements.length > 0) {
                item.achievements.forEach((ach, i) => {
                    segments.push({ id: ach.id, type: 'achievement', text: ach.text, label: `Bullet ${i+1}` });
                });
            }
        }
        return segments;
    };


    // CONTACT MANAGER ACTIONS
    const getContactIcon = (key, size = 16) => {
        const k = (key || '').toLowerCase();
        const props = { size, className: "currentColor" };
        if (k.includes('phone') || k.includes('tel') || k.includes('mobile')) return <Phone {...props} />;
        if (k.includes('mail') || k.includes('email')) return <Mail {...props} />;
        if (k.includes('linkedin')) return <Linkedin {...props} />;
        if (k.includes('address') || k.includes('location') || k.includes('city')) return <MapPin {...props} />;
        return <Globe {...props} />; 
    };

    const openContactPanel = () => {
        const asArray = Object.entries(cvData.contact_info || {}).map(([key, value], index) => ({ id: Date.now() + index, key, value }));
        setContactFormData(asArray);
        setIsContactModalOpen(true);
    };

    const saveContactPanel = () => {
        const newContactInfo = {};
        contactFormData.forEach(item => {
            if (item.key && item.key.trim() && item.value && item.value.trim()) newContactInfo[item.key.trim()] = item.value.trim();
        });
        setCvData(prev => ({ ...prev, contact_info: newContactInfo }));
        setIsContactModalOpen(false);
    };

    const updateContactFormField = (index, field, val) => {
        setContactFormData(prev => prev.map((item, i) => i === index ? { ...item, [field]: val } : item));
    };

    // SKILL MANAGER ACTIONS
    const updateSkill = (idx, field, val) => {
        setCvData(prev => {
            const skills = [...prev.skills];
            skills[idx] = { ...skills[idx], [field]: val };
            return { ...prev, skills };
        });
    };

    const deleteSkill = (idx) => {
        setCvData(prev => {
            const skillToRemove = prev.skills[idx];
            const updatedSkills = prev.skills.filter((_, i) => i !== idx);
            
            // Only add to inactive if it has an actual name to prevent saving blanks
            if (skillToRemove && skillToRemove.name && skillToRemove.name.trim() !== '') {
                return {
                    ...prev,
                    skills: updatedSkills,
                    inactive_skills: [skillToRemove, ...(prev.inactive_skills || [])]
                };
            }
            
            return { ...prev, skills: updatedSkills };
        });
    };

    const addSkillToCategory = (category) => {
        setCvData(prev => ({ ...prev, skills: [{ id: `new_skill_${Date.now()}`, name: '', category }, ...prev.skills] }));
    };

    const handleInsertCategory = (index) => {
        if (!newCategoryName.trim()) { setAddingCategoryAt(null); return; }
        const normalized = newCategoryName.trim().toLowerCase().replace(/\s+/g, '_');
        if (!skillCategories.includes(normalized)) {
            setSkillCategories(prev => {
                const newCats = [...prev];
                newCats.splice(index, 0, normalized);
                return newCats;
            });
        }
        setNewCategoryName("");
        setAddingCategoryAt(null);
    };

    const handleRenameCategory = (oldCat, newCatRaw) => {
        const newCat = newCatRaw.trim().toLowerCase().replace(/\s+/g, '_');
        if (!newCat || newCat === oldCat) return; 
        if (skillCategories.includes(newCat)) return alert('A category with this name already exists.');
        setSkillCategories(prev => prev.map(c => c === oldCat ? newCat : c));
        setCvData(prev => ({ ...prev, skills: prev.skills.map(s => (s.category || 'other') === oldCat ? { ...s, category: newCat } : s) }));
    };

    const removeSkillCategory = (cat) => {
        if(!window.confirm(`Delete category '${cat.replace(/_/g, ' ')}' and all its skills?`)) return;
        setSkillCategories(prev => prev.filter(c => c !== cat));
        setCvData(prev => ({ ...prev, skills: prev.skills.filter(s => s.category !== cat) }));
    };

    const moveSkillCategory = (index, direction) => {
        const newCats = [...skillCategories];
        const [moved] = newCats.splice(index, 1);
        newCats.splice(index + direction, 0, moved);
        setSkillCategories(newCats);
    };

    const toggleGhostCat = (cat) => setExpandedGhostCats(p => ({ ...p, [cat]: !p[cat] }));

    const restoreGhostSkillDirectly = (index, skill) => {
        setCvData(prev => {
            const newInactive = [...prev.inactive_skills];
            newInactive.splice(index, 1);
            return {
                ...prev,
                inactive_skills: newInactive,
                skills: [skill, ...prev.skills]
            };
        });
    };

    const closeSkillModal = () => {
        setCvData(prev => {
            if(!prev) return prev;
            return {
                ...prev,
                // Instantly remove any skills that are fully blank to prevent ghost categories
                skills: (prev.skills || []).filter(s => s.name && s.name.trim() !== ''),
                inactive_skills: (prev.inactive_skills || []).filter(s => s.name && s.name.trim() !== '')
            };
        });
        setIsSkillModalOpen(false);
    };


    // API ACTIONS
    const handleSave = async () => {
        setIsSaving(true);
        try { 
            const getFullList = (listName) => isDerived ? [...(cvData[listName]||[]), ...(cvData[`inactive_${listName}`]||[])] : cvData[listName];
            const getSelectedIds = (listName) => isDerived ? (cvData[listName]||[]).map(i => i.id) : undefined;

            const payload = { 
                ...cvData, 
                just_imported: false, 
                section_order: sectionOrder, 
                section_titles: sectionTitles, 
                skill_category_order: skillCategories,
                experiences: getFullList('experiences'), selected_experience_ids: getSelectedIds('experiences'),
                education: getFullList('education'), selected_education_ids: getSelectedIds('education'),
                projects: getFullList('projects'), selected_project_ids: getSelectedIds('projects'),
                hobbies: getFullList('hobbies'), selected_hobby_ids: getSelectedIds('hobbies'),
                skills: getFullList('skills').filter(s => s.name && s.name.trim() !== ''), // safety sweep
                selected_skill_ids: getSelectedIds('skills'),
                manual_mappings_to_add: pendingManualMappings
            };

            // Update is_title_in_name based on presence of title
            if (payload.title && payload.title.trim() !== '') {
                payload.is_title_in_name = true;
            } else {
                payload.is_title_in_name = false;
            }

            if (applicationId || isDerived) await updateDerivedCV(cvData.id, payload); 
            else await updateBaseCV(cvData.id, payload);
            
            setCvData(prev => ({ ...prev, just_imported: false }));
            setPendingManualMappings([]); 
        } 
        catch (e) { console.error(e); alert("Save failed"); } 
        finally { setIsSaving(false); }
    };

    const handleSync = async () => {
        if(!window.confirm("Resync will overwrite your manual edits. Continue?")) return;
        setIsSyncing(true);
        try {
            const res = await getOrCreateDerivedCV(applicationId, { force_refresh: true });
            setCvData(hydrateCV(res));
        } catch (e) { alert("Sync failed"); } 
        finally { setIsSyncing(false); }
    };

    const handleSyncToBase = async () => {
        if(!window.confirm("This will overwrite bullets in your Master CV with these changes. Continue?")) return;
        setIsSyncing(true);
        try {
            await apiClient.post(`/cv/${cvData.id}/sync-to-base`);
            alert("Successfully synced changes back to Base CV!");
        } catch (e) { console.error(e); alert("Failed to sync to Base CV."); } 
        finally { setIsSyncing(false); }
    };

    const handleExport = async (format) => {
        setIsExporting(true);
        try {
            await handleSave();
            const payload = { section_order: sectionOrder, section_titles: sectionTitles, file_format: format, font_size: fontSize, font_family: fontFamily };
            const response = await exportCV(cvData.id, payload);
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            const cd = response.headers['content-disposition'];
            let filename = `cv_export.${format === 'tex' ? 'tex' : format === 'zip' ? 'zip' : format}`;
            if (cd) {
                const match = cd.match(/filename="?([^"]+)"?/);
                if (match && match.length === 2) filename = match[1];
            }
            link.setAttribute('download', filename);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
        } catch (error) { alert("Failed to export CV."); } 
        finally { setIsExporting(false); }
    };

    if (isLoading) return <div className="vh-100 d-flex align-items-center justify-content-center"><Loader2 className="animate-spin"/></div>;
    if (!cvData) return <div>Error loading.</div>;

    // SHARED BULLET RENDERER
    const renderBullets = (item, listName, idx) => (
        <ul className="mb-0 cv-bullet">
            {item.description && (
                <li className="position-relative group" style={{ listStyleType: 'circle', marginBottom: '4pt' }}>
                    <div className="d-flex align-items-start gap-1 w-100">
                        <Editable value={item.description} onChange={v => updateItem(listName, idx, 'description', v)} multiline resizeTrigger={fontSize} placeholder="Description..."/>
                        <div className="d-flex gap-1 opacity-0 group-hover-opacity flex-shrink-0">
                            <button onClick={() => updateItem(listName, idx, 'description', '')} className="btn btn-link p-0 text-danger" title="Remove Description"><Trash2 size={12}/></button>
                        </div>
                    </div>
                </li>
            )}

            {item.achievements?.map((ach, aIdx) => (
                <li key={aIdx} className="position-relative group" style={{ marginBottom: '4pt' }}>
                    <div className="d-flex align-items-start gap-1 w-100">
                        <Editable value={ach.text} onChange={v => updateAchievementText(listName, idx, aIdx, v)} multiline resizeTrigger={fontSize} placeholder="Achievement detail"/>
                        <div className="d-flex gap-1 opacity-0 group-hover-opacity flex-shrink-0">
                            <button onClick={() => moveAchievement(listName, idx, aIdx, -1)} className="btn btn-link p-0 text-muted" disabled={aIdx===0}><MoveUp size={12}/></button>
                            <button onClick={() => moveAchievement(listName, idx, aIdx, 1)} className="btn btn-link p-0 text-muted" disabled={aIdx===item.achievements.length-1}><MoveDown size={12}/></button>
                            <button onClick={() => deleteAchievement(listName, idx, aIdx)} className="btn btn-link p-0 text-danger"><Trash2 size={12}/></button>
                        </div>
                    </div>
                </li>
            ))}

            <li className="list-unstyled opacity-50 hover-opacity-100 d-flex gap-3" style={{ fontSize: '0.85em', marginTop: '4pt' }}>
                {(!item.description || item.description.trim() === '') && (
                    <span className="cursor-pointer text-primary d-flex align-items-center" onClick={() => updateItem(listName, idx, 'description', 'New description')}>
                        <Plus size={12} className="me-1"/> Add description
                    </span>
                )}
                <span className="cursor-pointer text-primary d-flex align-items-center" onClick={() => addAchievement(listName, idx)}>
                    <Plus size={12} className="me-1"/> Add achievement
                </span>
            </li>
        </ul>
    );

    const renderGhostChildren = (listKey) => {
        const inactiveKey = `inactive_${listKey}`;
        const ghosts = cvData[inactiveKey];
        if (!isDerived || !ghosts || ghosts.length === 0) return null;

        return (
            <div className="mt-4 pt-3 border-top" style={{ borderStyle: 'dashed' }}>
                <div className="d-flex align-items-center justify-content-between mb-3">
                    <span className="text-muted small fw-bold text-uppercase d-flex align-items-center gap-2">
                        <UploadCloud size={14}/> {ghosts.length} Unused Items from Base CV
                    </span>
                </div>
                <div className="opacity-75">
                    {ghosts.map((item, idx) => (
                        <div key={item.id} className="d-flex justify-content-between align-items-center mb-2 p-2 border rounded bg-light hover-bg-white transition-colors group">
                            <div className="flex-grow-1 text-truncate pe-3">
                                <strong className="d-block text-dark text-truncate" style={{ fontSize: '0.9rem' }}>
                                    {item.title || item.degree || item.name}
                                </strong>
                                <span className="text-muted small text-truncate d-block">
                                    {item.company || item.institution || item.description || 'No additional details'}
                                </span>
                            </div>
                            <button 
                                className="btn btn-sm btn-white border shadow-sm text-primary d-flex align-items-center gap-1 flex-shrink-0 opacity-0 group-hover-opacity"
                                onClick={() => handleRestoreClick(listKey, idx, item)}
                            >
                                <Plus size={14}/> Restore
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        );
    };

    const renderCategoryDivider = (index) => {
        if (addingCategoryAt === index) {
            return (
                <div className="my-3 p-3 bg-white border-0 rounded-3 shadow-sm d-flex gap-2 align-items-center animate-fade-in">
                    <input 
                        autoFocus type="text" className="form-control bg-light border-0" placeholder="Category name..."
                        value={newCategoryName} onChange={e => setNewCategoryName(e.target.value)}
                        onKeyDown={e => { if (e.key === 'Enter') handleInsertCategory(index); else if (e.key === 'Escape') setAddingCategoryAt(null); }}
                    />
                    <button className="btn btn-primary rounded-pill px-3" onClick={() => handleInsertCategory(index)}>Add</button>
                    <button className="btn btn-light rounded-pill px-3" onClick={() => setAddingCategoryAt(null)}>Cancel</button>
                </div>
            );
        }
        return (
            <div className="position-relative group cursor-pointer" style={{ height: '32px', margin: '-8px 0' }} onClick={() => { setAddingCategoryAt(index); setNewCategoryName(''); }}>
                <div className="position-absolute top-50 w-100 border-top border-primary opacity-0 group-hover-opacity transition-opacity" style={{ zIndex: 1, borderWidth: '2px !important' }}></div>
                <div className="position-absolute top-50 start-50 translate-middle opacity-0 group-hover-opacity transition-opacity" style={{ zIndex: 2 }}>
                    <span className="badge rounded-pill bg-primary shadow-sm d-flex align-items-center justify-content-center px-3 py-1 gap-1 border border-white border-2" style={{fontSize: '0.75rem'}}>
                        <Plus size={12}/> New Category
                    </span>
                </div>
            </div>
        );
    };

    const renderSection = (key, idx) => {
        switch(key) {
            case 'summary':
                return cvData.summary ? (
                    <div style={{ marginBottom: '16pt' }}>
                        <div style={sectionHeaderStyle}><Editable value={sectionTitles[key]} onChange={v => setSectionTitles(p => ({...p, [key]: v}))} /></div>
                        <Editable value={cvData.summary} onChange={v => updateRoot('summary', v)} multiline resizeTrigger={fontSize}/>
                    </div>
                ) : null;

            case 'education':
                return (cvData.education?.length > 0 || cvData.inactive_education?.length > 0) ? (
                    <div style={{ marginBottom: '16pt' }}>
                        <div style={sectionHeaderStyle}><Editable value={sectionTitles[key]} onChange={v => setSectionTitles(p => ({...p, [key]: v}))} /></div>
                        {cvData.education.map((edu, idx) => (
                            <ItemWrapper key={edu.id} index={idx} total={cvData.education.length} onMove={(i,d) => moveItem('education',i,d)} onDelete={(i) => deleteItem('education',i)}>
                                <div className="d-flex justify-content-between align-items-baseline">
                                    <div className="d-flex gap-2 w-100">
                                        <EditableDegreeField 
                                            degree={edu.degree} field={edu.field} 
                                            onDegreeChange={v => updateItem('education', idx, 'degree', v)} 
                                            onFieldChange={v => updateItem('education', idx, 'field', v)} 
                                        />
                                    </div>
                                    <div className="d-flex gap-1 ms-3 text-nowrap">
                                        <EditableDate value={edu.start_date} onChange={v => updateItem('education', idx, 'start_date', v)} placeholder="Start"/>
                                        <span className="opacity-50">-</span>
                                        <EditableDate value={edu.end_date} onChange={v => updateItem('education', idx, 'end_date', v)} placeholder="End"/>
                                    </div>
                                </div>
                                <div style={{ marginBottom: '4pt' }}><Editable value={edu.institution} onChange={v => updateItem('education', idx, 'institution', v)} placeholder="Institution" italic/></div>
                                {renderBullets(edu, 'education', idx)}
                            </ItemWrapper>
                        ))}
                        {renderGhostChildren('education')}
                    </div>
                ) : null;

            case 'skills':
                return (cvData.skills?.length > 0 || cvData.inactive_skills?.length > 0) ? (
                    <div className="group" style={{ marginBottom: '16pt' }}>
                        <div style={sectionHeaderStyle} className="position-relative">
                            <Editable value={sectionTitles[key]} onChange={v => setSectionTitles(p => ({...p, [key]: v}))} />
                            <button className="btn btn-link p-1 text-muted opacity-0 group-hover-opacity position-absolute bg-white" onClick={() => setIsSkillModalOpen(true)} style={{ right: 0, bottom: '2px' }} title="Manage Skills">
                                <Settings size={16}/>
                            </button>
                        </div>
                        <div className="cursor-pointer editable-hover p-1 rounded transition-colors" onClick={() => setIsSkillModalOpen(true)} title="Click to manage skills">
                            {skillCategories.map((cat) => {
                                const skills = cvData.skills.filter(s => (s.category || 'other') === cat && s.name && s.name.trim() !== '');
                                if(!skills.length) return null;
                                return (
                                    <div key={cat} className="d-flex" style={{lineHeight: '1.2', marginBottom: '4pt'}}>
                                        <strong className="text-capitalize me-2" style={{minWidth: '130px'}}>{cat.replace(/_/g, ' ')}:</strong>
                                        <span>{skills.map(s => s.name).join(", ")}</span>
                                    </div>
                                )
                            })}
                        </div>
                    </div>
                ) : null;

            case 'projects':
                return (cvData.projects?.length > 0 || cvData.inactive_projects?.length > 0) ? (
                    <div style={{ marginBottom: '16pt' }}>
                        <div style={sectionHeaderStyle}><Editable value={sectionTitles[key]} onChange={v => setSectionTitles(p => ({...p, [key]: v}))} /></div>
                        {cvData.projects.map((proj, idx) => (
                            <ItemWrapper key={proj.id} index={idx} total={cvData.projects.length} onMove={(i,d) => moveItem('projects',i,d)} onDelete={(i) => deleteItem('projects',i)}>
                                <div className="d-flex justify-content-between align-items-baseline" style={{ marginBottom: '4pt' }}>
                                    <Editable value={proj.title} onChange={v => updateItem('projects', idx, 'title', v)} bold placeholder="Project Title"/>
                                    <div className="d-flex gap-1 ms-3 text-nowrap">
                                        <EditableDate value={proj.start_date} onChange={v => updateItem('projects', idx, 'start_date', v)} placeholder="Start"/>
                                        <span className="opacity-50">-</span>
                                        <EditableDate value={proj.end_date} onChange={v => updateItem('projects', idx, 'end_date', v)} placeholder="End"/>
                                    </div>
                                </div>
                                {renderBullets(proj, 'projects', idx)}
                            </ItemWrapper>
                        ))}
                        {renderGhostChildren('projects')}
                    </div>
                ) : null;

            case 'experiences':
                return (cvData.experiences?.length > 0 || cvData.inactive_experiences?.length > 0) ? (
                    <div style={{ marginBottom: '16pt' }}>
                        <div style={sectionHeaderStyle}><Editable value={sectionTitles[key]} onChange={v => setSectionTitles(p => ({...p, [key]: v}))} /></div>
                        {cvData.experiences.map((exp, idx) => (
                            <ItemWrapper key={exp.id} index={idx} total={cvData.experiences.length} onMove={(i,d) => moveItem('experiences',i,d)} onDelete={(i) => deleteItem('experiences',i)}>
                                <div className="d-flex justify-content-between align-items-baseline">
                                    <Editable value={exp.title} onChange={v => updateItem('experiences', idx, 'title', v)} bold placeholder="Job Title"/>
                                    <div className="d-flex gap-1 ms-3 text-nowrap">
                                        <EditableDate value={exp.start_date} onChange={v => updateItem('experiences', idx, 'start_date', v)} placeholder="Start"/>
                                        <span className="opacity-50">-</span>
                                        <EditableDate value={exp.end_date} onChange={v => updateItem('experiences', idx, 'end_date', v)} placeholder="End"/>
                                    </div>
                                </div>
                                <div className="d-flex justify-content-between align-items-baseline" style={{ marginBottom: '4pt' }}>
                                    <Editable value={exp.company} onChange={v => updateItem('experiences', idx, 'company', v)} italic placeholder="Company"/>
                                    <Editable value={exp.location} onChange={v => updateItem('experiences', idx, 'location', v)} italic align="right" placeholder="Location"/>
                                </div>
                                {renderBullets(exp, 'experiences', idx)}
                            </ItemWrapper>
                        ))}
                        {renderGhostChildren('experiences')}
                    </div>
                ) : null;

            case 'hobbies':
                return (cvData.hobbies?.length > 0 || cvData.inactive_hobbies?.length > 0) ? (
                    <div style={{ marginBottom: '16pt' }}>
                        <div style={sectionHeaderStyle}><Editable value={sectionTitles[key]} onChange={v => setSectionTitles(p => ({...p, [key]: v}))} /></div>
                        <ul className="mb-0 cv-bullet">
                            {cvData.hobbies.map((hobby, idx) => (
                                <ItemWrapper key={hobby.id} index={idx} total={cvData.hobbies.length} onMove={(i,d) => moveItem('hobbies',i,d)} onDelete={(i) => deleteItem('hobbies',i)}>
                                    <div className="d-flex flex-column w-100">
                                        <Editable value={hobby.name} onChange={v => updateItem('hobbies', idx, 'name', v)} bold placeholder="Hobby Name"/>
                                        <div style={{ marginTop: '4pt' }}>{renderBullets(hobby, 'hobbies', idx)}</div>
                                    </div>
                                </ItemWrapper>
                            ))}
                        </ul>
                        {renderGhostChildren('hobbies')}
                    </div>
                ) : null;
            
            default: return null;
        }
    };

    const getTypeIcon = (listName) => {
        if (listName === 'experiences') return <Briefcase size={20} className="text-white"/>;
        if (listName === 'education') return <GraduationCap size={20} className="text-white"/>;
        if (listName === 'projects') return <FolderGit2 size={20} className="text-white"/>;
        if (listName === 'skills') return <Wrench size={20} className="text-white"/>;
        return <Heart size={20} className="text-white"/>;
    };

    return (
        <>
            <div className="container-fluid bg-light min-vh-100 pb-5">
                <style>{`
                    .editable-hover:hover { background-color: rgba(0,0,0,0.05); cursor: text; border-radius: 2px; }
                    .group:hover > .group-hover-opacity { opacity: 1 !important; }
                    .section-hover:hover .group-hover-opacity { opacity: 1 !important; }
                    .cv-bullet { list-style-type: disc; margin-left: 1.5em; padding-left: 0.2em; }
                    .hover-bg-white:hover { background-color: white !important; }
                    .hover-bg-light:hover { background-color: #f8f9fa !important; }
                `}</style>

                {/* HEADER CONTROLS */}
                <div className="sticky-top bg-white border-bottom shadow-sm px-4 py-2 mb-4 d-flex justify-content-between align-items-center z-3 flex-wrap gap-2">
                    <div className="d-flex align-items-center gap-3">
                        <button onClick={() => navigate(-1)} className="btn btn-link text-dark p-0"><ArrowLeft size={20}/></button>
                        <div>
                            <h5 className="fw-bold mb-0">{isDerived ? 'Tailored CV Editor' : 'Quick CV Editor'}</h5>
                            <small className="text-muted">{cvData?.name}</small>
                        </div>
                    </div>
                    
                    <div className="d-flex gap-2 align-items-center flex-wrap">
                        <div className="d-flex gap-1 me-2 align-items-center bg-light border rounded px-1">
                            <select 
                                className="form-select form-select-sm border-0 bg-transparent shadow-none fw-medium text-secondary" 
                                style={{width: 'auto'}}
                                value={fontFamily} onChange={(e) => setFontFamily(e.target.value)}
                            >
                                <option value="Arial">Arial</option>
                                <option value="Helvetica">Helvetica</option>
                                <option value="Times New Roman">Times</option>
                                <option value="Calibri">Calibri</option>
                            </select>
                            <div className="d-flex align-items-center border-start ps-1">
                                <button className="btn btn-sm btn-link text-muted p-1" onClick={() => changeFontSize(-1)}><Type size={14} />-</button>
                                <span className="fw-bold text-dark px-1 small" style={{minWidth: '2.5rem', textAlign: 'center'}}>{fontSize}pt</span>
                                <button className="btn btn-sm btn-link text-dark p-1" onClick={() => changeFontSize(1)}><Type size={14} />+</button>
                            </div>
                        </div>

                        {isDerived && (
                            <button className="btn btn-sm btn-outline-info d-flex gap-1 align-items-center" onClick={handleSyncToBase} disabled={isSyncing}>
                                {isSyncing ? <Loader2 size={14} className="animate-spin"/> : <CheckCircle size={14}/>} Sync to Base
                            </button>
                        )}
                        {applicationId && (
                            <button className="btn btn-sm btn-outline-secondary d-flex gap-1 align-items-center" onClick={handleSync} disabled={isSyncing}>
                                <RefreshCw size={14} className={isSyncing ? "animate-spin":""}/> Reset
                            </button>
                        )}
                        <button className="btn btn-sm btn-primary d-flex gap-1 align-items-center me-2 shadow-sm" onClick={handleSave} disabled={isSaving}>
                            {isSaving ? <Loader2 size={14} className="animate-spin"/> : <Save size={14}/>} Save Changes
                        </button>

                        <div className="dropdown shadow-sm">
                            <button className="btn btn-sm btn-dark d-flex align-items-center gap-2 dropdown-toggle" type="button" data-bs-toggle="dropdown" aria-expanded="false" disabled={isExporting}>
                                {isExporting ? <Loader2 size={14} className="animate-spin"/> : <Download size={14}/>} Download As...
                            </button>
                            <ul className="dropdown-menu dropdown-menu-end shadow border-0">
                                <li><button className="dropdown-item d-flex align-items-center gap-2 py-2" onClick={() => handleExport('pdf')}><FileText size={16} className="text-danger"/> PDF Document (.pdf)</button></li>
                                <li><button className="dropdown-item d-flex align-items-center gap-2 py-2" onClick={() => handleExport('docx')}><FileTypeIcon size={16} className="text-primary"/> Word Document (.docx)</button></li>
                                <li><button className="dropdown-item d-flex align-items-center gap-2 py-2" onClick={() => handleExport('tex')}><FileCode size={16} className="text-success"/> LaTeX Source (.tex)</button></li>
                                <li><hr className="dropdown-divider" /></li>
                                <li><button className="dropdown-item d-flex align-items-center gap-2 py-2" onClick={() => handleExport('zip')}><Archive size={16} className="text-warning"/> ZIP Bundle (All formats)</button></li>
                            </ul>
                        </div>
                    </div>
                </div>

                {/* Post-Import Banner */}
                {cvData && cvData.just_imported && (
                    <div className="container mb-3">
                        <div className="alert alert-warning d-flex align-items-center gap-2 py-2">
                            <UploadCloud size={16}/> <strong>Import Successful!</strong> Please review the formatting and wording before saving.
                        </div>
                    </div>
                )}

                {/* DOCUMENT */}
                {cvData && (
                    <div style={paperStyle}>
                        
                        {/* Header Info (Contact Area) */}
                        <div className="text-center position-relative group" style={{ marginBottom: '24pt' }}>
                            <div className="d-flex justify-content-center align-items-baseline w-100 group" style={{ marginBottom: '4pt' }}>
                                
                                {/* LEFT ZONE: Title Picker */}
                                <div className="text-end d-flex justify-content-end" style={{ flex: '1 1 25%', paddingRight: '12px' }}>
                                    {cvData.is_title_in_name ? (
                                        <select 
                                            className="form-select border-0 shadow-none fw-bold p-0 text-end bg-transparent cursor-pointer editable-hover"
                                            style={{ fontSize: '24pt', width: 'auto', backgroundPosition: 'right center', paddingRight: '1.5rem' }}
                                            value={cvData.title || ""}
                                            onChange={(e) => {
                                                if (e.target.value === "REMOVE") {
                                                    updateRoot('is_title_in_name', false);
                                                    updateRoot('title', '');
                                                } else {
                                                    updateRoot('title', e.target.value);
                                                }
                                            }}
                                        >
                                            <option value="" disabled>Title...</option>
                                            <option value="Mr.">Mr.</option>
                                            <option value="Ms.">Ms.</option>
                                            <option value="Mrs.">Mrs.</option>
                                            <option value="Dr.">Dr.</option>
                                            <option value="Prof.">Prof.</option>
                                            <option value="Mx.">Mx.</option>
                                            <option value="REMOVE" className="text-danger fw-normal" style={{fontSize: '12pt'}}>Remove Title</option>
                                        </select>
                                    ) : (
                                        <span 
                                            className="opacity-0 group-hover-opacity transition-opacity text-muted fst-italic cursor-pointer d-flex align-items-center gap-1"
                                            style={{ fontSize: '11pt' }}
                                            onClick={() => updateRoot('is_title_in_name', true)}
                                            title="Add Title"
                                        >
                                            <Plus size={14}/> Title
                                        </span>
                                    )}
                                </div>

                                {/* CENTER ZONE: Perfectly Centered Names */}
                                <div className="d-flex justify-content-center gap-2" style={{ flex: '0 0 50%' }}>
                                    <div className="flex-grow-1 text-end">
                                        <Editable value={cvData.first_name} onChange={v => updateRoot('first_name', v)} style={{fontSize: '24pt', textAlign: 'right'}} bold placeholder="First Name"/>
                                    </div>
                                    <div className="flex-grow-1 text-start">
                                        <Editable value={cvData.last_name} onChange={v => updateRoot('last_name', v)} style={{fontSize: '24pt', textAlign: 'left'}} bold placeholder="Last Name"/>
                                    </div>
                                </div>

                                {/* RIGHT ZONE: Invisible Balancer */}
                                <div style={{ flex: '1 1 25%' }}></div>
                            </div>
                            
                            <div 
                                className="d-inline-flex justify-content-center flex-wrap position-relative cursor-pointer editable-hover p-1 rounded transition-colors" 
                                style={{fontSize: '11pt', color: '#555'}}
                                onClick={openContactPanel}
                                title="Click to manage contact details"
                            >
                                <button className="btn btn-link p-1 text-muted opacity-0 group-hover-opacity position-absolute bg-white shadow-sm rounded-circle border" style={{ right: '-35px', top: '-5px' }} >
                                    <Settings size={14}/>
                                </button>

                                {(!cvData.contact_info || Object.keys(cvData.contact_info).length === 0) && (
                                    <span className="text-muted fst-italic">Click to add contact details (email, phone, etc)</span>
                                )}

                                {Object.entries(cvData.contact_info || {}).map(([key, value], idx, arr) => (
                                    <React.Fragment key={key}>
                                        <span>{value}</span>
                                        {idx < arr.length - 1 && <span className="mx-2">|</span>}
                                    </React.Fragment>
                                ))}
                            </div>
                        </div>

                        {/* DYNAMIC SECTIONS */}
                        {sectionOrder.map((sectionKey, idx) => (
                            <SectionWrapper key={sectionKey} sectionKey={sectionKey} index={idx} total={sectionOrder.length} onMove={moveSection}>
                                {renderSection(sectionKey, idx)}
                            </SectionWrapper>
                        ))}
                    </div>
                )}
            </div>

            {/* INTENT RESTORE MODAL - CONTACT DETAILS */}
            {cvData && isContactModalOpen && (
                <>
                    <div className="modal-backdrop fade show" style={{zIndex: 1040}} onClick={saveContactPanel}></div>
                    <div className="bg-white shadow-lg d-flex flex-column border-start" style={{ position: 'fixed', top: 0, right: 0, bottom: 0, height: '100vh', width: '100%', maxWidth: '400px', zIndex: 1050 }}>
                        <div className="px-4 pt-4 pb-3 d-flex justify-content-between align-items-start flex-shrink-0 bg-white">
                            <div>
                                <h5 className="fw-bold mb-0 text-dark d-flex align-items-center gap-2">
                                    <div className="bg-primary bg-opacity-10 text-primary rounded p-1"><Settings size={18}/></div>
                                    Contact Details
                                </h5>
                            </div>
                            <button type="button" className="btn-close shadow-none mt-1" onClick={saveContactPanel}></button>
                        </div>
                        
                        <div className="px-4 py-4 flex-grow-1 overflow-y-auto bg-light d-flex flex-column gap-3">
                            {contactFormData.map((item, idx) => (
                                <div key={item.id} className="d-flex align-items-center gap-2 p-3 bg-white rounded-3 shadow-sm border-0 group transition-colors hover-bg-light">
                                    <div className="d-flex align-items-center justify-content-center text-primary bg-primary bg-opacity-10 rounded p-2 flex-shrink-0">
                                        {getContactIcon(item.key, 18)}
                                    </div>
                                    <div className="border-end pe-3 flex-shrink-0" style={{width: '110px'}}>
                                        <input 
                                            type="text" 
                                            className="form-control form-control-sm border-0 shadow-none fw-bold text-secondary p-0 bg-transparent" 
                                            style={{fontSize: '0.8rem', textTransform: 'uppercase'}}
                                            value={item.key} 
                                            onChange={(e) => updateContactFormField(idx, 'key', e.target.value)} 
                                            placeholder="Label"
                                        />
                                    </div>
                                    <div className="flex-grow-1 ps-2">
                                        <input 
                                            type="text" 
                                            className="form-control form-control-sm border-0 shadow-none p-0 bg-transparent text-dark" 
                                            value={item.value} 
                                            onChange={(e) => updateContactFormField(idx, 'value', e.target.value)} 
                                            placeholder="Value (e.g. jdoe@mail.com)"
                                        />
                                    </div>
                                    <button className="btn btn-link text-danger p-1 opacity-0 group-hover-opacity transition-opacity" onClick={() => setContactFormData(prev => prev.filter((_, i) => i !== idx))} title="Remove">
                                        <Trash2 size={16}/>
                                    </button>
                                </div>
                            ))}
                            
                            <button className="btn btn-white w-100 d-flex align-items-center justify-content-center gap-2 border border-dashed rounded-3 text-muted py-3 hover-bg-light transition-colors" onClick={() => setContactFormData(prev => [...prev, { id: Date.now(), key: '', value: '' }])}>
                                <Plus size={16}/> Add New Detail
                            </button>
                        </div>
                        <div className="border-top bg-white p-3 flex-shrink-0">
                            <button className="btn btn-primary w-100 fw-bold py-2 rounded-pill shadow-sm" onClick={saveContactPanel}>Save Changes</button>
                        </div>
                    </div>
                </>
            )}

            {/* INTENT RESTORE MODAL - MANAGE SKILLS */}
            {cvData && isSkillModalOpen && (
                <>
                    <div className="modal-backdrop fade show" style={{zIndex: 1040}} onClick={closeSkillModal}></div>
                    <div className="bg-white shadow-lg d-flex flex-column border-start" style={{ position: 'fixed', top: 0, right: 0, bottom: 0, height: '100vh', width: '100%', maxWidth: '450px', zIndex: 1050 }}>
                        <div className="px-4 pt-4 pb-3 d-flex justify-content-between align-items-start flex-shrink-0 bg-white">
                            <div>
                                <h5 className="fw-bold mb-0 text-dark d-flex align-items-center gap-2">
                                    <div className="bg-primary bg-opacity-10 text-primary rounded p-1"><Settings size={18}/></div>
                                    Manage Skills
                                </h5>
                            </div>
                            <button type="button" className="btn-close shadow-none mt-1" onClick={closeSkillModal}></button>
                        </div>
                        
                        <div className="px-4 py-4 flex-grow-1 overflow-y-auto bg-light">
                            {renderCategoryDivider(0)}
                            
                            {skillCategories.map((cat, cIdx) => {
                                const catSkills = cvData.skills.map((s, i) => ({...s, _origIdx: i})).filter(s => (s.category || 'other') === cat);
                                const ghostSkills = (cvData.inactive_skills || []).map((s, i) => ({...s, _origIdx: i})).filter(s => (s.category || 'other') === cat);
                                
                                const isGhostCategory = catSkills.filter(s => s.name && s.name.trim() !== '').length === 0 && ghostSkills.length > 0;
                                const isExpanded = expandedGhostCats[cat] || false;

                                return (
                                    <React.Fragment key={cat}>
                                        <div className={`bg-white rounded-3 shadow-sm border-0 mb-2 overflow-hidden transition-opacity ${isGhostCategory ? 'opacity-75' : ''}`}>
                                            
                                            {/* Category Header */}
                                            <div className="px-3 py-3 d-flex justify-content-between align-items-center border-bottom bg-light group">
                                                <input 
                                                    className="form-control form-control-sm border-0 bg-transparent text-uppercase text-dark fw-bold p-0 shadow-none m-0"
                                                    style={{ letterSpacing: '1px', fontSize: '0.8rem', width: '180px' }}
                                                    defaultValue={cat.replace(/_/g, ' ')}
                                                    onBlur={(e) => handleRenameCategory(cat, e.target.value)}
                                                    onKeyDown={(e) => e.key === 'Enter' && e.target.blur()}
                                                    title="Click to rename category"
                                                />
                                                <div className="d-flex gap-1 opacity-0 group-hover-opacity transition-opacity">
                                                    <button onClick={() => moveSkillCategory(cIdx, -1)} className="btn btn-sm btn-link p-1 text-muted" disabled={cIdx===0}><MoveUp size={14}/></button>
                                                    <button onClick={() => moveSkillCategory(cIdx, 1)} className="btn btn-sm btn-link p-1 text-muted" disabled={cIdx===skillCategories.length-1}><MoveDown size={14}/></button>
                                                    <button onClick={() => removeSkillCategory(cat)} className="btn btn-sm btn-link p-1 text-danger ms-1" title="Delete Category & Skills"><Trash2 size={14}/></button>
                                                </div>
                                            </div>

                                            {/* Skill List */}
                                            <div className="p-2 d-flex flex-column gap-1">
                                                
                                                {/* Active Skills */}
                                                {catSkills.map(skill => (
                                                    <div key={skill.id || skill._origIdx} className="d-flex align-items-center group py-2 px-2 rounded-3 hover-bg-light transition-colors">
                                                        <div className="flex-grow-1 pe-2">
                                                            <input type="text" className="form-control form-control-sm border-0 shadow-none fw-medium bg-transparent p-0 px-1 text-dark" value={skill.name} onChange={(e) => updateSkill(skill._origIdx, 'name', e.target.value)} placeholder="Skill name"/>
                                                        </div>
                                                        <div className="d-flex align-items-center gap-1 opacity-0 group-hover-opacity bg-white rounded pe-1 shadow-sm border transition-opacity">
                                                            <select className="form-select form-select-sm border-0 text-secondary bg-transparent cursor-pointer text-truncate py-1" style={{fontSize: '0.75rem', width: '110px'}} value={skill.category || 'other'} onChange={(e) => updateSkill(skill._origIdx, 'category', e.target.value)}>
                                                                {skillCategories.map(c => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
                                                                {!skillCategories.includes(skill.category) && skill.category && <option value={skill.category}>{skill.category.replace(/_/g, ' ')}</option>}
                                                            </select>
                                                            <button className="btn btn-sm btn-light text-danger rounded-circle p-1 d-flex justify-content-center align-items-center" onClick={() => deleteSkill(skill._origIdx)} title="Remove Skill"><Trash2 size={13}/></button>
                                                        </div>
                                                    </div>
                                                ))}

                                                <button className="btn btn-sm btn-link text-primary text-decoration-none text-start p-2 ms-1 small fw-medium d-flex align-items-center gap-1" onClick={() => addSkillToCategory(cat)}>
                                                    <Plus size={14}/> Add Skill
                                                </button>

                                                {/* Ghost Skills */}
                                                {isDerived && ghostSkills.length > 0 && (
                                                    <div className="mt-2 pt-2 border-top border-dashed">
                                                        <button 
                                                            className="btn btn-link text-muted text-decoration-none w-100 text-start d-flex justify-content-between align-items-center p-2 rounded-3 hover-bg-light transition-colors" 
                                                            onClick={() => toggleGhostCat(cat)}
                                                        >
                                                            <span className="small fw-medium d-flex align-items-center gap-2">
                                                                {isExpanded ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}
                                                                {ghostSkills.length} Unused {ghostSkills.length === 1 ? 'Skill' : 'Skills'}
                                                            </span>
                                                        </button>
                                                        {isExpanded && (
                                                            <div className="d-flex flex-column gap-1 mt-1 px-2 pb-2">
                                                                {ghostSkills.map(skill => (
                                                                    <div key={skill.id || skill._origIdx} className="d-flex align-items-center justify-content-between py-1 px-2 rounded border border-dashed bg-light group">
                                                                        <span className="text-muted small text-truncate fst-italic">{skill.name}</span>
                                                                        <button 
                                                                            className="btn btn-sm btn-white border shadow-sm text-primary py-0 px-2 small rounded-pill opacity-0 group-hover-opacity transition-opacity" 
                                                                            style={{fontSize: '0.75rem'}}
                                                                            onClick={() => restoreGhostSkillDirectly(skill._origIdx, skill)}
                                                                        >
                                                                            Restore
                                                                        </button>
                                                                    </div>
                                                                ))}
                                                            </div>
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        </div>
                                        {renderCategoryDivider(cIdx + 1)}
                                    </React.Fragment>
                                );
                            })}
                            
                            {/* RENDER UNCATEGORIZED SKILLS */}
                            {(() => {
                                const otherCatSkills = cvData.skills.map((s, i) => ({...s, _origIdx: i})).filter(s => !skillCategories.includes(s.category || 'other'));
                                const otherGhostSkills = (cvData.inactive_skills || []).map((s, i) => ({...s, _origIdx: i})).filter(s => !skillCategories.includes(s.category || 'other'));
                                
                                if (otherCatSkills.length === 0 && otherGhostSkills.length === 0) return null;

                                const isGhostCategory = otherCatSkills.filter(s => s.name && s.name.trim() !== '').length === 0 && otherGhostSkills.length > 0;
                                const isExpanded = expandedGhostCats['other'] || false;

                                return (
                                    <div className={`bg-white rounded-3 shadow-sm border-0 mb-2 overflow-hidden transition-opacity border-warning border border-2 ${isGhostCategory ? 'opacity-75' : ''}`}>
                                        <div className="px-3 py-3 d-flex justify-content-between align-items-center border-bottom bg-warning bg-opacity-10 group">
                                            <h6 className="text-uppercase text-dark fw-bold m-0" style={{letterSpacing: '1px', fontSize: '0.8rem'}}>Other (Uncategorized)</h6>
                                        </div>
                                        <div className="p-2 d-flex flex-column gap-1">
                                            
                                            {/* Active Skills */}
                                            {otherCatSkills.map(skill => (
                                                <div key={skill.id} className="d-flex align-items-center group py-2 px-2 rounded-3 hover-bg-light transition-colors">
                                                    <div className="flex-grow-1 pe-2">
                                                        <input type="text" className="form-control form-control-sm border-0 shadow-none fw-medium bg-transparent p-0 px-1 text-dark" value={skill.name} onChange={(e) => updateSkill(skill._origIdx, 'name', e.target.value)} placeholder="Skill name" />
                                                    </div>
                                                    <div className="d-flex align-items-center gap-1 opacity-0 group-hover-opacity bg-white rounded pe-1 shadow-sm border transition-opacity">
                                                        <select className="form-select form-select-sm border-0 text-secondary bg-transparent cursor-pointer text-truncate py-1" style={{fontSize: '0.75rem', width: '110px'}} value={skill.category || 'other'} onChange={(e) => updateSkill(skill._origIdx, 'category', e.target.value)}>
                                                            <option value={skill.category || 'other'}>Other</option>
                                                            {skillCategories.map(c => <option key={c} value={c}>{c.replace(/_/g, ' ')}</option>)}
                                                        </select>
                                                        <button className="btn btn-sm btn-light text-danger rounded-circle p-1 d-flex justify-content-center align-items-center" onClick={() => deleteSkill(skill._origIdx)} title="Remove Skill"><Trash2 size={13}/></button>
                                                    </div>
                                                </div>
                                            ))}

                                            <button className="btn btn-sm btn-link text-primary text-decoration-none text-start p-2 ms-1 small fw-medium d-flex align-items-center gap-1" onClick={() => addSkillToCategory('other')}>
                                                <Plus size={14}/> Add Skill
                                            </button>

                                            {/* Ghost Skills */}
                                            {isDerived && otherGhostSkills.length > 0 && (
                                                <div className="mt-2 pt-2 border-top border-dashed">
                                                    <button 
                                                        className="btn btn-link text-muted text-decoration-none w-100 text-start d-flex justify-content-between align-items-center p-2 rounded-3 hover-bg-light transition-colors" 
                                                        onClick={() => toggleGhostCat('other')}
                                                    >
                                                        <span className="small fw-medium d-flex align-items-center gap-2">
                                                            {isExpanded ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}
                                                            {otherGhostSkills.length} Unused {otherGhostSkills.length === 1 ? 'Skill' : 'Skills'}
                                                        </span>
                                                    </button>
                                                    {isExpanded && (
                                                        <div className="d-flex flex-column gap-1 mt-1 px-2 pb-2">
                                                            {otherGhostSkills.map(skill => (
                                                                <div key={skill.id || skill._origIdx} className="d-flex align-items-center justify-content-between py-1 px-2 rounded border border-dashed bg-light group">
                                                                    <span className="text-muted small text-truncate fst-italic">{skill.name}</span>
                                                                    <button 
                                                                        className="btn btn-sm btn-white border shadow-sm text-primary py-0 px-2 small rounded-pill opacity-0 group-hover-opacity transition-opacity" 
                                                                        style={{fontSize: '0.75rem'}}
                                                                        onClick={() => restoreGhostSkillDirectly(skill._origIdx, skill)}
                                                                    >
                                                                        Restore
                                                                    </button>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                );
                            })()}
                        </div>
                    </div>
                </>
            )}

            {/* INTENT RESTORE MODAL - MODERNIZED SIDE PANEL */}
            {restoreModalData && (
                <>
                    <div className="modal-backdrop fade show" style={{zIndex: 1060}} onClick={() => {setRestoreModalData(null); setMappingWizardStep('FEATURE'); setSelectedFeatureForMapping(null);}}></div>
                    <div className="bg-white shadow-lg d-flex flex-column border-start" style={{ position: 'fixed', top: 0, right: 0, bottom: 0, height: '100vh', width: '100%', maxWidth: '550px', zIndex: 1070 }}>
                        
                        {/* Dynamic Header */}
                        <div className="px-4 pt-4 pb-3 d-flex justify-content-between align-items-start flex-shrink-0 bg-white">
                            <div>
                                <div className="d-flex align-items-center gap-2 mb-2">
                                    <span className="badge bg-primary bg-opacity-10 text-primary rounded-pill px-2 py-1 d-flex align-items-center gap-1" style={{fontSize: '0.7rem'}}>
                                        <LinkIcon size={12}/> AI RoleCase Engine
                                    </span>
                                </div>
                                <h5 className="fw-bold mb-0 text-dark">
                                    {mappingWizardStep === 'FEATURE' && "Select matching requirement"}
                                    {mappingWizardStep === 'SEGMENT' && "Which part matches best?"}
                                    {mappingWizardStep === 'FALLBACK_INTENT' && "Categorize as an extra"}
                                </h5>
                            </div>
                            <button type="button" className="btn-close shadow-none mt-1" onClick={() => { setRestoreModalData(null); setSelectedFeatureForMapping(null); setMappingWizardStep('FEATURE'); }}></button>
                        </div>

                        {/* Sleek Context Item Card */}
                        <div className="mx-4 mb-3 p-2 bg-light rounded-3 d-flex align-items-center gap-3 border flex-shrink-0">
                            <div className="bg-primary text-white rounded d-flex align-items-center justify-content-center shadow-sm" style={{width: '40px', height: '40px'}}>
                                {getTypeIcon(restoreModalData.listName)}
                            </div>
                            <div className="flex-grow-1 overflow-hidden pe-2">
                                <div className="text-uppercase text-muted fw-bold mb-1" style={{ fontSize: '0.65rem', letterSpacing: '1px' }}>Restoring {restoreModalData.listName.replace(/_/g, ' ')}</div>
                                <div className="fw-bold text-dark text-truncate lh-sm" style={{ fontSize: '0.95rem' }}>{restoreModalData.item.title || restoreModalData.item.name || restoreModalData.item.degree || 'Untitled Item'}</div>
                                {(restoreModalData.item.company || restoreModalData.item.institution) && (
                                    <div className="text-muted small text-truncate mt-1">{restoreModalData.item.company || restoreModalData.item.institution}</div>
                                )}
                            </div>
                        </div>

                        {/* Scrollable Body Container */}
                        <div className="px-4 pb-4 flex-grow-1 d-flex flex-column bg-white" style={{ overflowY: 'auto', minHeight: 0 }}>
                            
                            {/* MAPPING MODE VIEW (Step 1) */}
                            {mappingWizardStep === 'FEATURE' && (
                                <div className="d-flex flex-column flex-grow-1 animate-fade-in">
                                    {/* Minimalist Search & Filter (Sticky) */}
                                    <div className="d-flex gap-2 mb-3 sticky-top bg-white pt-1 pb-2 z-1" style={{top: '-1px'}}>
                                        <div className="input-group input-group-sm flex-grow-1 border rounded-3 overflow-hidden focus-ring-primary">
                                            <span className="input-group-text bg-transparent border-0 text-muted ps-3"><Search size={14}/></span>
                                            <input 
                                                type="text" className="form-control border-0 shadow-none bg-transparent py-2" 
                                                placeholder="Search job description..." 
                                                value={mappingSearch} onChange={e => setMappingSearch(e.target.value)} 
                                            />
                                        </div>
                                        <select 
                                            className="form-select form-select-sm w-auto border rounded-3 shadow-none text-muted py-2" 
                                            value={mappingFilterType} onChange={e => setMappingFilterType(e.target.value)}
                                        >
                                            <option value="ALL">All Types</option>
                                            {[...new Set(jobFeatures.map(f => f.type))].sort().map(t => (
                                                <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>
                                            ))}
                                        </select>
                                    </div>

                                    {/* Clean Feature List */}
                                    <div className="d-flex flex-column gap-2 flex-grow-1">
                                        {jobFeatures.filter(f => (mappingFilterType === 'ALL' || f.type === mappingFilterType) && f.description.toLowerCase().includes(mappingSearch.toLowerCase())).map(feat => (
                                            <div 
                                                key={feat.id} 
                                                className="group p-3 rounded-3 border bg-white hover-bg-light cursor-pointer transition-colors d-flex align-items-start gap-3"
                                                onClick={() => selectFeatureForMapping(feat)}
                                            >
                                                <div className="flex-grow-1">
                                                    <span className="badge bg-light text-secondary border border-secondary border-opacity-25 mb-2" style={{fontSize: '0.65rem'}}>{feat.type.replace(/_/g, ' ')}</span>
                                                    <div className="small text-dark lh-sm" style={{textAlign: 'left'}}>{feat.description}</div>
                                                </div>
                                                <div className="opacity-0 group-hover-opacity text-primary d-flex align-items-center flex-shrink-0 mt-1 transition-opacity">
                                                    <ChevronRight size={18}/>
                                                </div>
                                            </div>
                                        ))}
                                        {jobFeatures.filter(f => (mappingFilterType === 'ALL' || f.type === mappingFilterType) && f.description.toLowerCase().includes(mappingSearch.toLowerCase())).length === 0 && (
                                            <div className="py-5 text-center text-muted small fst-italic">
                                                No matching job requirements found.
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* STEP 2: SELECT SEGMENT */}
                            {mappingWizardStep === 'SEGMENT' && (
                                <div className="d-flex flex-column flex-grow-1 pb-2 animate-fade-in">
                                    {/* Target Context */}
                                    <div className="mb-4 p-3 rounded-3 bg-primary bg-opacity-10 border border-primary border-opacity-25 relative">
                                        <div className="text-primary fw-bold small mb-2 d-flex align-items-center gap-1"><CheckCircle size={14}/> Targeted Requirement</div>
                                        <div className="small text-dark lh-sm">{selectedFeatureForMapping?.description}</div>
                                    </div>

                                    <div className="d-flex flex-column gap-2 flex-grow-1">
                                        {getSegmentsForItem(restoreModalData.item, restoreModalData.listName).map((segment, sIdx) => (
                                            <button key={sIdx} className="btn btn-white text-start p-3 rounded-3 border shadow-sm hover-bg-light transition-colors group d-flex align-items-start gap-3" onClick={() => finalizeRestore('MAPPED', segment)}>
                                                <div className="flex-grow-1">
                                                    <span className="badge bg-light text-muted border mb-2 px-2">{segment.label}</span>
                                                    <div className="small text-dark lh-sm">{segment.text}</div>
                                                </div>
                                                <div className="opacity-0 group-hover-opacity text-primary fw-bold small d-flex align-items-center gap-1 mt-1 transition-opacity">
                                                    Confirm <CheckCircle size={14}/>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* FALLBACK INTENT VIEW */}
                            {mappingWizardStep === 'FALLBACK_INTENT' && (
                                <div className="d-flex flex-column flex-grow-1 pb-2 animate-fade-in">
                                    <p className="text-muted small mb-4">If this doesn't map directly to a job requirement, how should our AI frame it to increase your overall profile strength?</p>
                                    <div className="d-flex flex-column gap-3 flex-grow-1">
                                        <button className="btn btn-white p-3 rounded-3 border text-start hover-bg-light shadow-sm d-flex align-items-start gap-3 transition-colors" onClick={() => finalizeRestore('VALUE_ADD')}>
                                            <div className="bg-primary bg-opacity-10 text-primary rounded p-2"><Plus size={18}/></div>
                                            <div>
                                                <strong className="d-block text-dark mb-1">Value Add / Bonus Skill</strong>
                                                <span className="small text-muted lh-sm d-block text-wrap">A professional attribute that goes above and beyond the job description.</span>
                                            </div>
                                        </button>
                                        <button className="btn btn-white p-3 rounded-3 border text-start hover-bg-light shadow-sm d-flex align-items-start gap-3 transition-colors" onClick={() => finalizeRestore('QUALIFICATION')}>
                                            <div className="bg-warning bg-opacity-10 text-warning rounded p-2"><Award size={18}/></div>
                                            <div>
                                                <strong className="d-block text-dark mb-1">Qualification / Certification</strong>
                                                <span className="small text-muted lh-sm d-block text-wrap">An official credential that validates your expertise and dedication.</span>
                                            </div>
                                        </button>
                                        <button className="btn btn-white p-3 rounded-3 border text-start hover-bg-light shadow-sm d-flex align-items-start gap-3 transition-colors" onClick={() => finalizeRestore('CULTURE_FIT')}>
                                            <div className="bg-info bg-opacity-10 text-info rounded p-2"><Globe size={18}/></div>
                                            <div>
                                                <strong className="d-block text-dark mb-1">Culture Fit / Personality</strong>
                                                <span className="small text-muted lh-sm d-block text-wrap">Demonstrates unique background, leadership traits, or team synergy.</span>
                                            </div>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Modernized Fixed Footer */}
                        <div className="border-top bg-white p-3 flex-shrink-0 d-flex justify-content-between align-items-center w-100" style={{ zIndex: 10 }}>
                            {mappingWizardStep === 'FEATURE' ? (
                                <button className="btn btn-light text-muted w-100 rounded-pill py-2 text-decoration-none fw-medium transition-colors border" onClick={() => setMappingWizardStep('FALLBACK_INTENT')}>
                                    Skip mapping & add as Extra Context
                                </button>
                            ) : (
                                <button className="btn btn-light text-muted w-100 rounded-pill py-2 text-decoration-none fw-medium transition-colors border d-flex justify-content-center align-items-center gap-2" onClick={() => setMappingWizardStep('FEATURE')}>
                                    <ArrowLeft size={16} /> Back to Job Requirements
                                </button>
                            )}
                        </div>
                    </div>
                </>
            )}

        </>
    );
};

export default QuickCVEditor;