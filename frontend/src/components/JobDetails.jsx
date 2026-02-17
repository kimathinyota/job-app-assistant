import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    MapPin, Banknote, Globe, Building2, 
    ArrowLeft, Trash2, Plus, X, Check,
    CheckCircle2, ListChecks, Zap, Star, Briefcase, GraduationCap, Heart, Award, HelpCircle, Calendar, Clock
} from 'lucide-react';
import { 
    fetchJobDetails, 
    updateJob, 
    deleteJob, 
    addJobFeature, 
    deleteJobFeature,
    updateJobFeature
} from '../api/applicationClient';

const JobDetails = () => {
    const { jobId } = useParams();
    const navigate = useNavigate();
    
    const [job, setJob] = useState(null);
    const [loading, setLoading] = useState(true);
    
    // -- Granular Editing State --
    const [editField, setEditField] = useState(null); 

    // Global Form State (for edits)
    const [formData, setFormData] = useState({});
    
    // Inline Feature Editing State
    const [editingFeatureId, setEditingFeatureId] = useState(null);
    const [editFeatureText, setEditFeatureText] = useState('');
    const [editFeatureType, setEditFeatureType] = useState('');

    useEffect(() => {
        loadJob();
    }, [jobId]);

    const loadJob = async () => {
        try {
            const res = await fetchJobDetails(jobId);
            setJob(res.data);
            setFormData(res.data); 
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    // --- GENERIC SAVE HANDLER ---
    const saveField = async (field, value) => {
        try {
            await updateJob(jobId, { [field]: value });
            setEditField(null);
            loadJob(); 
        } catch (err) { 
            console.error(err);
            alert(`Failed to save ${field}.`); 
        }
    };

    const handleKeyDown = (e, field) => {
        if (e.key === 'Enter') {
            if(field === 'description' || field === 'notes') {
                if(e.shiftKey) return; 
            }
            saveField(field);
        } else if (e.key === 'Escape') {
            setEditField(null);
            setFormData(job); 
        }
    };

    const handleDelete = async () => {
        if(window.confirm("Are you sure? This cannot be undone.")) {
            await deleteJob(jobId);
            navigate('/jobs');
        }
    };

    const handleStartApplication = () => {
        alert("To start an application, please use the 'Start' button on the Job Library card.");
        navigate('/jobs'); 
    };

    // --- FEATURE MANAGEMENT ---
    
    const handleAddFeature = async (text, type) => {
        if(!text.trim()) return;
        await addJobFeature(jobId, text, type);
        loadJob();
    };

    const handleRemoveFeature = async (featureId) => {
        if(!window.confirm("Remove this item?")) return;
        await deleteJobFeature(jobId, featureId);
        loadJob();
    };

    const startEditingFeature = (feature) => {
        setEditingFeatureId(feature.id);
        setEditFeatureText(feature.description);
        setEditFeatureType(feature.type);
    };

    const cancelEditFeature = () => {
        setEditingFeatureId(null);
        setEditFeatureText('');
        setEditFeatureType('');
    };

    const saveEditFeature = async () => {
        if (!editFeatureText.trim()) return;
        try {
            await updateJobFeature(jobId, editingFeatureId, { 
                description: editFeatureText, 
                type: editFeatureType 
            });
            await loadJob(); 
            cancelEditFeature();
        } catch (err) {
            alert("Failed to update feature.");
        }
    };

    // --- CONFIG ---
    const TYPE_CONFIG = {
        responsibility: { label: 'Task', icon: ListChecks, color: 'text-primary', badge: 'bg-primary' },
        requirement:    { label: 'Must Have', icon: CheckCircle2, color: 'text-danger', badge: 'bg-danger' },
        experience:     { label: 'Experience', icon: Briefcase, color: 'text-danger', badge: 'bg-danger' },
        qualification:  { label: 'Qualification', icon: GraduationCap, color: 'text-danger', badge: 'bg-danger' },
        hard_skill:     { label: 'Hard Skill', icon: Zap, color: 'text-dark', isTag: true, badge: 'bg-dark' },
        soft_skill:     { label: 'Soft Skill', icon: Zap, color: 'text-info', isTag: true, badge: 'bg-info' },
        // Ensure keys match backend Literal types
        nice_to_have:     { label: 'Bonus', icon: Star, color: 'text-success', badge: 'bg-success' },
        benefit:          { label: 'Benefit', icon: Award, color: 'text-success', badge: 'bg-success' },
        employer_mission: { label: 'Mission', icon: Heart, color: 'text-secondary', badge: 'bg-secondary' },
        employer_culture: { label: 'Culture', icon: Heart, color: 'text-secondary', badge: 'bg-secondary' },
        role_value:       { label: 'Value', icon: Star, color: 'text-secondary', badge: 'bg-secondary' },
        other:            { label: 'Other', icon: HelpCircle, color: 'text-muted', badge: 'bg-secondary' }
    };

    const getFeaturesByType = (types) => {
        if (!job?.features) return [];
        return job.features.filter(f => types.includes(f.type));
    };

    const groupResponsibilities = getFeaturesByType(['responsibility']);
    const groupRequirements = getFeaturesByType(['requirement', 'experience', 'qualification']);
    const groupSkills = getFeaturesByType(['hard_skill', 'soft_skill']);
    const groupCulture = getFeaturesByType(['nice_to_have', 'benefit', 'employer_mission', 'employer_culture', 'role_value', 'other']);

    // --- SUB-COMPONENT: QUICK ADD ---
    const QuickAddFeature = ({ allowedTypes }) => {
        const [isAdding, setIsAdding] = useState(false);
        const [text, setText] = useState('');
        // Initialize type with the one the user clicked, or default to first
        const [selectedType, setSelectedType] = useState(allowedTypes[0]); 

        const onAdd = async () => {
            if (!text.trim()) return;
            await handleAddFeature(text, selectedType);
            setText('');
            setIsAdding(false);
        };

        const startAdding = (type) => {
            setSelectedType(type);
            setIsAdding(true);
        };

        if (!isAdding) {
            return (
                <div className="d-flex flex-wrap gap-2 mt-2">
                    {allowedTypes.map(type => (
                        <button 
                            key={type}
                            onClick={() => startAdding(type)} 
                            className="btn btn-sm btn-link text-decoration-none text-muted p-0 d-flex align-items-center gap-1 opacity-75 hover-opacity-100"
                            style={{fontSize: '0.75rem'}}
                            title={`Add new ${TYPE_CONFIG[type]?.label}`}
                        >
                            <Plus size={12}/> {TYPE_CONFIG[type]?.label}
                        </button>
                    ))}
                </div>
            );
        }

        return (
            <div className="mt-2 p-2 bg-light rounded border border-light-subtle shadow-sm animate-fade-in">
                <div className="d-flex gap-2 align-items-center">
                    {/* Label showing what we are adding */}
                    <span className="badge bg-secondary bg-opacity-25 text-secondary border border-secondary border-opacity-25 fw-normal" style={{fontSize: '0.7rem'}}>
                        {TYPE_CONFIG[selectedType]?.label}
                    </span>
                   
                    <input 
                        className="form-control form-control-sm border-0 bg-white"
                        autoFocus
                        placeholder="Description..."
                        value={text}
                        onChange={e => setText(e.target.value)}
                        onKeyDown={e => {
                            if(e.key === 'Enter') onAdd();
                            if(e.key === 'Escape') setIsAdding(false);
                        }}
                    />
                    <button onClick={onAdd} className="btn btn-sm btn-primary py-0 px-2"><Check size={14}/></button>
                    <button onClick={() => setIsAdding(false)} className="btn btn-sm btn-light border py-0 px-2"><X size={14}/></button>
                </div>
            </div>
        );
    };

    // --- HELPER: FEATURE SELECTOR OPTIONS ---
    const TypeOptions = () => (
        <>
            <optgroup label="Core">
                <option value="responsibility">Task</option>
                <option value="requirement">Requirement</option>
                <option value="experience">Experience</option>
                <option value="qualification">Qualification</option>
                <option value="nice_to_have">Nice to Have</option>

            </optgroup>
            <optgroup label="Skills">
                <option value="hard_skill">Hard Skill</option>
                <option value="soft_skill">Soft Skill</option>
            </optgroup>
            <optgroup label="Culture">
                <option value="benefit">Benefit</option>
                <option value="employer_mission">Mission</option>
                <option value="employer_culture">Culture</option>
                <option value="role_value">Value</option>
                <option value="other">Other</option>
            </optgroup>
        </>
    );

    // --- RENDERER ---
    const renderFeatureSection = (title, items, defaultIcon, allowedTypes) => {
        // Always render if allowedTypes exist so we can show Add buttons
        const isTagSection = items.some(i => TYPE_CONFIG[i.type]?.isTag) || (allowedTypes.includes('hard_skill'));

        return (
            <div className="mb-4">
                <h6 className="small fw-bold text-uppercase text-dark mb-2 d-flex align-items-center gap-2 border-bottom pb-2">
                    {defaultIcon} {title}
                </h6>
                
                {isTagSection ? (
                    // --- TAGS LAYOUT (Skills) ---
                    <div className="d-flex flex-wrap gap-2">
                        {items.map(f => {
                            // EDIT MODE (Skills with Type Selector)
                            if (editingFeatureId === f.id) {
                                return (
                                    <div key={f.id} className="d-flex align-items-center gap-1 border rounded px-1 py-1 bg-white shadow-sm" style={{borderColor: '#dee2e6', zIndex: 10}}>
                                        <select 
                                            className="form-select form-select-sm border-0 py-0 ps-1 pe-2 bg-light text-primary fw-medium" 
                                            style={{
                                                width: 'auto', 
                                                fontSize: '0.75rem', 
                                                height: '24px', 
                                                cursor: 'pointer',
                                                appearance: 'none',
                                                WebkitAppearance: 'none',
                                                backgroundImage: 'none'
                                            }}
                                            value={editFeatureType}
                                            onChange={e => setEditFeatureType(e.target.value)}
                                        >
                                            <option value="hard_skill">Hard</option>
                                            <option value="soft_skill">Soft</option>
                                        </select>
                                        <div className="vr mx-1" style={{height: '16px'}}></div>
                                        <input 
                                            className="form-control form-control-sm border-0 p-0 shadow-none fw-normal bg-transparent" 
                                            value={editFeatureText}
                                            autoFocus
                                            onChange={e => setEditFeatureText(e.target.value)}
                                            onKeyDown={e => e.key === 'Enter' && saveEditFeature()}
                                            style={{width: `${Math.max(editFeatureText.length, 5)}ch`, minWidth: '60px', maxWidth: '200px', fontSize: '0.9rem'}}
                                        />
                                        <button onClick={saveEditFeature} className="btn btn-sm btn-link text-success p-0 lh-1"><Check size={14}/></button>
                                        <button onClick={cancelEditFeature} className="btn btn-sm btn-link text-muted p-0 ms-1 lh-1"><X size={14}/></button>
                                    </div>
                                );
                            }

                            // VIEW MODE
                            const badgeStyle = f.type === 'hard_skill' ? 'bg-dark text-white' : 'bg-info bg-opacity-10 text-info border border-info';
                            return (
                                <span 
                                    key={f.id} 
                                    className={`badge ${badgeStyle} fw-normal p-2 d-flex align-items-center gap-2 cursor-pointer`}
                                    onDoubleClick={() => startEditingFeature(f)}
                                    title="Double-click to edit"
                                >
                                    {f.description}
                                    <button onClick={(e) => { e.stopPropagation(); handleRemoveFeature(f.id); }} className="btn btn-link text-inherit p-0 ms-1 opacity-75 hover-opacity-100">
                                        <X size={12}/>
                                    </button>
                                </span>
                            );
                        })}
                    </div>
                ) : (
                    // --- LIST LAYOUT ---
                    <ul className="list-unstyled mb-0 ps-1">
                        {items.map(f => {
                            const config = TYPE_CONFIG[f.type] || TYPE_CONFIG['other'];
                            const Icon = config.icon;

                            if (editingFeatureId === f.id) {
                                return (
                                    <li key={f.id} className="mb-3">
                                        <div className="bg-white border border-primary rounded shadow-sm p-2 position-relative">
                                            {/* Header */}
                                            <div className="d-flex justify-content-between align-items-center mb-2 pb-2 border-bottom">
                                                <div className="d-flex align-items-center gap-2">
                                                    <span className={`text-${config.color.split('-')[1] || 'muted'}`}>
                                                        <Icon size={16}/>
                                                    </span>
                                                    <select 
                                                        className="form-select form-select-sm border-0 bg-transparent py-0 px-0 text-muted fw-medium" 
                                                        style={{width: 'auto', cursor: 'pointer', boxShadow: 'none'}}
                                                        value={editFeatureType}
                                                        onChange={e => setEditFeatureType(e.target.value)}
                                                    >
                                                        <TypeOptions />
                                                    </select>
                                                </div>
                                                <div className="d-flex gap-1">
                                                    <button onClick={saveEditFeature} className="btn btn-sm btn-success py-0 px-2 d-flex align-items-center gap-1" style={{fontSize: '0.75rem'}}>
                                                        <Check size={12}/> Save
                                                    </button>
                                                    <button onClick={cancelEditFeature} className="btn btn-sm btn-light border py-0 px-2 text-muted" style={{fontSize: '0.75rem'}}>
                                                        Cancel
                                                    </button>
                                                    <button onClick={() => handleRemoveFeature(f.id)} className="btn btn-sm btn-light border py-0 px-2 text-danger" style={{fontSize: '0.75rem'}} title="Delete">
                                                        <Trash2 size={12}/>
                                                    </button>
                                                </div>
                                            </div>
                                            {/* Body */}
                                            <textarea 
                                                className="form-control form-control-sm border-0 bg-light"
                                                value={editFeatureText}
                                                autoFocus
                                                rows={3}
                                                onChange={e => setEditFeatureText(e.target.value)}
                                                style={{resize: 'vertical', fontSize: '0.9rem'}}
                                            />
                                        </div>
                                    </li>
                                );
                            }
                            // VIEW MODE LIST ITEM
                            return (
                                <li 
                                    key={f.id} 
                                    className={`mb-2 d-flex align-items-start gap-2 p-2 rounded transition-all cursor-pointer border border-transparent hover-border-secondary hover-bg-light`}
                                    onDoubleClick={() => startEditingFeature(f)}
                                    title="Double-click to edit"
                                >
                                    <div className={`mt-1 ${config.color}`}>
                                        <Icon size={14} />
                                    </div>
                                    <div className="flex-grow-1">
                                        {/* Type Label (New: Helps visibility) */}
                                        <div className="d-flex align-items-center gap-2 mb-1">
                                            <span className="badge bg-light text-muted border px-1 py-0 fw-normal" style={{fontSize: '0.65rem', letterSpacing: '0.5px'}}>
                                                {config.label.toUpperCase()}
                                            </span>
                                        </div>
                                        <span className="text-secondary small" style={{lineHeight: '1.5', whiteSpace: 'pre-wrap'}}>
                                            {f.description}
                                        </span>
                                    </div>
                                    <button onClick={(e) => { e.stopPropagation(); handleRemoveFeature(f.id); }} className="btn btn-link text-danger p-0 ms-2 opacity-0 hover-opacity-100 transition-opacity">
                                        <X size={14}/>
                                    </button>
                                </li>
                            );
                        })}
                    </ul>
                )}
                {/* Local Add Button */}
                <QuickAddFeature allowedTypes={allowedTypes} />
            </div>
        );
    };

    if (loading) return <div className="p-5 text-center"><div className="spinner-border text-primary"/></div>;
    if (!job) return <div className="p-5 text-center">Job not found</div>;

    // --- HELPER FOR INLINE EDITABLE TEXT ---
    const EditableText = ({ field, value, placeholder, emptyLabel, className, style, as='input', type='text' }) => {
        const [localValue, setLocalValue] = useState(value || '');
        const isEditing = editField === field;
        const isEmpty = !value || value.trim() === '';

        useEffect(() => {
            if (isEditing) {
                setLocalValue(value || '');
            }
        }, [isEditing, value]);

        const handleSave = () => {
            if (localValue !== value) {
                saveField(field, localValue);
            } else {
                setEditField(null); 
            }
        };

        const handleKeyDownLocal = (e) => {
            if (e.key === 'Enter') {
                if(as === 'textarea' && !e.shiftKey) {
                    e.preventDefault(); 
                    handleSave();
                } else if (as !== 'textarea') {
                    handleSave();
                }
            } else if (e.key === 'Escape') {
                setEditField(null);
                setLocalValue(value || ''); 
            }
        };

        if (!isEditing && isEmpty) {
            return (
                <div 
                    className={`d-inline-flex align-items-center gap-1 cursor-pointer text-muted opacity-50 hover-opacity-100 transition-opacity ${className}`}
                    onClick={(e) => { e.stopPropagation(); setEditField(field); }}
                    title={`Click to add ${placeholder}`}
                    style={style}
                >
                    <Plus size={12}/> {emptyLabel || placeholder}
                </div>
            );
        }

        if (!isEditing) {
            return (
                <div 
                    className={`cursor-pointer hover-bg-light rounded transition-all ${className}`}
                    onClick={(e) => { 
                        e.stopPropagation(); 
                        setEditField(field); 
                    }}
                    title="Click to edit"
                    style={style}
                >
                    {value}
                </div>
            );
        }

        if (as === 'textarea') {
            return (
                <div className="position-relative h-100">
                    <textarea
                        className={`form-control ${className}`}
                        value={localValue}
                        onChange={(e) => setLocalValue(e.target.value)}
                        onKeyDown={handleKeyDownLocal}
                        onBlur={handleSave}
                        placeholder={placeholder}
                        autoFocus
                        style={{...style, resize: 'none'}}
                    />
                </div>
            );
        }
        
        return (
            <input
                type={type}
                className={`form-control ${className}`}
                value={localValue}
                onChange={(e) => setLocalValue(e.target.value)}
                onKeyDown={handleKeyDownLocal}
                onBlur={handleSave}
                placeholder={placeholder}
                autoFocus
                style={style}
            />
        );
    };

    return (
        <div className="container py-4" style={{ maxWidth: '1400px', height: '85vh', minHeight: '600px', display: 'flex', flexDirection: 'column' }}>
            <style>{`
                .scrollable-card-body { overflow-y: auto; height: 100%; }
                .scrollable-card-body::-webkit-scrollbar { width: 8px; }
                .scrollable-card-body::-webkit-scrollbar-thumb { background-color: #dee2e6; border-radius: 4px; }
                .hover-bg-light:hover { background-color: #f8f9fa; outline: 1px solid #dee2e6; }
                .hover-border-secondary:hover { border-color: #dee2e6 !important; }
                .hover-opacity-100:hover { opacity: 1 !important; }
                .transition-opacity { transition: opacity 0.2s ease; }
                .cursor-pointer { cursor: pointer; }
                .animate-fade-in { animation: fadeIn 0.2s ease-in-out; }
                @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; transform: translateY(0); } }
            `}</style>

            {/* --- 1. TOP NAVIGATION --- */}
            <div className="d-flex align-items-center justify-content-between mb-3 flex-shrink-0">
                <button onClick={() => navigate('/jobs')} className="btn btn-link text-decoration-none text-muted ps-0 d-flex align-items-center gap-2 hover-opacity-75">
                    <ArrowLeft size={18} /> Back to Library
                </button>
                <div className="d-flex gap-2">
                    <button onClick={() => handleDelete()} className="btn btn-outline-danger btn-sm d-flex align-items-center gap-2"><Trash2 size={16}/> Delete</button>
                    <button onClick={handleStartApplication} className="btn btn-primary btn-sm shadow-sm">Start Application</button>
                </div>
            </div>

            {/* --- 2. HEADER CARD --- */}
            <div className="card border-0 shadow-sm mb-3 flex-shrink-0">
                <div className="card-body p-4">
                    <div className="d-flex gap-4">
                        <div className="d-flex align-items-center justify-content-center bg-light rounded-3 text-primary border" style={{width:'64px', height:'64px'}}>
                            <Building2 size={32}/>
                        </div>
                        
                        {/* HEADER EDITABLE AREA */}
                        <div className="flex-grow-1">
                            <div className="mb-2">
                                <EditableText 
                                    field="title" 
                                    value={job.title} 
                                    placeholder="Job Title" 
                                    className="h4 fw-bold mb-1 d-inline-block px-1 -mx-1"
                                />
                                <div>
                                    <EditableText 
                                        field="company" 
                                        value={job.company} 
                                        placeholder="Company Name" 
                                        className="text-muted fw-medium fs-6 d-inline-block px-1 -mx-1"
                                    />
                                </div>
                            </div>
                            
                            <div className="d-flex flex-wrap gap-3 text-sm text-muted align-items-center mt-3">
                                {/* Location */}
                                <div className="d-flex align-items-center gap-1">
                                    <MapPin size={14} className={!job.location ? "opacity-50" : ""}/> 
                                    <EditableText 
                                        field="location" 
                                        value={job.location} 
                                        placeholder="Location" 
                                        emptyLabel="Location"
                                        className="d-inline-block bg-transparent border-0 p-0"
                                        style={{minWidth: '50px'}}
                                    />
                                </div>

                                {/* Salary */}
                                <div className="d-flex align-items-center gap-1">
                                    <Banknote size={14} className={!job.salary_range ? "opacity-50" : ""}/> 
                                    <EditableText 
                                        field="salary_range" 
                                        value={job.salary_range} 
                                        placeholder="Salary" 
                                        emptyLabel="Salary"
                                        className="d-inline-block bg-transparent border-0 p-0"
                                        style={{minWidth: '50px'}}
                                    />
                                </div>

                                {/* Posted Date */}
                                <div className="d-flex align-items-center gap-1">
                                    <Calendar size={14} className={!job.date_posted ? "opacity-50" : ""}/> 
                                    {job.date_posted && <span className="small text-muted me-1">Posted:</span>}
                                    <EditableText 
                                        field="date_posted" 
                                        value={job.date_posted} 
                                        placeholder="dd/mm/yyyy" 
                                        emptyLabel="Posted Date"
                                        type="date"
                                        className="d-inline-block bg-transparent border-0 p-0"
                                        style={{minWidth: '80px'}}
                                    />
                                </div>

                                {/* Closing Date */}
                                <div className="d-flex align-items-center gap-1">
                                    <Clock size={14} className={!job.date_closing ? "opacity-50" : ""}/> 
                                    {job.date_closing && <span className="small text-muted me-1">Closing:</span>}
                                    <EditableText 
                                        field="date_closing" 
                                        value={job.date_closing} 
                                        placeholder="dd/mm/yyyy" 
                                        emptyLabel="Closing Date"
                                        type="date"
                                        className="d-inline-block bg-transparent border-0 p-0"
                                        style={{minWidth: '80px'}}
                                    />
                                </div>

                                {job.job_url && (
                                    <a href={job.job_url} target="_blank" rel="noreferrer" className="btn btn-sm btn-outline-primary d-flex align-items-center gap-2 ms-auto">
                                        <Globe size={14}/> Original Post
                                    </a>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* --- 3. MAIN CONTENT --- */}
            <div className="row g-3 flex-grow-1" style={{ minHeight: 0 }}>
                
                {/* --- LEFT: DESCRIPTION --- */}
                <div className="col-lg-7 h-100">
                    <div className="card border-0 shadow-sm h-100 d-flex flex-column">
                        <div className="card-header bg-white border-bottom-0 pt-4 px-4 pb-0 flex-shrink-0">
                            <h5 className="fw-bold mb-0 text-dark">Job Description</h5>
                        </div>
                        <div className="card-body p-4 scrollable-card-body">
                            <EditableText 
                                field="description"
                                value={job.displayed_description || job.description}
                                placeholder="Paste job description here..."
                                emptyLabel="Add Job Description"
                                as="textarea"
                                className="h-100 w-100 border-0 px-2 py-2 -mx-2"
                                style={{whiteSpace: 'pre-wrap', lineHeight: '1.7', fontSize: '0.95rem'}}
                            />
                        </div>
                    </div>
                </div>

                {/* --- RIGHT: NOTES + FEATURES --- */}
                <div className="col-lg-5 h-100 d-flex flex-column">
                    
                    {/* User Notes */}
                    <div className="card border-0 shadow-sm mb-3 bg-warning bg-opacity-10 flex-shrink-0">
                        <div className="card-body py-3">
                            <h6 className="fw-bold text-warning-emphasis mb-2 text-uppercase" style={{fontSize: '0.75rem'}}>My Notes</h6>
                            <EditableText 
                                field="notes"
                                value={job.notes}
                                placeholder="Add your notes here..."
                                emptyLabel="Click to add notes..."
                                as="textarea"
                                className="w-100 bg-transparent border-0 p-1 -m-1"
                                style={{whiteSpace: 'pre-wrap', minHeight: '60px', maxHeight: '120px', overflowY: 'auto'}}
                            />
                        </div>
                    </div>

                    {/* AI Features */}
                    <div className="card border-0 shadow-sm flex-grow-1 d-flex flex-column" style={{ minHeight: 0 }}>
                        <div className="card-header bg-white border-bottom-0 pt-3 px-3 flex-shrink-0">
                            <h6 className="fw-bold mb-0 d-flex align-items-center gap-2 text-dark">
                                <ListChecks size={18} className="text-primary"/> AI Extracted Data
                            </h6>
                        </div>
                        
                        <div className="card-body p-3 scrollable-card-body">
                            {renderFeatureSection("Responsibilities", groupResponsibilities, <ListChecks size={16} className="text-primary"/>, ['responsibility'])}
                            {renderFeatureSection("Requirements & Qualifications", groupRequirements, <CheckCircle2 size={16} className="text-danger"/>, ['requirement', 'experience', 'qualification', 'nice_to_have'])}
                            {renderFeatureSection("Skills & Tech", groupSkills, <Zap size={16} className="text-dark"/>, ['hard_skill', 'soft_skill'])}
                            {renderFeatureSection("Culture & Benefits", groupCulture, <Star size={16} className="text-success"/>, ['benefit', 'employer_mission', 'employer_culture', 'role_value', 'other'])}

                            {(!job.features || job.features.length === 0) && (
                                <div className="text-center text-muted py-4 small fst-italic">No structured features extracted yet.</div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
export default JobDetails;