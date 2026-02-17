import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
    MapPin, Banknote, Globe, Building2, 
    ArrowLeft, Save, Edit2, Trash2, Plus, X, Check,
    CheckCircle2, ListChecks, Zap, Star, Briefcase, GraduationCap, Heart, Award, HelpCircle, ChevronDown
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
    const [isEditing, setIsEditing] = useState(false);
    
    // Global Form State
    const [formData, setFormData] = useState({});
    
    // Add New Feature State
    const [newFeatureText, setNewFeatureText] = useState('');
    const [newFeatureType, setNewFeatureType] = useState('responsibility');

    // Inline Editing State
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

    // --- GLOBAL ACTIONS ---
    const handleSave = async () => {
        try {
            await updateJob(jobId, formData);
            setIsEditing(false);
            loadJob(); 
        } catch (err) {
            alert("Failed to save changes");
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
    const handleAddFeature = async () => {
        if(!newFeatureText.trim()) return;
        await addJobFeature(jobId, newFeatureText, newFeatureType);
        setNewFeatureText('');
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
        responsibility: { label: 'Task', icon: ListChecks, color: 'text-primary' },
        requirement:    { label: 'Must Have', icon: CheckCircle2, color: 'text-danger' },
        experience:     { label: 'Experience', icon: Briefcase, color: 'text-danger' },
        qualification:  { label: 'Qualification', icon: GraduationCap, color: 'text-danger' },
        hard_skill:     { label: 'Hard Skill', icon: Zap, color: 'text-dark', isTag: true },
        soft_skill:     { label: 'Soft Skill', icon: Zap, color: 'text-info', isTag: true },
        nice_to_have:     { label: 'Bonus', icon: Star, color: 'text-success' },
        benefit:          { label: 'Benefit', icon: Award, color: 'text-success' },
        employer_mission: { label: 'Mission', icon: Heart, color: 'text-secondary' },
        employer_culture: { label: 'Culture', icon: Heart, color: 'text-secondary' },
        role_value:       { label: 'Value', icon: Star, color: 'text-secondary' },
        other:            { label: 'Other', icon: HelpCircle, color: 'text-muted' }
    };

    const getFeaturesByType = (types) => {
        if (!job?.features) return [];
        return job.features.filter(f => types.includes(f.type));
    };

    const groupResponsibilities = getFeaturesByType(['responsibility']);
    const groupRequirements = getFeaturesByType(['requirement', 'experience', 'qualification']);
    const groupSkills = getFeaturesByType(['hard_skill', 'soft_skill']);
    const groupCulture = getFeaturesByType(['nice_to_have', 'benefit', 'employer_mission', 'employer_culture', 'role_value', 'other']);

    // --- HELPER: FEATURE SELECTOR OPTIONS ---
    const TypeOptions = () => (
        <>
            <optgroup label="Core">
                <option value="responsibility">Task</option>
                <option value="requirement">Requirement</option>
                <option value="experience">Experience</option>
                <option value="qualification">Qualification</option>
            </optgroup>
            <optgroup label="Skills">
                <option value="hard_skill">Hard Skill</option>
                <option value="soft_skill">Soft Skill</option>
            </optgroup>
            <optgroup label="Culture">
                <option value="nice_to_have">Nice to Have</option>
                <option value="benefit">Benefit</option>
                <option value="employer_mission">Mission</option>
                <option value="employer_culture">Culture</option>
                <option value="role_value">Value</option>
                <option value="other">Other</option>
            </optgroup>
        </>
    );

    // --- RENDERER ---
    const renderFeatureSection = (title, items, defaultIcon) => {
        if (items.length === 0 && !isEditing) return null;
        const isTagSection = items.some(i => TYPE_CONFIG[i.type]?.isTag);

        return (
            <div className="mb-4">
                <h6 className="small fw-bold text-uppercase text-dark mb-2 d-flex align-items-center gap-2 border-bottom pb-2">
                    {defaultIcon} {title}
                </h6>
                
                {isTagSection ? (
                    // --- TAGS LAYOUT (Skills) ---
                    <div className="d-flex flex-wrap gap-2">
                        {items.map(f => {
                            // EDIT MODE (Compact Popover Style)
                            if (editingFeatureId === f.id) {
                                return (
                                    <div key={f.id} className="d-flex align-items-center gap-1 bg-white border border-primary rounded p-1 shadow-sm" style={{zIndex: 10}}>
                                        <select 
                                            className="form-select form-select-sm border-0 py-0 ps-1 pe-3 bg-light text-primary fw-medium" 
                                            style={{width: 'auto', fontSize: '0.8rem', cursor: 'pointer'}}
                                            value={editFeatureType}
                                            onChange={e => setEditFeatureType(e.target.value)}
                                        >
                                            <option value="hard_skill">Hard</option>
                                            <option value="soft_skill">Soft</option>
                                        </select>
                                        <div className="vr mx-1"></div>
                                        <input 
                                            className="form-control form-control-sm border-0 py-0 px-1" 
                                            value={editFeatureText}
                                            autoFocus
                                            onChange={e => setEditFeatureText(e.target.value)}
                                            onKeyDown={e => e.key === 'Enter' && saveEditFeature()}
                                            style={{minWidth: '100px'}}
                                        />
                                        <button onClick={saveEditFeature} className="btn btn-sm btn-link text-success p-0"><Check size={14}/></button>
                                        <button onClick={cancelEditFeature} className="btn btn-sm btn-link text-muted p-0 ms-1"><X size={14}/></button>
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
                                    {isEditing && (
                                        <button onClick={(e) => { e.stopPropagation(); handleRemoveFeature(f.id); }} className="btn btn-link text-inherit p-0 ms-1 opacity-75">
                                            <X size={12}/>
                                        </button>
                                    )}
                                </span>
                            );
                        })}
                    </div>
                ) : (
                    // --- LIST LAYOUT (Responsibilities, etc.) ---
                    <ul className="list-unstyled mb-0 ps-1">
                        {items.map(f => {
                            const config = TYPE_CONFIG[f.type] || TYPE_CONFIG['other'];
                            const Icon = config.icon;

                            // [NEW] EDIT MODE: Card Layout for cleanliness
                            if (editingFeatureId === f.id) {
                                return (
                                    <li key={f.id} className="mb-3">
                                        <div className="bg-white border border-primary rounded shadow-sm p-2 position-relative">
                                            
                                            {/* Header: Compact Type Selector + Buttons */}
                                            <div className="d-flex justify-content-between align-items-center mb-2 pb-2 border-bottom">
                                                <div className="d-flex align-items-center gap-2">
                                                    {/* The Icon reflects the CURRENT selection visually */}
                                                    <span className={`text-${config.color.split('-')[1] || 'muted'}`}>
                                                        <Icon size={16}/>
                                                    </span>
                                                    
                                                    {/* Minimalist Select */}
                                                    <select 
                                                        className="form-select form-select-sm border-0 bg-transparent py-0 px-0 text-muted fw-medium" 
                                                        style={{width: 'auto', cursor: 'pointer', boxShadow: 'none'}}
                                                        value={editFeatureType}
                                                        onChange={e => setEditFeatureType(e.target.value)}
                                                    >
                                                        <TypeOptions />
                                                    </select>
                                                    <ChevronDown size={12} className="text-muted ms-n1"/>
                                                </div>

                                                <div className="d-flex gap-1">
                                                    <button onClick={saveEditFeature} className="btn btn-sm btn-success py-0 px-2 d-flex align-items-center gap-1" style={{fontSize: '0.75rem'}}>
                                                        <Check size={12}/> Save
                                                    </button>
                                                    <button onClick={cancelEditFeature} className="btn btn-sm btn-light border py-0 px-2 text-muted" style={{fontSize: '0.75rem'}}>
                                                        Cancel
                                                    </button>
                                                </div>
                                            </div>

                                            {/* Body: Multi-line Text Area */}
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

                            // VIEW MODE (Standard List Item)
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
                                    <span className="text-secondary small flex-grow-1" style={{lineHeight: '1.5', whiteSpace: 'pre-wrap'}}>
                                        {f.description}
                                    </span>
                                    {isEditing && (
                                        <button onClick={(e) => { e.stopPropagation(); handleRemoveFeature(f.id); }} className="btn btn-link text-danger p-0 ms-2 opacity-50 hover-opacity-100">
                                            <X size={14}/>
                                        </button>
                                    )}
                                </li>
                            );
                        })}
                    </ul>
                )}
            </div>
        );
    };

    if (loading) return <div className="p-5 text-center"><div className="spinner-border text-primary"/></div>;
    if (!job) return <div className="p-5 text-center">Job not found</div>;

    return (
        <div className="container py-4" style={{ maxWidth: '1400px', height: '85vh', minHeight: '600px', display: 'flex', flexDirection: 'column' }}>
            <style>{`
                .scrollable-card-body { overflow-y: auto; height: 100%; }
                .scrollable-card-body::-webkit-scrollbar { width: 8px; }
                .scrollable-card-body::-webkit-scrollbar-thumb { background-color: #dee2e6; border-radius: 4px; }
                .hover-bg-light:hover { background-color: #f8f9fa; }
                .hover-border-secondary:hover { border-color: #dee2e6 !important; }
                .hover-opacity-100:hover { opacity: 1 !important; }
            `}</style>

            {/* --- 1. TOP NAVIGATION --- */}
            <div className="d-flex align-items-center justify-content-between mb-3 flex-shrink-0">
                <button onClick={() => navigate('/jobs')} className="btn btn-link text-decoration-none text-muted ps-0 d-flex align-items-center gap-2 hover-opacity-75">
                    <ArrowLeft size={18} /> Back to Library
                </button>
                <div className="d-flex gap-2">
                    {!isEditing ? (
                        <>
                            <button onClick={() => handleDelete()} className="btn btn-outline-danger btn-sm d-flex align-items-center gap-2"><Trash2 size={16}/> Delete</button>
                            <button onClick={() => setIsEditing(true)} className="btn btn-outline-primary btn-sm d-flex align-items-center gap-2"><Edit2 size={16}/> Edit Details</button>
                            <button onClick={handleStartApplication} className="btn btn-primary btn-sm shadow-sm">Start Application</button>
                        </>
                    ) : (
                        <>
                            <button onClick={() => { setIsEditing(false); setFormData(job); }} className="btn btn-secondary btn-sm">Cancel</button>
                            <button onClick={handleSave} className="btn btn-success btn-sm d-flex align-items-center gap-2"><Save size={16}/> Save Changes</button>
                        </>
                    )}
                </div>
            </div>

            {/* --- 2. HEADER CARD --- */}
            <div className="card border-0 shadow-sm mb-3 flex-shrink-0">
                <div className="card-body p-4">
                    <div className="d-flex gap-4">
                        <div className="d-flex align-items-center justify-content-center bg-light rounded-3 text-primary border" style={{width:'64px', height:'64px'}}>
                            <Building2 size={32}/>
                        </div>
                        <div className="flex-grow-1">
                            {isEditing ? (
                                <div className="row g-2 mb-2">
                                    <div className="col-md-7">
                                        <input className="form-control form-control-lg fw-bold" value={formData.title} onChange={e=>setFormData({...formData, title: e.target.value})} placeholder="Job Title"/>
                                    </div>
                                    <div className="col-md-5">
                                        <input className="form-control form-control-lg" value={formData.company} onChange={e=>setFormData({...formData, company: e.target.value})} placeholder="Company"/>
                                    </div>
                                </div>
                            ) : (
                                <div className="mb-2">
                                    <h2 className="h4 fw-bold mb-0 text-dark">{job.title}</h2>
                                    <div className="text-muted fw-medium fs-6">{job.company}</div>
                                </div>
                            )}
                            
                            <div className="d-flex flex-wrap gap-2 text-sm text-muted align-items-center">
                                {/* Location */}
                                {isEditing ? (
                                    <div className="input-group input-group-sm" style={{maxWidth: '200px'}}>
                                        <span className="input-group-text"><MapPin size={14}/></span>
                                        <input className="form-control" value={formData.location || ''} onChange={e=>setFormData({...formData, location: e.target.value})} placeholder="Location"/>
                                    </div>
                                ) : (
                                    <div className="d-flex align-items-center gap-1 bg-light px-2 py-1 rounded border border-light">
                                        <MapPin size={14}/> {job.location || 'Remote'}
                                    </div>
                                )}

                                {/* Salary */}
                                {isEditing ? (
                                    <div className="input-group input-group-sm" style={{maxWidth: '200px'}}>
                                        <span className="input-group-text"><Banknote size={14}/></span>
                                        <input className="form-control" value={formData.salary_range || ''} onChange={e=>setFormData({...formData, salary_range: e.target.value})} placeholder="Salary"/>
                                    </div>
                                ) : (
                                    <div className="d-flex align-items-center gap-1 bg-light px-2 py-1 rounded border border-light">
                                        <Banknote size={14}/> {job.salary_range || 'Not listed'}
                                    </div>
                                )}

                                {job.job_url && (
                                    <a href={job.job_url} target="_blank" rel="noreferrer" className="btn btn-sm btn-outline-primary d-flex align-items-center gap-2">
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
                            {isEditing ? (
                                <textarea className="form-control h-100" value={formData.description} onChange={e=>setFormData({...formData, description: e.target.value})} style={{resize: 'none', lineHeight: '1.6'}}/>
                            ) : (
                                <div className="text-dark opacity-75" style={{whiteSpace: 'pre-wrap', lineHeight: '1.7', fontSize: '0.95rem'}}>
                                    {job.displayed_description || job.description}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* --- RIGHT: NOTES + FEATURES --- */}
                <div className="col-lg-5 h-100 d-flex flex-column">
                    
                    {/* User Notes */}
                    <div className="card border-0 shadow-sm mb-3 bg-warning bg-opacity-10 flex-shrink-0">
                        <div className="card-body py-3">
                            <h6 className="fw-bold text-warning-emphasis mb-2 text-uppercase" style={{fontSize: '0.75rem'}}>My Notes</h6>
                            {isEditing ? (
                                <textarea className="form-control" rows={2} value={formData.notes || ''} onChange={e=>setFormData({...formData, notes: e.target.value})} placeholder="Thoughts..."/>
                            ) : (
                                <p className="small mb-0 text-dark opacity-75" style={{whiteSpace: 'pre-wrap', maxHeight: '80px', overflowY: 'auto'}}>{job.notes || "No notes added."}</p>
                            )}
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
                            {/* Pro Tip Banner */}
                            {isEditing && (
                                <div className="alert alert-info py-2 mb-3">
                                    <div className="d-flex gap-2 align-items-start">
                                        <Zap size={16} className="mt-1 flex-shrink-0"/>
                                        <div className="small">
                                            <strong>Reviewing these items helps improve your match score accuracy.</strong>
                                            <br/>
                                            <span className="text-muted" style={{fontSize: '0.8rem'}}>Tip: Double-click any item to edit individually.</span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {renderFeatureSection("Responsibilities", groupResponsibilities, <ListChecks size={16} className="text-primary"/>)}
                            {renderFeatureSection("Requirements & Qualifications", groupRequirements, <CheckCircle2 size={16} className="text-danger"/>)}
                            {renderFeatureSection("Skills & Tech", groupSkills, <Zap size={16} className="text-dark"/>)}
                            {renderFeatureSection("Culture & Benefits", groupCulture, <Star size={16} className="text-success"/>)}

                            {(!job.features || job.features.length === 0) && (
                                <div className="text-center text-muted py-4 small fst-italic">No structured features extracted yet.</div>
                            )}

                            {/* Add Feature Row */}
                            <div className="mt-4 pt-3 border-top">
                                <label className="form-label small fw-bold text-muted text-uppercase mb-2">Add Missing Highlight</label>
                                <div className="input-group input-group-sm">
                                    <select className="form-select bg-light" style={{maxWidth: '140px'}} value={newFeatureType} onChange={e => setNewFeatureType(e.target.value)}>
                                        <TypeOptions />
                                    </select>
                                    <input className="form-control" placeholder="Paste item..." value={newFeatureText} onChange={e => setNewFeatureText(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleAddFeature()}/>
                                    <button onClick={handleAddFeature} className="btn btn-outline-primary" title="Add"><Plus size={16}/></button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default JobDetails;