import React, { useState, useEffect } from 'react';
import { Edit2, Phone, Mail, Globe, MapPin, Linkedin, Plus, X, Save } from 'lucide-react';

export const ContactInfoManager = ({ contactInfo = {}, onSave }) => {
    const [isEditing, setIsEditing] = useState(false);
    const [formData, setFormData] = useState([]);

    // Initialize form data when editing starts or contactInfo changes
    useEffect(() => {
        if (!isEditing && contactInfo) {
            const asArray = Object.entries(contactInfo || {}).map(([key, value], index) => ({
                id: Date.now() + index, // Use timestamp to ensure unique keys
                key: key, 
                value: value
            }));
            setFormData(asArray);
        }
    }, [contactInfo, isEditing]);

    const handleSave = async () => {
        const newContactInfo = {};
        formData.forEach(item => {
            if (item.key && item.key.trim() && item.value && item.value.trim()) {
                newContactInfo[item.key.trim()] = item.value.trim();
            }
        });
        await onSave(newContactInfo);
        setIsEditing(false);
    };

    const addField = () => {
        setFormData(prev => [...prev, { id: Date.now(), key: '', value: '' }]);
    };

    const removeField = (index) => {
        setFormData(prev => prev.filter((_, i) => i !== index));
    };

    // FIXED: Immutable state update
    const updateField = (index, field, val) => {
        setFormData(prev => prev.map((item, i) => 
            i === index ? { ...item, [field]: val } : item
        ));
    };

    const getIcon = (key, size = 16) => {
        const k = (key || '').toLowerCase();
        const props = { size, className: "text-primary" };
        if (k.includes('phone') || k.includes('tel') || k.includes('mobile')) return <Phone {...props} />;
        if (k.includes('mail') || k.includes('email')) return <Mail {...props} />;
        if (k.includes('linkedin')) return <Linkedin {...props} />;
        if (k.includes('github') || k.includes('git')) return <Globe {...props} />; 
        if (k.includes('web') || k.includes('site') || k.includes('portfolio') || k.includes('link')) return <Globe {...props} />;
        if (k.includes('address') || k.includes('location') || k.includes('city')) return <MapPin {...props} />;
        return <Globe {...props} />; 
    };

    const renderValue = (key, value) => {
        const k = (key || '').toLowerCase();
        const v = (value || '').trim();
        const linkClass = "text-dark fw-bold text-decoration-none hover-underline";

        if (k.includes('mail') || k.includes('email')) {
            return <a href={`mailto:${v}`} className={linkClass}>{v}</a>;
        }
        if (k.includes('phone') || k.includes('tel') || k.includes('mobile')) {
             return <a href={`tel:${v}`} className={linkClass}>{v}</a>;
        }
        if (v.startsWith('http') || v.startsWith('www') || v.includes('.co')) {
            const href = `https://${v}`;
            return <a href={href}  className={linkClass}>{v}</a>;
        }
        return <span className="text-dark fw-bold">{v}</span>;
    };

    // --- EDIT VIEW ---
    if (isEditing) {
        return (
            <div className="card border-0 shadow-sm bg-light mt-4 animate-fade-in">
                <div className="card-body p-4">
                    <div className="d-flex align-items-center justify-content-between mb-4">
                        <h5 className="fw-bold text-dark mb-0">Edit Contact Details</h5>
                        <button onClick={() => setIsEditing(false)} className="btn-close" aria-label="Close"></button>
                    </div>

                    <div className="d-flex flex-column gap-3">
                        {formData.map((item, idx) => (
                            <div key={item.id} className="row g-2 align-items-center">
                                <div className="col-4">
                                    <div className="input-group input-group-sm">
                                        <span className="input-group-text bg-white border-end-0 text-muted">
                                            {getIcon(item.key, 14)}
                                        </span>
                                        <input 
                                            type="text" 
                                            className="form-control form-control-sm border-start-0 ps-1 text-dark fw-medium" 
                                            placeholder="Label (e.g. Email)"
                                            value={item.key}
                                            onChange={(e) => updateField(idx, 'key', e.target.value)}
                                        />
                                    </div>
                                </div>
                                <div className="col-7">
                                    <input 
                                        type="text" 
                                        className="form-control form-control-sm" 
                                        placeholder="Value"
                                        value={item.value}
                                        onChange={(e) => updateField(idx, 'value', e.target.value)}
                                    />
                                </div>
                                <div className="col-1 text-center">
                                    <button 
                                        onClick={() => removeField(idx)}
                                        className="btn btn-outline-danger btn-sm border-0 rounded-circle p-1"
                                        title="Remove"
                                    >
                                        <X size={16} />
                                    </button>
                                </div>
                            </div>
                        ))}
                        {formData.length === 0 && (
                            <div className="text-center p-3 border border-dashed rounded bg-white">
                                <p className="text-muted mb-0 small">No contact fields added yet.</p>
                            </div>
                        )}
                    </div>

                    <div className="d-flex align-items-center justify-content-between mt-4 pt-3 border-top">
                         <button onClick={addField} className="btn btn-light btn-sm text-primary fw-medium d-flex align-items-center gap-2 border">
                            <Plus size={14} /> Add Field
                        </button>
                        <div className="d-flex gap-2">
                            <button onClick={() => setIsEditing(false)} className="btn btn-light btn-sm">Cancel</button>
                            <button onClick={handleSave} className="btn btn-primary btn-sm d-flex align-items-center gap-2 px-3 shadow-sm">
                                <Save size={14} /> Save Changes
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // --- DISPLAY VIEW ---
    return (
        <div className="mt-4 pt-2">
            <style>
                {`.hover-underline:hover { text-decoration: underline !important; color: var(--bs-primary) !important; }`}
            </style>
             <div className="d-flex align-items-center justify-content-between mb-3">
                <h5 className="fw-bold text-dark mb-0">Contact Details</h5>
                <button 
                    onClick={() => {
                        const asArray = Object.entries(contactInfo || {}).map(([key, value], index) => ({
                            id: Date.now() + index, key: key, value: value
                        }));
                        setFormData(asArray);
                        setIsEditing(true);
                    }} 
                    className="btn btn-outline-secondary btn-sm d-flex align-items-center gap-2 px-3"
                >
                    <Edit2 size={14} /> Edit
                </button>
            </div>
            
            <div className="d-flex flex-wrap gap-2">
                {(!contactInfo || Object.keys(contactInfo).length === 0) && (
                     <span className="text-muted fst-italic py-2">No contact info provided. Click edit to add details.</span>
                )}
                
                {Object.entries(contactInfo || {}).map(([key, value]) => (
                    <div 
                        key={key} 
                        className="d-inline-flex align-items-center bg-white border rounded-pill shadow-sm px-3 py-2 hover-lift transition-all"
                        style={{ minHeight: '42px' }}
                    >
                        <div className="me-2 d-flex align-items-center justify-content-center opacity-75">
                             {getIcon(key, 18)} 
                        </div>
                        <div className="d-flex align-items-center">
                            <span className="text-uppercase text-secondary fw-bold small me-2" style={{fontSize: '0.7rem', letterSpacing: '0.5px'}}>
                                {key}:
                            </span>
                            <span className="fs-6">
                                {renderValue(key, value)}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};