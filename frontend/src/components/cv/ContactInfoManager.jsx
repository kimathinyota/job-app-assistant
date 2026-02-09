import React, { useState, useEffect, useRef } from 'react';
import { Phone, Mail, Globe, MapPin, Linkedin, Plus, X } from 'lucide-react';

export const ContactInfoManager = ({ contactInfo = {}, isEditing, onChange }) => {
    // Local state for the inputs
    const [formData, setFormData] = useState([]);
    
    // We use a ref to track if we have initialized the edit form data
    // to prevent the useEffect from overwriting user input while editing.
    const isInitializedRef = useRef(false);

    // --- EFFECT: SYNC DATA ---
    useEffect(() => {
        // 1. If we are NOT editing, always display the latest props (View Mode)
        if (!isEditing) {
            const asArray = Object.entries(contactInfo || {}).map(([key, value], index) => ({
                id: index, 
                key: key, 
                value: value
            }));
            setFormData(asArray);
            isInitializedRef.current = false; // Reset init flag
        } 
        // 2. If we ARE editing, but haven't initialized the form yet, load the data once.
        else if (isEditing && !isInitializedRef.current) {
             const asArray = Object.entries(contactInfo || {}).map(([key, value], index) => ({
                id: index, 
                key: key, 
                value: value
            }));
            setFormData(asArray);
            isInitializedRef.current = true; // Mark as initialized so we don't overwrite again
        }
        
        // 3. If we are editing AND initialized, DO NOTHING. 
        // This effectively ignores prop updates while typing, preserving your local state (including empty fields).
        
    }, [contactInfo, isEditing]);

    // Helper to notify parent of changes
    const emitChange = (newData) => {
        // 1. Update Local State immediately (UI Update)
        setFormData(newData);
        
        // 2. Prepare Valid Object for Parent (Data Update)
        const newContactObject = {};
        newData.forEach(item => {
            // Only save fields that have a Key
            if (item.key && item.key.trim()) {
                newContactObject[item.key.trim()] = item.value;
            }
        });
        
        // 3. Send to parent
        onChange(newContactObject);
    };

    const addField = () => {
        // Add unique ID based on timestamp to avoid key collisions
        emitChange([...formData, { id: Date.now(), key: '', value: '' }]);
    };

    const removeField = (index) => {
        const newDat = [...formData];
        newDat.splice(index, 1);
        emitChange(newDat);
    };

    const updateField = (index, field, val) => {
        const newDat = [...formData];
        newDat[index][field] = val;
        emitChange(newDat);
    };

    const getIcon = (key, size = 16) => {
        const k = (key || "").toLowerCase();
        const props = { size, className: "text-primary" };
        if (k.includes('phone') || k.includes('tel')) return <Phone {...props} />;
        if (k.includes('mail') || k.includes('email')) return <Mail {...props} />;
        if (k.includes('linkedin')) return <Linkedin {...props} />;
        if (k.includes('git') || k.includes('hub')) return <Globe {...props} />; 
        if (k.includes('address') || k.includes('location')) return <MapPin {...props} />;
        return <Globe {...props} />; 
    };

    const renderValue = (key, value) => {
        const k = (key || "").toLowerCase();
        const v = value || "";
        
        // 'stretched-link' makes the parent card clickable
        const linkClass = "text-dark fw-bold text-decoration-none hover-underline text-break stretched-link";

        if (k.includes('mail')) return <a href={`mailto:${v}`} className={linkClass}>{v}</a>;
        if (k.includes('phone') || k.includes('tel')) return <a href={`tel:${v}`} className={linkClass}>{v}</a>;
        
        const isWebLink = v.startsWith('http') || v.startsWith('www') || k.includes('link') || k.includes('git') || k.includes('web') || k.includes('site') || k.includes('hub');

        if (isWebLink) {
            let href = v;
            if (!v.startsWith('http') && !v.startsWith('//')) {
                href = `https://${v}`;
            }
            return <a href={href} target="_blank" rel="noopener noreferrer" className={linkClass}>{v}</a>;
        }

        return <span className="text-dark fw-bold text-break">{v}</span>;
    };

    // --- 1. EDIT MODE ---
    if (isEditing) {
        return (
            <div className="mt-3 w-100">
                <div className="d-flex align-items-center justify-content-between mb-2">
                    <label className="form-label fw-bold small text-uppercase text-muted mb-0">Contact Details</label>
                    <button type="button" onClick={addField} className="btn btn-sm btn-link text-decoration-none p-0 d-flex align-items-center gap-1">
                        <Plus size={14}/> Add Field
                    </button>
                </div>
                
                <div className="d-flex flex-column gap-2">
                    {formData.map((item, idx) => (
                        <div key={item.id} className="row g-2 align-items-center">
                            <div className="col-4">
                                <div className="input-group input-group-sm">
                                    <span className="input-group-text bg-white border-end-0 text-muted ps-2 pe-2">
                                        {getIcon(item.key, 14)}
                                    </span>
                                    <input 
                                        type="text" 
                                        className="form-control form-control-sm border-start-0 ps-1 fw-medium" 
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
                                <button type="button" onClick={() => removeField(idx)} className="btn btn-outline-danger btn-sm border-0 rounded-circle p-1 d-flex align-items-center justify-content-center" style={{width: '24px', height: '24px'}}>
                                    <X size={14} />
                                </button>
                            </div>
                        </div>
                    ))}
                    {formData.length === 0 && <div className="text-muted small fst-italic border rounded p-2 text-center bg-light">No contact info. Click Add Field.</div>}
                </div>
            </div>
        );
    }

    // --- 2. VIEW MODE ---
    return (
        <div className="mt-4 pt-2 w-100">
            <style>{`.hover-underline:hover { text-decoration: underline !important; color: var(--bs-primary) !important; }`}</style>
            
            <div className="row g-2">
                {(!contactInfo || Object.keys(contactInfo).length === 0) && (
                     <div className="col-12 text-muted small fst-italic">No contact details provided.</div>
                )}
                
                {Object.entries(contactInfo || {}).map(([key, value]) => (
                    <div key={key} className="col-12 col-md-6 col-lg-4">
                        <div className="d-flex align-items-center bg-white border rounded shadow-sm px-3 py-2 h-100 hover-lift transition-all position-relative">
                            <div className="me-3 d-flex align-items-center justify-content-center opacity-75 bg-light p-2 rounded-circle">
                                 {getIcon(key, 18)} 
                            </div>
                            <div className="d-flex flex-column justify-content-center overflow-hidden">
                                <span className="text-uppercase text-secondary fw-bold small" style={{fontSize: '0.65rem', letterSpacing: '0.5px'}}>
                                    {key}
                                </span>
                                <span className="fs-6 text-truncate w-100">
                                    {renderValue(key, value)}
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};