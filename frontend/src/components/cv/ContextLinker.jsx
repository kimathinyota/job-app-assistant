import React, { useState } from 'react';
import { Briefcase, BookOpen, Smile, Check, Plus, Search } from 'lucide-react';

const ContextLinker = ({
    allExperiences = [],
    allEducation = [],
    allHobbies = [],
    selectedIds = { experiences: [], education: [], hobbies: [] },
    onToggle // Function(type, id)
}) => {
    const [searchTerm, setSearchTerm] = useState('');

    // Helper to render a section
    const renderSection = (title, items, type, Icon) => {
        const filtered = items.filter(i => 
            (i.title || i.degree || i.name || '').toLowerCase().includes(searchTerm.toLowerCase())
        );

        if (filtered.length === 0) return null;

        return (
            <div className="mb-3">
                <h6 className="text-uppercase text-muted small fw-bold mb-2">{title}</h6>
                <div className="d-flex flex-wrap gap-2">
                    {filtered.map(item => {
                        const isSelected = selectedIds[type].includes(item.id);
                        const label = item.title || item.degree || item.name;
                        const subLabel = item.company || item.institution || '';

                        return (
                            <div
                                key={item.id}
                                onClick={() => onToggle(type, item.id)}
                                className={`
                                    px-3 py-2 rounded border cursor-pointer user-select-none d-flex align-items-center gap-2 transition-all
                                    ${isSelected 
                                        ? 'bg-primary-subtle border-primary text-primary-emphasis' 
                                        : 'bg-white hover:bg-light'}
                                `}
                            >
                                {isSelected ? <Check size={14} /> : <Plus size={14} className="text-muted" />}
                                <Icon size={14} className={isSelected ? 'text-primary' : 'text-muted'} />
                                <div className="d-flex flex-column" style={{lineHeight: 1}}>
                                    <span className="small fw-medium">{label}</span>
                                    {subLabel && <span className="text-muted" style={{fontSize: '0.7rem'}}>{subLabel}</span>}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>
        );
    };

    return (
        <div className="p-3 bg-light rounded border">
            <div className="mb-3 position-relative">
                <Search className="position-absolute text-muted" size={14} style={{top: '10px', left: '10px'}} />
                <input 
                    type="text" 
                    className="form-control form-control-sm ps-4"
                    placeholder="Search contexts..."
                    value={searchTerm}
                    onChange={e => setSearchTerm(e.target.value)}
                />
            </div>
            
            <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {renderSection('Experience', allExperiences, 'experiences', Briefcase)}
                {renderSection('Education', allEducation, 'education', BookOpen)}
                {renderSection('Hobbies', allHobbies, 'hobbies', Smile)}
            </div>
        </div>
    );
};

export default ContextLinker;