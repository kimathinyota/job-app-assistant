import React from 'react';
import { FileText, Plus, CheckCircle } from 'lucide-react';
import { getCVDisplayName } from '../../utils/cvHelpers'; // <--- IMPORT

const CVSelector = ({ cvs, onSelect, selectedCVId, onCreate }) => {
    return (
        <div className="d-flex align-items-center justify-content-between mb-4 border-bottom pb-2">
            
            {/* Scrollable Tabs Area */}
            <div className="d-flex gap-2 overflow-auto py-2" style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}>
                {cvs && cvs.map(cv => {
                    const isActive = selectedCVId === cv.id;
                    return (
                        <button 
                            key={cv.id} 
                            onClick={() => onSelect(cv.id)} 
                            className={`btn d-flex align-items-center gap-2 px-3 py-2 rounded-pill border transition-all ${
                                isActive 
                                ? 'btn-primary shadow-sm' 
                                : 'btn-light text-muted hover:bg-slate-100'
                            }`}
                            style={{ whiteSpace: 'nowrap' }}
                        >
                            {isActive ? <CheckCircle size={16} /> : <FileText size={16} />}
                            {/* USE HELPER HERE */}
                            <span className="fw-medium">{cv.name}</span>
                        </button>
                    );
                })}
            </div>

            {/* Create New Action */}
            <div className="ps-3 border-start ms-2">
                <button 
                    onClick={onCreate}
                    className="btn btn-outline-primary d-flex align-items-center gap-2 rounded-pill px-3"
                    style={{ whiteSpace: 'nowrap' }}
                >
                    <Plus size={16} />
                    New CV
                </button>
            </div>
        </div>
    );
};

export default CVSelector;