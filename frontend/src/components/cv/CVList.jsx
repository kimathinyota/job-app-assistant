import React from 'react';
import { FileText, Plus, CheckCircle, Loader2 } from 'lucide-react'; 

const CVSelector = ({ cvs, onSelect, selectedCVId, onCreate, activeImport, onImportClick }) => {
    return (
        <div className="d-flex align-items-center justify-content-between mb-4 border-bottom pb-2">
            <div className="d-flex gap-2 overflow-auto py-2">
                
                {/* --- THE GHOST TAB (Background Process) --- */}
                {activeImport && (
                    <button 
                        onClick={onImportClick}
                        className={`btn d-flex align-items-center gap-2 px-3 py-2 rounded-pill shadow-sm ${
                            activeImport.status === 'error' ? 'bg-danger-subtle text-danger border-danger' : 'bg-white border-primary text-primary'
                        }`}
                        style={{ whiteSpace: 'nowrap' }}
                    >
                        {activeImport.status === 'processing' ? (
                            <Loader2 size={16} className="animate-spin" />
                        ) : (
                            <span className="fw-bold">!</span>
                        )}
                        <span className="fw-bold small">
                            {activeImport.status === 'error' ? 'Import Failed' : `Importing: ${activeImport.name}`}
                        </span>
                    </button>
                )}

                {/* --- Normal Tabs --- */}
                {cvs && cvs.map(cv => {
                    const isActive = selectedCVId === cv.id;
                    return (
                        <button 
                            key={cv.id} 
                            onClick={() => onSelect(cv.id)} 
                            className={`btn d-flex align-items-center gap-2 px-3 py-2 rounded-pill border ${isActive ? 'btn-primary' : 'btn-light text-muted'}`}
                        >
                            {isActive ? <CheckCircle size={16} /> : <FileText size={16} />}
                            <span className="fw-medium">{cv.name}</span>
                        </button>
                    );
                })}
            </div>
            
            <div className="ps-3 border-start ms-2">
                <button onClick={onCreate} className="btn btn-outline-primary rounded-pill px-3 gap-2 d-flex align-items-center">
                    <Plus size={16} /> New CV
                </button>
            </div>
        </div>
    );
};
export default CVSelector;