import React from 'react';
import { FileText, Plus, Trash2, Clock, MoreVertical, Loader2, Star, CheckCircle, ShieldCheck } from 'lucide-react';
import { timeAgo } from '../../utils/cvHelpers';

const CVCard = ({ cv, isPrimary, onSelect, onDelete, onSetPrimary }) => {
    
    // 1. Subtle Difference: Primary gets a medium shadow, others get small
    // Primary gets a generic white background, but with a top border accent
    const cardStyle = isPrimary 
        ? { borderTop: '4px solid var(--bs-primary)', transform: 'translateY(-2px)' } 
        : { borderTop: '4px solid transparent' };
        
    const shadowClass = isPrimary ? "shadow" : "shadow-sm";

    return (
        <div className="col-md-4 col-lg-3 d-flex align-items-stretch">
            <div 
                className={`card w-100 hover-lift transition-all group border-0 ${shadowClass}`} 
                style={{cursor: 'pointer', ...cardStyle}}
                onClick={() => onSelect(cv.id)}
            >
                <div className="card-body d-flex flex-column p-4 position-relative">
                    
                    {/* Header Section */}
                    <div className="d-flex justify-content-between align-items-start mb-3">
                        {/* ICON: If Primary, use a Star with a golden/primary tint. If not, standard file icon */}
                        <div 
                            className={`p-3 rounded-circle d-flex align-items-center justify-content-center transition-all`} 
                            style={{
                                width: '50px', 
                                height: '50px', 
                                backgroundColor: isPrimary ? 'rgba(var(--bs-primary-rgb), 0.1)' : '#f8f9fa',
                                color: isPrimary ? 'var(--bs-primary)' : '#6c757d'
                            }}
                        >
                            {isPrimary ? <Star size={24} fill="currentColor" className="opacity-75" /> : <FileText size={24} />}
                        </div>
                        
                        {/* Dropdown Menu */}
                        <div className="dropdown" onClick={(e) => e.stopPropagation()}>
                            <button className="btn btn-link text-muted p-0" type="button" data-bs-toggle="dropdown">
                                <MoreVertical size={18} />
                            </button>
                            <ul className="dropdown-menu dropdown-menu-end border-0 shadow-sm">
                                {!isPrimary && (
                                    <li>
                                        <button 
                                            className="dropdown-item d-flex align-items-center gap-2 text-primary fw-medium" 
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onSetPrimary(cv.id);
                                            }}
                                        >
                                            <CheckCircle size={14} /> Set as Default
                                        </button>
                                    </li>
                                )}
                                
                                {!isPrimary && <li><hr className="dropdown-divider"/></li>}

                                {isPrimary ? (
                                    <li>
                                        <span className="dropdown-item text-muted small fst-italic cursor-not-allowed">
                                            Default CV cannot be deleted
                                        </span>
                                    </li>
                                ) : (
                                    <li>
                                        <button 
                                            className="dropdown-item text-danger d-flex align-items-center gap-2" 
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onDelete(cv.id);
                                            }}
                                        >
                                            <Trash2 size={14} /> Delete
                                        </button>
                                    </li>
                                )}
                            </ul>
                        </div>
                    </div>

                    {/* SUBTLE INDICATOR: Text + Icon next to title */}
                    <div className="mb-1 d-flex align-items-center gap-2">
                        <h5 className="fw-bold text-dark mb-0 text-truncate" title={cv.name} style={{maxWidth: '85%'}}>
                            {cv.name}
                        </h5>
                        {/* Small Verified Badge */}
                        {isPrimary && (
                            <div className="text-primary" title="Default CV">
                                <ShieldCheck size={18} />
                            </div>
                        )}
                    </div>
                    
                    {/* Primary Label (Small text instead of big badge) */}
                    {isPrimary ? (
                         <p className="text-primary small mb-3 fw-medium" style={{fontSize: '0.75rem'}}>
                            Default Profile
                         </p>
                    ) : (
                         <p className="text-muted small mb-3" style={{fontSize: '0.75rem'}}>
                            Secondary Profile
                         </p>
                    )}

                    <p className="text-secondary small mb-4 flex-grow-1" style={{
                        display: '-webkit-box',
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: 'vertical',
                        overflow: 'hidden',
                        opacity: 0.8,
                        fontSize: '0.85rem',
                        lineHeight: '1.5'
                    }}>
                        {cv.summary || "No summary provided. Click to add details..."}
                    </p>

                    {/* Footer */}
                    <div className="d-flex align-items-center justify-content-between pt-3 border-top mt-auto">
                        <span className="text-muted extra-small d-flex align-items-center gap-1" style={{fontSize: '0.75rem'}}>
                            <Clock size={12} /> {timeAgo(cv.updated_at) || 'Just now'}
                        </span>
                        
                        <span className={`badge ${isPrimary ? 'bg-primary bg-opacity-10 text-primary' : 'bg-light text-secondary'} border fw-normal`}>
                            {isPrimary ? 'Active' : 'Edit'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

const NewCVCard = ({ onClick }) => (
    <div className="col-md-4 col-lg-3 d-flex align-items-stretch">
        <div 
            onClick={onClick}
            className="card border border-2 border-dashed border-secondary bg-light w-100 d-flex align-items-center justify-content-center text-center p-4 hover-scale transition-all"
            style={{cursor: 'pointer', minHeight: '320px', borderStyle: 'dashed'}}
        >
            <div className="text-muted">
                <div className="mb-3 mx-auto p-3 rounded-circle bg-white shadow-sm d-inline-block text-primary">
                    <Plus size={32} />
                </div>
                <h6 className="fw-bold mb-1">Upload CV</h6>
                <p className="small mb-0 opacity-75">PDF or Word</p>
            </div>
        </div>
    </div>
);

const ImportingCard = ({ name }) => (
    <div className="col-md-4 col-lg-3 d-flex align-items-stretch">
        <div className="card border-0 shadow-sm w-100 bg-white position-relative overflow-hidden" style={{minHeight: '320px'}}>
            <div className="position-absolute top-0 start-0 w-100 h-100 bg-primary bg-opacity-10 animate-pulse" style={{ zIndex: 0 }}></div>
            <div className="card-body d-flex flex-column align-items-center justify-content-center text-center p-4 position-relative" style={{ zIndex: 1 }}>
                <div className="mb-3 position-relative">
                    <div className="spinner-border text-primary" role="status" style={{width: '3rem', height: '3rem'}}></div>
                    <div className="position-absolute top-50 start-50 translate-middle text-primary">
                       <Loader2 size={20} className="opacity-50"/> 
                    </div>
                </div>
                <h6 className="fw-bold text-dark mb-1">Importing CV...</h6>
                <p className="small text-muted mb-0">Analyzing document...</p>
                <p className="extra-small text-muted mt-3 mb-0 text-truncate" style={{maxWidth: '100%'}}>
                    {name}
                </p>
            </div>
        </div>
    </div>
);

const CVSelector = ({ cvs, primaryCvId, onSelect, onDelete, onCreate, onSetPrimary }) => {
    
    // Safety check
    const safeCvs = cvs || [];

    // Sorting logic
    const sortedCvs = [...safeCvs].sort((a, b) => {
        if (a.id === primaryCvId) return -1;
        if (b.id === primaryCvId) return 1;
        return new Date(b.updated_at) - new Date(a.updated_at);
    });

    return (
        <div className="container-fluid p-0">
             <style>{`
                .hover-lift:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important; }
                .hover-scale:hover { transform: scale(1.02); background-color: #f8f9fa !important; border-color: var(--bs-primary) !important; }
                
                @keyframes pulse-bg { 0% { opacity: 0.05; } 50% { opacity: 0.15; } 100% { opacity: 0.05; } }
                .animate-pulse { animation: pulse-bg 2s infinite; }

                @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
                .animate-fade-in { animation: fadeIn 0.4s ease-out forwards; }
            `}</style>

            <div className="row g-4 animate-fade-in">
                <NewCVCard onClick={onCreate} />
                
                {sortedCvs.map(cv => {
                    if (cv.is_importing) {
                        return <ImportingCard key={cv.id} name={cv.name} />;
                    }
                    
                    return (
                        <CVCard 
                            key={cv.id} 
                            cv={cv}
                            isPrimary={cv.id === primaryCvId}
                            onSelect={onSelect} 
                            onDelete={onDelete} 
                            onSetPrimary={onSetPrimary}
                        />
                    );
                })}
            </div>
        </div>
    );
};

export default CVSelector;