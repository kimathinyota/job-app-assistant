import React from 'react';
import { FileText, Plus, Trash2, Clock, MoreVertical, Loader2 } from 'lucide-react';
import { timeAgo } from '../../utils/cvHelpers'; // Import the new helper

const CVCard = ({ cv, onSelect, onDelete }) => (
    <div className="col-md-4 col-lg-3 d-flex align-items-stretch">
        <div 
            className="card border-0 shadow-sm w-100 hover-lift transition-all group" 
            style={{cursor: 'pointer'}}
            onClick={() => onSelect(cv.id)}
        >
            <div className="card-body d-flex flex-column p-4">
                {/* Icon & Title */}
                <div className="d-flex justify-content-between align-items-start mb-3">
                    <div className="p-3 bg-primary bg-opacity-10 rounded-circle text-primary">
                        <FileText size={24} />
                    </div>
                    <div className="dropdown" onClick={(e) => e.stopPropagation()}>
                        <button className="btn btn-link text-muted p-0" type="button" data-bs-toggle="dropdown">
                            <MoreVertical size={18} />
                        </button>
                        <ul className="dropdown-menu dropdown-menu-end border-0 shadow-sm">
                            <li>
                                <button className="dropdown-item text-danger d-flex align-items-center gap-2" onClick={() => onDelete(cv.id)}>
                                    <Trash2 size={14} /> Delete
                                </button>
                            </li>
                        </ul>
                    </div>
                </div>

                <h5 className="fw-bold text-dark mb-1 text-truncate" title={cv.name}>
                    {cv.name}
                </h5>
                <p className="text-muted small mb-3">Master CV</p>

                {/* Summary Preview */}
                <p className="text-secondary small mb-4 flex-grow-1" style={{
                    display: '-webkit-box',
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: 'vertical',
                    overflow: 'hidden',
                    opacity: 0.75
                }}>
                    {cv.summary || "No summary provided. Click to add details..."}
                </p>

                {/* Footer */}
                <div className="d-flex align-items-center justify-content-between pt-3 border-top mt-auto">
                    <span className="badge bg-light text-secondary border fw-normal">
                        Click to Edit
                    </span>
                    <span className="text-muted extra-small d-flex align-items-center gap-1" style={{fontSize: '0.75rem'}}>
                        <Clock size={12} /> {timeAgo(cv.updated_at) || 'Recent'}
                    </span>
                </div>
            </div>
        </div>
    </div>
);

const NewCVCard = ({ onClick }) => (
    <div className="col-md-4 col-lg-3 d-flex align-items-stretch">
        <div 
            onClick={onClick}
            className="card border border-2 border-dashed border-secondary bg-light w-100 d-flex align-items-center justify-content-center text-center p-4 hover-scale transition-all"
            style={{cursor: 'pointer', minHeight: '280px', borderStyle: 'dashed'}}
        >
            <div className="text-muted">
                <div className="mb-3 mx-auto p-3 rounded-circle bg-white shadow-sm d-inline-block text-primary">
                    <Plus size={32} />
                </div>
                <h6 className="fw-bold mb-1">Create New CV</h6>
                <p className="small mb-0 opacity-75">Start from scratch</p>
            </div>
        </div>
    </div>
);

const ImportingCard = ({ task }) => (
    <div className="col-md-4 col-lg-3 d-flex align-items-stretch">
        <div className="card border-0 shadow-sm w-100 bg-light">
            <div className="card-body d-flex flex-column align-items-center justify-content-center text-center p-4">
                {task.status === 'error' ? (
                    <>
                        <div className="text-danger mb-3"><FileText size={32}/></div>
                        <h6 className="fw-bold text-danger">Import Failed</h6>
                        <p className="small text-muted">{task.name}</p>
                    </>
                ) : (
                    <>
                        <div className="text-primary mb-3"><Loader2 size={32} className="animate-spin"/></div>
                        <h6 className="fw-bold text-dark">Importing...</h6>
                        <p className="small text-muted mb-0">{task.name}</p>
                        <span className="badge bg-white text-primary border mt-3">Processing</span>
                    </>
                )}
            </div>
        </div>
    </div>
);

const CVSelector = ({ cvs, onSelect, onDelete, onCreate, activeImport }) => {
    return (
        <div className="row g-4 animate-fade-in">
            <style>{`
                .hover-lift:hover { transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.1) !important; }
                .hover-scale:hover { transform: scale(1.02); background-color: #f8f9fa !important; border-color: var(--bs-primary) !important; }
            `}</style>

            {activeImport && <ImportingCard task={activeImport} />}
            <NewCVCard onClick={onCreate} />
            
            {cvs.map(cv => (
                <CVCard 
                    key={cv.id} 
                    cv={cv} 
                    onSelect={onSelect} 
                    onDelete={onDelete} 
                />
            ))}
        </div>
    );
};

export default CVSelector;