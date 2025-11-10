// frontend/src/components/applications/JobPreviewModal.jsx
import React from 'react';

/**
 * A read-only modal to display the full job description details.
 */
const JobPreviewModal = ({ job, isOpen, onClose }) => {
    if (!isOpen || !job) return null;

    return (
        <div
            className="modal"
            style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}
            onClick={onClose}
        >
            <div
                className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable"
                onClick={(e) => e.stopPropagation()}
            >
                <div className="modal-content">
                    <div className="modal-header">
                        <div>
                            <h5 className="modal-title mb-0">{job.title}</h5>
                            <small className="text-muted">{job.company}</small>
                        </div>
                        <button
                            type="button"
                            className="btn-close"
                            onClick={onClose}
                        ></button>
                    </div>
                    <div className="modal-body">
                        {/* --- Job Details Section --- */}
                        <div className="mb-3">
                            <div className="row g-2 small mb-3">
                                {job.location && (
                                    <div className="col-auto">
                                        <span className="badge bg-light text-dark border me-1">Location</span>
                                        {job.location}
                                    </div>
                                )}
                                {job.salary_range && (
                                    <div className="col-auto">
                                        <span className="badge bg-light text-dark border me-1">Salary</span>
                                        {job.salary_range}
                                    </div>
                                )}
                                {job.application_end_date && (
                                    <div className="col-auto">
                                        <span className="badge bg-light text-dark border me-1">Apply By</span>
                                        {job.application_end_date}
                                    </div>
                                )}
                            </div>

                            {job.job_url && (
                                <div className="mb-3">
                                    <a
                                        href={job.job_url}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="btn btn-outline-info btn-sm"
                                    >
                                        View Original Job Posting
                                    </a>
                                </div>
                            )}

                            {job.notes && (
                                <div className="alert alert-secondary small">
                                    <strong>Your Notes:</strong>
                                    <p className="mb-0" style={{ whiteSpace: 'pre-wrap' }}>{job.notes}</p>
                                </div>
                            )}

                            {job.features && job.features.length > 0 && (
                                <div>
                                    <h6 className="small fw-bold text-muted">Requirements & Features</h6>
                                    <ul className="list-group list-group-flush">
                                        {job.features.map(feature => (
                                            <li key={feature.id} className="list-group-item d-flex align-items-center p-1">
                                                <span className={`badge bg-secondary me-2 text-capitalize`} style={{ width: '100px', textAlign: 'center' }}>
                                                    {feature.type.replace('_', ' ')}
                                                </span>
                                                <small>{feature.description}</small>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    </div>
                    <div className="modal-footer">
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={onClose}
                        >
                            Close
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default JobPreviewModal;