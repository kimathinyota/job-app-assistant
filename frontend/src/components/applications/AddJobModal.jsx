// frontend/src/components/applications/AddJobModal.jsx
import React, { useState } from 'react';

const AddJobModal = ({ isOpen, onClose, onSubmit }) => {
    const [title, setTitle] = useState('');
    const [company, setCompany] = useState('');

    if (!isOpen) return null;

    const handleSubmit = (e) => {
        e.preventDefault();
        onSubmit(title, company);
        setTitle('');
        setCompany('');
    };

    return (
        <div className="modal" style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}>
            <div className="modal-dialog modal-dialog-centered">
                <div className="modal-content">
                    <form onSubmit={handleSubmit}>
                        <div className="modal-header">
                            <h5 className="modal-title">Add New Job</h5>
                            <button type="button" className="btn-close" onClick={onClose}></button>
                        </div>
                        <div className="modal-body">
                            <div className="mb-3">
                                <label htmlFor="jobTitle" className="form-label">Job Title</label>
                                <input
                                    type="text"
                                    className="form-control"
                                    id="jobTitle"
                                    value={title}
                                    onChange={(e) => setTitle(e.target.value)}
                                    required
                                />
                            </div>
                            <div className="mb-3">
                                <label htmlFor="jobCompany" className="form-label">Company</label>
                                <input
                                    type="text"
                                    className="form-control"
                                    id="jobCompany"
                                    value={company}
                                    onChange={(e) => setCompany(e.target.value)}
                                    required
                                />
                            </div>
                            <div className="form-text">
                                You can add requirements and details in the next step.
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button type="button" className="btn btn-secondary" onClick={onClose}>Close</button>
                            <button type="submit" className="btn btn-primary">Save Job</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default AddJobModal;