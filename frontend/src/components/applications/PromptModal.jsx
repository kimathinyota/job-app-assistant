// frontend/src/components/applications/PromptModal.jsx
import React from 'react';

const PromptModal = ({ isOpen, jsonString, onClose }) => {
    if (!isOpen) return null;

    return (
        <div className="modal" style={{ display: 'block', backgroundColor: 'rgba(0,0,0,0.5)' }}>
            <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <div className="modal-content">
                    <div className="modal-header">
                        <h5 className="modal-title">Generated AI Prompt</h5>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>
                    <div className="modal-body">
                        <p className="small text-muted">
                            This is the structured JSON payload to be sent to an AI service.
                        </p>
                        <pre 
                            className="bg-dark text-light p-3 rounded"
                            style={{ maxHeight: '60vh' }}
                        >
                            <code>
                                {jsonString}
                            </code>
                        </pre>
                    </div>
                    <div className="modal-footer">
                        <button type="button" className="btn btn-secondary" onClick={onClose}>Close</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default PromptModal;