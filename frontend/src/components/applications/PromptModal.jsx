// frontend/src/components/applications/PromptModal.jsx
import React, { useState } from 'react';

const PromptModal = ({ isOpen, jsonString, onClose }) => {
    const [activeTab, setActiveTab] = useState('manual');
    const [copyPromptText, setCopyPromptText] = useState('Copy Instructions');
    
    // The instructions for the user to paste into their AI of choice
    const userInstructions = `You are an expert career assistant and copywriter. Your task is to generate a tailored document (like a CV or cover letter) based on the structured JSON payload I provide below. Please follow all instructions, use the provided data, and generate only the requested document.`;

    // --- Helper Functions ---

    // Handles downloading the JSON payload
    const handleDownloadJson = () => {
        try {
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            // Differentiate file name based on prompt type if possible
            const isCV = jsonString.includes("generate-cv-prompt");
            a.download = isCV ? 'cv_ai_prompt.json' : 'cover_letter_ai_prompt.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (error) {
            console.error("Failed to download JSON:", error);
            alert("Error downloading file.");
        }
    };

    // Handles copying the user-facing instructions
    const handleCopyInstructions = () => {
        navigator.clipboard.writeText(userInstructions).then(() => {
            setCopyPromptText('Copied!');
            setTimeout(() => setCopyPromptText('Copy Instructions'), 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            alert('Failed to copy instructions.');
        });
    };

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
                        {/* --- Tab Navigation --- */}
                        <ul className="nav nav-tabs mb-3">
                            <li className="nav-item">
                                <button 
                                    className={`nav-link ${activeTab === 'manual' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('manual')}
                                >
                                    Manual Walkthrough
                                </button>
                            </li>
                            <li className="nav-item">
                                <button 
                                    className={`nav-link ${activeTab === 'service' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('service')}
                                >
                                    AI Service (Coming Soon)
                                </button>
                            </li>
                        </ul>

                        {/* --- Tab Content --- */}
                        <div className="tab-content">
                            
                            {/* === Manual Walkthrough Tab === */}
                            <div className={`tab-pane fade ${activeTab === 'manual' ? 'show active' : ''}`}>
                                <h6 className="h5">Step 1: Copy the Prompt Instructions</h6>
                                <p className="small text-muted">
                                    Copy the instructions below and paste them into your preferred AI service (e.g., ChatGPT, Claude, etc.).
                                </p>
                                <div className="card bg-light border mb-3">
                                    <div className="card-body">
                                        <pre className="p-0 m-0" style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                                            {userInstructions}
                                        </pre>
                                    </div>
                                    <div className="card-footer text-end p-2">
                                        <button className="btn btn-sm btn-outline-primary" onClick={handleCopyInstructions}>
                                            {copyPromptText}
                                        </button>
                                    </div>
                                </div>
                                
                                <h6 className="h5">Step 2: Get the JSON Payload</h6>
                                <p className="small text-muted">
                                    This is the structured JSON data. Copy and paste this payload *below* the instructions in your AI service's prompt window.
                                </p>
                                <pre 
                                    className="bg-dark text-light p-3 rounded"
                                    style={{ maxHeight: '300px' }}
                                >
                                    <code>
                                        {jsonString}
                                    </code>
                                </pre>
                                <button className="btn btn-sm btn-info" onClick={handleDownloadJson}>
                                    Download JSON Payload
                                </button>
                            </div>

                            {/* === AI Service Tab === */}
                            <div className={`tab-pane fade ${activeTab === 'service' ? 'show active' : ''}`}>
                                <p className="text-muted fst-italic">
                                    This feature will allow you to send the prompt directly to an AI model from here. (Coming Soon)
                                </p>
                                
                                <div className="mb-3">
                                    <label htmlFor="ai-service" className="form-label">GenAI Service</label>
                                    <select id="ai-service" className="form-select" disabled>
                                        <option>OpenAI (e.g., gpt-4o)</option>
                                    </select>
                                </div>
                                
                                <div className="mb-3">
                                    <label htmlFor="ai-key" className="form-label">API Key</label>
                                    <input 
                                        type="password" 
                                        id="ai-key" 
                                        className="form-control" 
                                        placeholder="Enter your API Key..." 
                                        disabled 
                                    />
                                </div>
                                
                                <button className="btn btn-primary" type="button" disabled>
                                    Send to AI Model
                                </button>
                            </div>
                        </div>
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