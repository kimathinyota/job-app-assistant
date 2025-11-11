import React, { useState } from 'react';

const PromptModal = ({ isOpen, jsonString, onClose }) => {
    const [activeTab, setActiveTab] = useState('manual');
    const [copyPromptText, setCopyPromptText] = useState('Copy Instructions');

    // Main instructions for AI
    const userInstructions = `You are an expert career assistant and copywriter. Your task is to generate a tailored document (like a CV or cover letter) based on the structured JSON payload I provide below. Please follow all instructions, use the provided data, and generate only the requested document.`;

    // --- Helper Functions ---

    const handleDownloadJson = () => {
        try {
            const blob = new Blob([jsonString], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
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

    const handleCopyInstructions = () => {
        navigator.clipboard.writeText(userInstructions).then(() => {
            setCopyPromptText('Copied!');
            setTimeout(() => setCopyPromptText('Copy Instructions'), 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            alert('Failed to copy instructions.');
        });
    };

    // --- Combined Copy + Open ChatGPT Flow ---
    const handleCopyPromptAndJson = () => {
        const fullPrompt = `${userInstructions}\n\n${jsonString}`;
        navigator.clipboard.writeText(fullPrompt).then(() => {
            alert('✅ Prompt and JSON copied to clipboard!\n\nNext step: paste it into ChatGPT when the tab opens.');
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            alert('Failed to copy text to clipboard.');
        });
    };

    const handleOpenChatGPT = () => {
        window.open('https://chat.openai.com/', '_blank');
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
                                    Open in ChatGPT
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

                            {/* === Open in ChatGPT Tab === */}
                            <div className={`tab-pane fade ${activeTab === 'service' ? 'show active' : ''}`}>
                                <p className="text-muted">
                                    This option will copy your full prompt (instructions + JSON data) to your clipboard, 
                                    then open ChatGPT in a new tab. Simply <strong>paste (Ctrl+V or ⌘+V)</strong> into the chat box.
                                </p>

                                <div className="alert alert-info small">
                                    Make sure you're logged into ChatGPT before using this feature.
                                </div>

                                <div className="d-flex gap-2">
                                    <button 
                                        className="btn btn-outline-primary"
                                        onClick={handleCopyPromptAndJson}
                                    >
                                        📋 Copy Prompt + JSON
                                    </button>

                                    <button 
                                        className="btn btn-success"
                                        onClick={handleOpenChatGPT}
                                    >
                                        💬 Open ChatGPT
                                    </button>
                                </div>
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
