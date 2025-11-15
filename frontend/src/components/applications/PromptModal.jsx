// frontend/src/components/applications/PromptModal.jsx
import React, { useState } from 'react';
import { 
    Copy, 
    Download, 
    ExternalLink, 
    Check, 
    Bot, 
    FileJson, 
    X, 
    ClipboardCopy,
    MessageSquare,
    FileText
} from 'lucide-react';

const PromptModal = ({ isOpen, jsonString, onClose }) => {
    const [activeTab, setActiveTab] = useState('manual');
    const [copyTextState, setCopyTextState] = useState('Copy'); // 'Copy' | 'Copied!'
    const [copyJsonState, setCopyJsonState] = useState('Copy'); 

    // Main instructions for AI
    const userInstructions = `You are an expert career assistant and copywriter. Your task is to generate a tailored document (like a CV or cover letter) based on the structured JSON payload I provide below. Please follow all instructions, use the provided data, and generate only the requested document.`;

    // --- Helpers ---
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

    const copyToClipboard = (text, stateSetter) => {
        navigator.clipboard.writeText(text).then(() => {
            stateSetter('Copied!');
            setTimeout(() => stateSetter('Copy'), 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
        });
    };

    const handleCopyPromptAndJson = () => {
        const fullPrompt = `${userInstructions}\n\n${jsonString}`;
        navigator.clipboard.writeText(fullPrompt).then(() => {
            alert('✅ Prompt and JSON copied to clipboard!\n\nNext step: Paste it into ChatGPT.');
        }).catch(err => {
            alert('Failed to copy text.');
        });
    };

    const handleOpenChatGPT = () => {
        window.open('https://chat.openai.com/', '_blank');
    };

    if (!isOpen) return null;

    return (
        <div className="modal show d-block" style={{ backgroundColor: 'rgba(15, 23, 42, 0.6)', backdropFilter: 'blur(4px)' }} tabIndex="-1">
            <div className="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                <div className="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                    
                    {/* Header */}
                    <div className="modal-header border-bottom-0 bg-white pb-0">
                        <div className="d-flex align-items-center gap-3">
                            <div className="bg-primary bg-opacity-10 p-2 rounded-circle text-primary">
                                <Bot size={24} />
                            </div>
                            <div>
                                <h5 className="modal-title fw-bold text-dark mb-0">AI Prompt Generator</h5>
                                <p className="text-muted small mb-0">Generate your tailored document with AI</p>
                            </div>
                        </div>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>

                    <div className="modal-body p-4">
                        
                        {/* Segmented Control Tabs */}
                        <div className="d-flex justify-content-center mb-4">
                            <div className="bg-light p-1 rounded-pill border d-inline-flex shadow-sm">
                                <button 
                                    className={`btn btn-sm rounded-pill px-4 d-flex align-items-center gap-2 border-0 transition-all ${activeTab === 'manual' ? 'bg-white text-primary shadow-sm fw-bold' : 'text-muted hover-text-dark'}`}
                                    onClick={() => setActiveTab('manual')}
                                >
                                    <FileText size={14}/> Manual Copy
                                </button>
                                <button 
                                    className={`btn btn-sm rounded-pill px-4 d-flex align-items-center gap-2 border-0 transition-all ${activeTab === 'service' ? 'bg-white text-primary shadow-sm fw-bold' : 'text-muted hover-text-dark'}`}
                                    onClick={() => setActiveTab('service')}
                                >
                                    <ExternalLink size={14}/> Quick Launch
                                </button>
                            </div>
                        </div>

                        {/* === Manual Tab === */}
                        {activeTab === 'manual' && (
                            <div className="animate-fade-in">
                                {/* Step 1: Instructions */}
                                <div className="mb-4">
                                    <div className="d-flex justify-content-between align-items-center mb-2 px-1">
                                        <h6 className="fw-bold small text-muted text-uppercase mb-0">1. System Instructions</h6>
                                        <button 
                                            className="btn btn-sm btn-light text-primary fw-medium d-flex align-items-center gap-1"
                                            onClick={() => copyToClipboard(userInstructions, setCopyTextState)}
                                        >
                                            {copyTextState === 'Copied!' ? <Check size={14}/> : <Copy size={14}/>}
                                            {copyTextState}
                                        </button>
                                    </div>
                                    <div className="bg-light border rounded-3 p-3 position-relative">
                                        <p className="mb-0 font-monospace small text-muted" style={{whiteSpace: 'pre-wrap', lineHeight: '1.5'}}>
                                            {userInstructions}
                                        </p>
                                    </div>
                                </div>

                                {/* Step 2: JSON Payload */}
                                <div>
                                    <div className="d-flex justify-content-between align-items-center mb-2 px-1">
                                        <h6 className="fw-bold small text-muted text-uppercase mb-0">2. Data Payload</h6>
                                        <div className="d-flex gap-2">
                                            <button 
                                                className="btn btn-sm btn-light text-dark fw-medium d-flex align-items-center gap-1"
                                                onClick={handleDownloadJson}
                                            >
                                                <Download size={14}/> JSON
                                            </button>
                                            <button 
                                                className="btn btn-sm btn-light text-primary fw-medium d-flex align-items-center gap-1"
                                                onClick={() => copyToClipboard(jsonString, setCopyJsonState)}
                                            >
                                                {copyJsonState === 'Copied!' ? <Check size={14}/> : <Copy size={14}/>}
                                                {copyJsonState}
                                            </button>
                                        </div>
                                    </div>
                                    <div className="bg-dark rounded-3 p-3 overflow-hidden position-relative">
                                        <div className="position-absolute top-0 end-0 p-2">
                                            <span className="badge bg-white bg-opacity-10 text-white border border-white border-opacity-10">JSON</span>
                                        </div>
                                        <pre className="m-0 custom-scroll" style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '0.75rem', color: '#a5b3ce' }}>
                                            <code>{jsonString}</code>
                                        </pre>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* === Service Tab === */}
                        {activeTab === 'service' && (
                            <div className="animate-fade-in py-3">
                                <div className="text-center mb-4">
                                    <div className="bg-success bg-opacity-10 text-success p-3 rounded-circle d-inline-flex mb-3">
                                        <Bot size={32}/>
                                    </div>
                                    <h5 className="fw-bold">Ready to Generate?</h5>
                                    <p className="text-muted small mx-auto" style={{maxWidth: '400px'}}>
                                        We'll copy the entire prompt (Instructions + Data) to your clipboard. 
                                        Then, simply paste it into ChatGPT to get your result.
                                    </p>
                                </div>

                                <div className="card bg-light border-0 p-3 mb-4">
                                    <div className="d-flex gap-3 align-items-center">
                                        <div className="bg-white p-2 rounded-circle shadow-sm text-primary">
                                            <ClipboardCopy size={20}/>
                                        </div>
                                        <div>
                                            <h6 className="fw-bold text-dark mb-0">Full Context Copy</h6>
                                            <p className="small text-muted mb-0">Includes role instructions & structured JSON data.</p>
                                        </div>
                                        <button 
                                            className="btn btn-white border shadow-sm ms-auto fw-medium text-primary"
                                            onClick={handleCopyPromptAndJson}
                                        >
                                            Copy All
                                        </button>
                                    </div>
                                </div>

                                <div className="d-grid">
                                    <button 
                                        className="btn btn-success btn-lg shadow-sm d-flex align-items-center justify-content-center gap-2"
                                        onClick={handleOpenChatGPT}
                                    >
                                        Open ChatGPT <ExternalLink size={18}/>
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="modal-footer border-top-0 bg-light bg-opacity-50">
                        <button type="button" className="btn btn-outline-secondary px-4" onClick={onClose}>Close</button>
                    </div>
                </div>
            </div>
            
            <style>{`
                .custom-scroll::-webkit-scrollbar { width: 6px; }
                .custom-scroll::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); }
                .custom-scroll::-webkit-scrollbar-thumb { background-color: rgba(255,255,255,0.2); border-radius: 10px; }
                .transition-all { transition: all 0.2s ease; }
            `}</style>
        </div>
    );
};

export default PromptModal;