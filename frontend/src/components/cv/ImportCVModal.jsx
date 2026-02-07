// frontend/src/components/cv/ImportCVModal.jsx
import React, { useState, useEffect } from 'react';
import { UploadCloud, FileText, AlertCircle, Clipboard, ArrowRight } from 'lucide-react';
// We import API client functions in the Parent, so we don't import importCV here anymore.

// --- PDF.js Setup (Vite Compatible) ---
import * as pdfjsLib from 'pdfjs-dist';
// CRITICAL FIX: Import worker directly from node_modules using Vite's ?url suffix
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;
// --------------------------------------

// Visual Steps for the Progress Bar (Cosmetic)
const PROCESSING_STEPS = [
    { pct: 10, msg: "Initializing LLM parser..." },
    { pct: 25, msg: "Extracting contact info & summary..." },
    { pct: 45, msg: "Structuring work history & dates..." },
    { pct: 60, msg: "Identifying technical skills..." },
    { pct: 80, msg: "Linking achievements to experience..." },
    { pct: 95, msg: "Finalizing relational database..." }
];

const ImportCVModal = ({ onClose, onStartBackgroundImport, activeImportTask }) => {
    
    // --- MODE 1: PROGRESS MONITORING (Read-Only) ---
    // If activeImportTask is provided, we show the status of the running job.
    if (activeImportTask) {
        const isError = activeImportTask.status === 'error';
        const progress = activeImportTask.status === 'processing' ? 60 : 100; // Simulated progress if real % not available
        
        // Find current step message based on progress (or default)
        const currentStep = PROCESSING_STEPS.slice().reverse().find(s => progress >= s.pct)?.msg || "Processing...";

        return (
            <>
                <div className="modal-backdrop fade show" style={{ zIndex: 1050 }}></div>
                <div className="modal fade show d-block" tabIndex="-1" style={{ zIndex: 1055 }}>
                    <div className="modal-dialog modal-dialog-centered">
                        <div className="modal-content shadow-lg border-0">
                            <div className="modal-header border-bottom-0 pb-0">
                                <h5 className="modal-title fw-bold">
                                    {isError ? "Import Failed" : `Importing: ${activeImportTask.name}`}
                                </h5>
                                <button type="button" className="btn-close" onClick={onClose}></button>
                            </div>
                            <div className="modal-body text-center py-5">
                                {isError ? (
                                    <div className="animate-fade-in">
                                        <div className="text-danger mb-3">
                                            <AlertCircle size={48} />
                                        </div>
                                        <h4 className="fw-bold text-danger mb-2">Something went wrong</h4>
                                        <p className="text-muted small px-4">{activeImportTask.error || "Unknown server error."}</p>
                                    </div>
                                ) : (
                                    <div className="animate-fade-in">
                                        <div className="mb-4 position-relative d-inline-block">
                                            <div className="spinner-border text-primary" role="status" style={{width: '4rem', height: '4rem'}}></div>
                                            <div className="position-absolute top-50 start-50 translate-middle">
                                                <CpuIcon />
                                            </div>
                                        </div>
                                        
                                        <h4 className="fw-bold text-dark">Processing in Background...</h4>
                                        <p className="text-primary fw-medium animate-pulse mb-4">{currentStep}</p>
                                        
                                        <div className="progress mt-3 mx-auto" style={{ height: '8px', maxWidth: '300px' }}>
                                            <div 
                                                className="progress-bar progress-bar-striped progress-bar-animated" 
                                                role="progressbar" 
                                                style={{ width: '100%' }} // Indeterminate for now
                                            ></div>
                                        </div>
                                        
                                        <div className="alert alert-info mt-4 small mb-0 d-inline-flex align-items-center gap-2">
                                            <ArrowRight size={14} />
                                            You can close this window. The task will continue running.
                                        </div>
                                    </div>
                                )}
                            </div>
                            <div className="modal-footer border-top-0 justify-content-center">
                                <button className="btn btn-secondary px-4" onClick={onClose}>Close Window</button>
                            </div>
                        </div>
                    </div>
                </div>
            </>
        );
    }

    // --- MODE 2: INPUT FORM (Default) ---
    // Standard setup to select file and extract text
    
    const [activeTab, setActiveTab] = useState('paste');
    const [cvName, setCvName] = useState('');
    const [textData, setTextData] = useState('');
    const [isExtracting, setIsExtracting] = useState(false);
    const [error, setError] = useState(null);

    const extractTextFromPDF = async (file) => {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            let fullText = "";
            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                const pageText = textContent.items.map(item => item.str).join(' ');
                fullText += pageText + "\n\n";
            }
            return fullText;
        } catch (err) {
            console.error("PDF Parse Error:", err);
            throw new Error("Could not parse PDF structure.");
        }
    };

    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        setError(null);

        if (file.type === "application/pdf") {
            setIsExtracting(true);
            try {
                const extractedText = await extractTextFromPDF(file);
                setTextData(extractedText);
                setActiveTab('paste');
            } catch (err) {
                setError("Could not auto-extract text. Please copy/paste manually.");
            } finally {
                setIsExtracting(false);
            }
        } else if (file.type === "text/plain" || file.name.endsWith('.md')) {
            const reader = new FileReader();
            reader.onload = (ev) => { setTextData(ev.target.result); setActiveTab('paste'); };
            reader.readAsText(file);
        } else {
            setError("Unsupported format. Please use PDF, TXT, or MD.");
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!cvName.trim() || !textData.trim()) {
            setError("Please provide both a name and CV text.");
            return;
        }
        // Kick off the parent process (Background Task)
        onStartBackgroundImport(cvName, textData);
    };

    return (
        <>
            <div className="modal-backdrop fade show" style={{ zIndex: 1050 }}></div>
            <div className="modal fade show d-block" tabIndex="-1" style={{ zIndex: 1055 }}>
                <div className="modal-dialog modal-dialog-centered modal-lg">
                    <div className="modal-content shadow-lg border-0">
                        <div className="modal-header border-bottom-0 pb-0">
                            <div>
                                <h5 className="modal-title fw-bold">Import Existing CV</h5>
                                <p className="text-muted small mb-0">Our AI will restructure your text into a database.</p>
                            </div>
                            <button type="button" className="btn-close" onClick={onClose}></button>
                        </div>

                        <div className="modal-body pt-4">
                            <form onSubmit={handleSubmit}>
                                {error && (
                                    <div className="alert alert-danger d-flex align-items-center gap-2">
                                        <AlertCircle size={18} /> {error}
                                    </div>
                                )}

                                {/* Name Input */}
                                <div className="mb-4">
                                    <label className="form-label fw-bold small text-uppercase">Internal CV Name</label>
                                    <input 
                                        type="text" 
                                        className="form-control" 
                                        placeholder="e.g. Senior Backend - 2026" 
                                        value={cvName} 
                                        onChange={(e) => setCvName(e.target.value)} 
                                        autoFocus
                                    />
                                </div>

                                {/* Tabs */}
                                <ul className="nav nav-tabs mb-3">
                                    <li className="nav-item">
                                        <button type="button" className={`nav-link gap-2 d-flex align-items-center ${activeTab === 'paste' ? 'active fw-bold' : ''}`} onClick={() => setActiveTab('paste')}>
                                            <Clipboard size={16}/> Paste/Edit Text
                                        </button>
                                    </li>
                                    <li className="nav-item">
                                        <button type="button" className={`nav-link gap-2 d-flex align-items-center ${activeTab === 'file' ? 'active fw-bold' : ''}`} onClick={() => setActiveTab('file')}>
                                            <UploadCloud size={16}/> Upload File
                                        </button>
                                    </li>
                                </ul>

                                {/* Input Area */}
                                <div className="mb-3">
                                    {activeTab === 'paste' ? (
                                        <textarea 
                                            className="form-control font-monospace small" rows="10" 
                                            placeholder="Paste text here..." value={textData} onChange={(e) => setTextData(e.target.value)}
                                        ></textarea>
                                    ) : (
                                        <div className="border rounded p-5 text-center bg-light border-dashed">
                                            {isExtracting ? (
                                                <div className="py-4"><div className="spinner-border text-secondary mb-2" role="status"></div><p className="mb-0 text-muted">Extracting text...</p></div>
                                            ) : (
                                                <>
                                                    <input type="file" id="cvFileUpload" className="d-none" accept=".pdf,.txt,.md" onChange={handleFileChange} />
                                                    <label htmlFor="cvFileUpload" className="btn btn-outline-secondary mb-3">Choose File</label>
                                                    <p className="text-muted small mb-0">Supported: .pdf, .txt, .md</p>
                                                </>
                                            )}
                                        </div>
                                    )}
                                </div>

                                <div className="modal-footer border-top-0 px-0 pb-0">
                                    <button type="button" className="btn btn-light" onClick={onClose}>Cancel</button>
                                    <button type="submit" className="btn btn-primary d-flex align-items-center gap-2" disabled={!cvName || !textData}>
                                        <FileText size={16} /> Start Background Import
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

const CpuIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-primary">
        <rect width="16" height="16" x="4" y="4" rx="2" />
        <rect width="6" height="6" x="9" y="9" rx="1" />
        <path d="M15 2v2" /><path d="M15 20v2" /><path d="M2 15h2" /><path d="M2 9h2" /><path d="M20 15h2" /><path d="M20 9h2" /><path d="M9 2v2" /><path d="M9 20v2" />
    </svg>
);

export default ImportCVModal;