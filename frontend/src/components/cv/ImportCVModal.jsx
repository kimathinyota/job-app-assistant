import React, { useState, useEffect } from 'react';
import { UploadCloud, FileText, CheckCircle, AlertCircle, Clipboard } from 'lucide-react';
import { importCV } from '../../api/cvClient';

// --- PDF.js Setup (Vite Compatible) ---
import * as pdfjsLib from 'pdfjs-dist';

// CRITICAL FIX: Import worker directly from node_modules using Vite's ?url suffix
// This prevents 404 errors and CORS issues with CDNs
import pdfWorker from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorker;
// --------------------------------------

const PROCESSING_STEPS = [
    { pct: 10, msg: "Initializing LLM parser..." },
    { pct: 25, msg: "Extracting contact info & summary..." },
    { pct: 45, msg: "Structuring work history & dates..." },
    { pct: 60, msg: "Identifying technical skills..." },
    { pct: 80, msg: "Linking achievements to experience..." },
    { pct: 95, msg: "Finalizing relational database..." }
];

const ImportCVModal = ({ onClose, onSuccess }) => {
    const [activeTab, setActiveTab] = useState('paste'); // 'paste' or 'file'
    const [cvName, setCvName] = useState('');
    const [textData, setTextData] = useState('');
    
    // Loading State
    const [isLoading, setIsLoading] = useState(false);
    const [isExtracting, setIsExtracting] = useState(false);
    const [progress, setProgress] = useState(0);
    const [currentStep, setCurrentStep] = useState(PROCESSING_STEPS[0].msg);
    const [error, setError] = useState(null);

    // --- PDF Extraction Helper ---
    const extractTextFromPDF = async (file) => {
        try {
            const arrayBuffer = await file.arrayBuffer();
            // Use the locally configured worker
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            let fullText = "";

            for (let i = 1; i <= pdf.numPages; i++) {
                const page = await pdf.getPage(i);
                const textContent = await page.getTextContent();
                // Join text items with space to preserve basic layout
                const pageText = textContent.items.map(item => item.str).join(' ');
                fullText += pageText + "\n\n";
            }
            return fullText;
        } catch (err) {
            console.error("PDF Read Error details:", err);
            throw new Error("Could not parse PDF structure.");
        }
    };

    // --- File Handler ---
    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setError(null);

        // 1. Handle PDF
        if (file.type === "application/pdf") {
            setIsExtracting(true);
            try {
                const extractedText = await extractTextFromPDF(file);
                setTextData(extractedText);
                setActiveTab('paste'); // Switch tab so user can see/edit the text
            } catch (err) {
                console.error("PDF Extraction failed", err);
                setError("Could not auto-extract text. Please copy/paste manually.");
            } finally {
                setIsExtracting(false);
            }
        } 
        // 2. Handle Text/Markdown
        else if (file.type === "text/plain" || file.name.endsWith('.md')) {
            const reader = new FileReader();
            reader.onload = (event) => {
                setTextData(event.target.result);
                setActiveTab('paste');
            };
            reader.readAsText(file);
        } 
        else {
            setError("Unsupported format. Please use PDF, TXT, or MD.");
        }
    };

    // --- Simulated Progress Engine ---
    useEffect(() => {
        if (!isLoading) return;

        let startTime = Date.now();
        const duration = 60000; // 60s fake timer
        
        const interval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            let rawPct = (elapsed / duration) * 100;
            
            if (rawPct > 95) rawPct = 95; // Cap until API returns

            setProgress(rawPct);

            const step = PROCESSING_STEPS.slice().reverse().find(s => rawPct >= s.pct);
            if (step) setCurrentStep(step.msg);

        }, 500);

        return () => clearInterval(interval);
    }, [isLoading]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!cvName.trim() || !textData.trim()) {
            setError("Please provide both a name and CV text.");
            return;
        }

        setIsLoading(true);
        setError(null);
        setProgress(5);

        try {
            // Actual API Call
            const result = await importCV(cvName, textData);
            
            // Success State
            setProgress(100);
            setCurrentStep("Success! Redirecting...");
            
            setTimeout(() => {
                onSuccess(result);
                onClose();
            }, 800);

        } catch (err) {
            console.error(err);
            setError("Import failed. Check backend logs.");
            setIsLoading(false);
        }
    };

    return (
        <>
            <div className="modal-backdrop fade show" style={{ zIndex: 1050 }}></div>
            <div className="modal fade show d-block" tabIndex="-1" style={{ zIndex: 1055 }}>
                <div className="modal-dialog modal-dialog-centered modal-lg">
                    <div className="modal-content shadow-lg border-0">
                        
                        {/* Header */}
                        <div className="modal-header border-bottom-0 pb-0">
                            <div>
                                <h5 className="modal-title fw-bold">Import Existing CV</h5>
                                <p className="text-muted small mb-0">Our AI will restructure your text into a database.</p>
                            </div>
                            {!isLoading && (
                                <button type="button" className="btn-close" onClick={onClose}></button>
                            )}
                        </div>

                        <div className="modal-body pt-4">
                            {isLoading ? (
                                // --- LOADING VIEW ---
                                <div className="text-center py-5">
                                    <div className="mb-4 position-relative d-inline-block">
                                        <div className="spinner-border text-primary" role="status" style={{width: '4rem', height: '4rem'}}></div>
                                        <div className="position-absolute top-50 start-50 translate-middle">
                                            <CpuIcon />
                                        </div>
                                    </div>
                                    
                                    <h4 className="fw-bold text-dark">{Math.round(progress)}%</h4>
                                    <p className="text-primary fw-medium animate-pulse">{currentStep}</p>
                                    
                                    <div className="progress mt-3 mx-auto" style={{ height: '8px', maxWidth: '400px' }}>
                                        <div 
                                            className="progress-bar progress-bar-striped progress-bar-animated" 
                                            role="progressbar" 
                                            style={{ width: `${progress}%` }}
                                        ></div>
                                    </div>
                                    <p className="text-muted small mt-3">This usually takes about a minute. Please don't close this window.</p>
                                </div>
                            ) : (
                                // --- INPUT FORM VIEW ---
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
                                        <div className="form-text">Just a label for you to identify this version later.</div>
                                    </div>

                                    {/* Tabs */}
                                    <ul className="nav nav-tabs mb-3">
                                        <li className="nav-item">
                                            <button 
                                                type="button"
                                                className={`nav-link gap-2 d-flex align-items-center ${activeTab === 'paste' ? 'active fw-bold' : ''}`}
                                                onClick={() => setActiveTab('paste')}
                                            >
                                                <Clipboard size={16}/> Paste/Edit Text
                                            </button>
                                        </li>
                                        <li className="nav-item">
                                            <button 
                                                type="button"
                                                className={`nav-link gap-2 d-flex align-items-center ${activeTab === 'file' ? 'active fw-bold' : ''}`}
                                                onClick={() => setActiveTab('file')}
                                            >
                                                <UploadCloud size={16}/> Upload File
                                            </button>
                                        </li>
                                    </ul>

                                    {/* Input Area */}
                                    <div className="mb-3">
                                        {activeTab === 'paste' ? (
                                            <div className="position-relative">
                                                <textarea 
                                                    className="form-control font-monospace small" 
                                                    rows="10" 
                                                    placeholder="Open your PDF, Ctrl+A, Ctrl+C, then Paste here..."
                                                    value={textData}
                                                    onChange={(e) => setTextData(e.target.value)}
                                                ></textarea>
                                                {textData && (
                                                    <div className="position-absolute bottom-0 end-0 p-2 text-muted small bg-light border-top border-start rounded-top">
                                                        {textData.length} chars
                                                    </div>
                                                )}
                                            </div>
                                        ) : (
                                            <div className="border rounded p-5 text-center bg-light border-dashed">
                                                {isExtracting ? (
                                                    <div className="py-4">
                                                        <div className="spinner-border text-secondary mb-2" role="status"></div>
                                                        <p className="mb-0 text-muted">Extracting text from PDF...</p>
                                                    </div>
                                                ) : (
                                                    <>
                                                        <input 
                                                            type="file" 
                                                            id="cvFileUpload" 
                                                            className="d-none" 
                                                            accept=".pdf,.txt,.md"
                                                            onChange={handleFileChange}
                                                        />
                                                        <label htmlFor="cvFileUpload" className="btn btn-outline-secondary mb-3">
                                                            Choose File (PDF, TXT, MD)
                                                        </label>
                                                        <p className="text-muted small mb-0">Supported: .pdf, .txt, .md</p>
                                                        <p className="text-muted small">We'll auto-extract text so you can review it.</p>
                                                    </>
                                                )}
                                            </div>
                                        )}
                                    </div>

                                    <div className="modal-footer border-top-0 px-0 pb-0">
                                        <button type="button" className="btn btn-light" onClick={onClose}>Cancel</button>
                                        <button 
                                            type="submit" 
                                            className="btn btn-primary d-flex align-items-center gap-2"
                                            disabled={!cvName || !textData}
                                        >
                                            <FileText size={16} /> Start Import
                                        </button>
                                    </div>
                                </form>
                            )}
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