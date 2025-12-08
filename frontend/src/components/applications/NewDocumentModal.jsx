// frontend/src/components/applications/NewDocumentModal.jsx
import React, { useState } from 'react';
import { FileText, AlignLeft, HelpCircle, Layers } from 'lucide-react';

const LAYOUTS = [
    { 
        id: 'standard', 
        name: 'Standard Cover Letter', 
        icon: AlignLeft, 
        desc: 'Professional flow: Why Me → Why You → Fit.' 
    },
    { 
        id: 'mission', 
        name: 'Mission-Driven Letter', 
        icon: Layers, 
        desc: 'Value alignment: Why You → Why Me → Fit.' 
    },
    { 
        id: 'specialist', 
        name: 'Specialist / Technical', 
        icon: AlignLeft, 
        desc: 'Focus on Hard Skills & Competency.' 
    },
    { 
        id: 'qa', 
        name: 'Q&A / Selection Criteria', 
        icon: HelpCircle, 
        desc: 'Generate sections based on specific questions.' 
    },
    { 
        id: 'blank', 
        name: 'Blank Document', 
        icon: FileText, 
        desc: 'Start from scratch (Not recommended).' 
    },
];

const NewDocumentModal = ({ isOpen, onClose, onCreate }) => {
    const [name, setName] = useState('Supporting Document');
    const [selectedLayout, setSelectedLayout] = useState('standard');
    const [questions, setQuestions] = useState('');
    const [isSubmitting, setIsSubmitting] = useState(false);

    if (!isOpen) return null;

    const handleSubmit = async () => {
        if (!name.trim()) return alert("Please enter a document name");
        
        setIsSubmitting(true);
        let qList = [];
        
        if (selectedLayout === 'qa') {
            qList = questions.split('\n').map(q => q.trim()).filter(q => q.length > 0);
            if (qList.length === 0) {
                setIsSubmitting(false);
                return alert("Please enter at least one question.");
            }
        }
        
        await onCreate(name, selectedLayout, qList);
        setIsSubmitting(false);
        // onClose is handled by parent after success
    };

    return (
        <div className="modal d-block" style={{ backgroundColor: 'rgba(0,0,0,0.5)', zIndex: 1050 }}>
            <div className="modal-dialog modal-lg modal-dialog-centered animate-fade-in-up">
                <div className="modal-content shadow-lg border-0 overflow-hidden">
                    
                    {/* Header */}
                    <div className="modal-header bg-light border-bottom-0 pb-0">
                        <div>
                            <h5 className="modal-title fw-bold text-dark">Create New Document</h5>
                            <p className="text-muted small mb-2">Choose a layout to jump-start your writing.</p>
                        </div>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>

                    <div className="modal-body p-4">
                        
                        {/* Name Input */}
                        <div className="mb-4">
                            <label className="form-label fw-bold small text-uppercase text-muted">Document Name</label>
                            <input 
                                type="text" 
                                className="form-control form-control-lg" 
                                placeholder="e.g. Selection Criteria, Personal Statement..."
                                value={name} 
                                onChange={(e) => setName(e.target.value)}
                                autoFocus
                            />
                        </div>

                        {/* Layout Picker */}
                        <label className="form-label fw-bold small text-uppercase text-muted mb-3">Select Layout Strategy</label>
                        <div className="row g-3 mb-4">
                            {LAYOUTS.map(layout => (
                                <div key={layout.id} className="col-md-6">
                                    <div 
                                        className={`card h-100 cursor-pointer transition-all border-0 shadow-sm ${selectedLayout === layout.id ? 'bg-primary-subtle ring-2 ring-primary' : 'bg-light hover-shadow'}`}
                                        onClick={() => setSelectedLayout(layout.id)}
                                        style={{ transition: '0.2s' }}
                                    >
                                        <div className="card-body d-flex align-items-start gap-3 p-3">
                                            <div className={`p-2 rounded-3 mt-1 ${selectedLayout === layout.id ? 'bg-primary text-white' : 'bg-white text-muted shadow-sm'}`}>
                                                <layout.icon size={20} />
                                            </div>
                                            <div>
                                                <h6 className={`fw-bold mb-1 ${selectedLayout === layout.id ? 'text-primary' : 'text-dark'}`}>{layout.name}</h6>
                                                <p className="text-muted small mb-0 lh-sm">{layout.desc}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>

                        {/* Q&A Input Section */}
                        {selectedLayout === 'qa' && (
                            <div className="animate-fade-in bg-light p-3 rounded-3 border">
                                <div className="d-flex justify-content-between align-items-center mb-2">
                                    <label className="form-label fw-bold small text-uppercase text-muted mb-0">Questions to Answer</label>
                                    <span className="badge bg-primary-subtle text-primary border border-primary-subtle">One per line</span>
                                </div>
                                <textarea 
                                    className="form-control border-0 shadow-sm" 
                                    rows={5}
                                    placeholder="e.g.&#10;1. Describe a time you demonstrated leadership.&#10;2. How do you handle tight deadlines?&#10;3. What is your experience with React?"
                                    value={questions}
                                    onChange={(e) => setQuestions(e.target.value)}
                                    style={{ resize: 'none' }}
                                ></textarea>
                                <div className="form-text small mt-2">
                                    <i className="bi bi-magic"></i> We will analyze your mapped evidence and automatically attach relevant items to each question.
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Footer */}
                    <div className="modal-footer border-top-0 pt-0 pb-4 px-4">
                        <button className="btn btn-light rounded-pill px-4" onClick={onClose} disabled={isSubmitting}>Cancel</button>
                        <button 
                            className="btn btn-primary rounded-pill px-4 fw-bold shadow-sm d-flex align-items-center gap-2" 
                            onClick={handleSubmit}
                            disabled={isSubmitting}
                        >
                            {isSubmitting ? 'Creating...' : 'Create Document'}
                        </button>
                    </div>
                </div>
            </div>
            <style>{`
                .hover-shadow:hover { transform: translateY(-2px); box-shadow: 0 .5rem 1rem rgba(0,0,0,.15)!important; }
                .ring-2 { box-shadow: 0 0 0 2px var(--bs-primary); }
                .animate-fade-in-up { animation: fadeInUp 0.3s ease-out; }
                @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
            `}</style>
        </div>
    );
};

export default NewDocumentModal;