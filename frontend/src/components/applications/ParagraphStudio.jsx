// frontend/src/components/applications/ParagraphStudio.jsx
import React, { useState, useMemo } from 'react';
import { Sparkles, Pencil, Plus, BookUser } from 'lucide-react';
import ArgumentCard from './ArgumentCard.jsx';
import ProseEditor from './ProseEditor.jsx'; // The new "heavy" editor

const ParagraphStudio = ({
    paragraph,
    jobFeatures,
    fullCV,
    ideaMap,
    pairMap,
    isSubmitting,
    onAddArgument,
    onDeleteIdea,
    onUpdateIdea,
    onUpdateParagraph,
    onShowPreview
}) => {
    const [activeView, setActiveView] = useState('strategy'); // 'strategy' or 'write'
    
    // Get the full Idea objects for this paragraph
    const ideas = useMemo(() => {
        return paragraph.idea_ids.map(id => ideaMap.get(id)).filter(Boolean);
    }, [paragraph.idea_ids, ideaMap]);

    // Determine classification for new ideas
    const classification = (paragraph.purpose || "").includes("Professional") ? "professional"
                         : (paragraph.purpose || "").includes("Personal") ? "personal"
                         : (paragraph.purpose || "").includes("Company") ? "company"
                         : "unclassified";
                         
    return (
        <div className="bg-white border rounded-3 shadow-sm">
            {/* --- 1. TABS (The "Flip" Button) --- */}
            <ul className="nav nav-tabs nav-fill" style={{ padding: '0 0.5rem' }}>
                <li className="nav-item">
                    <button 
                        className={`nav-link ${activeView === 'strategy' ? 'active' : ''}`}
                        onClick={() => setActiveView('strategy')}
                        style={{ fontWeight: activeView === 'strategy' ? '600' : '400' }}
                    >
                        <Sparkles size={16} className="me-2" /> Strategy
                    </button>
                </li>
                <li className="nav-item">
                    <button 
                        className={`nav-link ${activeView === 'write' ? 'active' : ''}`}
                        onClick={() => setActiveView('write')}
                        style={{ fontWeight: activeView === 'write' ? '600' : '400' }}
                    >
                        <Pencil size={16} className="me-2" /> Write
                    </button>
                </li>
            </ul>

            {/* --- 2. TAB CONTENT --- */}
            <div className="p-3">
                {activeView === 'strategy' && (
                    <div>
                        <div className="d-flex justify-content-between align-items-center mb-3">
                            <div>
                                <h5 className="fw-bold text-dark mb-0">{paragraph.purpose}</h5>
                                <p className="mb-0 text-muted small">
                                    Build your arguments. Add evidence and notes for the AI.
                                </p>
                            </div>
                            {paragraph.owner === 'user' && (
                                <span className="badge bg-success-subtle text-success-emphasis rounded-pill d-flex align-items-center gap-1">
                                    <BookUser size={12} /> Custom Section
                                </span>
                            )}
                        </div>
                        
                        <div className="d-flex flex-column gap-3">
                            {ideas.map(idea => (
                                <ArgumentCard
                                    key={idea.id}
                                    idea={idea}
                                    paragraph={paragraph}
                                    pairMap={pairMap}
                                    fullCV={fullCV}
                                    jobFeatures={jobFeatures}
                                    onUpdateIdea={onUpdateIdea}
                                    onDeleteIdea={onDeleteIdea}
                                    onShowPreview={onShowPreview}
                                />
                            ))}
                        </div>
                        
                        <button
                            className="btn btn-outline-primary d-flex align-items-center gap-2 mt-3"
                            onClick={() => onAddArgument(paragraph.id, classification)}
                            disabled={isSubmitting}
                        >
                            <Plus size={16} /> Add Custom Argument
                        </button>
                    </div>
                )}

                {activeView === 'write' && (
                    <div>
                        <div className="d-flex justify-content-between align-items-center mb-2">
                             <div>
                                <h5 className="fw-bold text-dark mb-0">{paragraph.purpose}</h5>
                                <p className="mb-0 text-muted small">
                                    Write your prose. Use the Strategy Rail for guidance.
                                </p>
                            </div>
                        </div>
                        <ProseEditor
                            paragraph={paragraph}
                            ideas={ideas}
                            fullCV={fullCV}
                            jobFeatures={jobFeatures}
                            onSave={onUpdateParagraph}
                            onShowPreview={onShowPreview}
                        />
                    </div>
                )}
            </div>
        </div>
    );
};

export default ParagraphStudio;