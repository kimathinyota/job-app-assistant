import React, { useState } from 'react';
import { ArrowUp, ArrowDown, Download, FileText, FileCode, FileType, CheckSquare, Square } from 'lucide-react';
import { exportCV } from '../../api/cvClient';

const ExportCVModal = ({ cvId, onClose }) => {
    const [loading, setLoading] = useState(false);
    const [format, setFormat] = useState('pdf');
    
    // ADDED 'summary' here so it can be reordered
    const [sections, setSections] = useState([
        { id: 'summary', defaultTitle: 'Professional Summary', included: true },
        { id: 'experience', defaultTitle: 'Experience', included: true },
        { id: 'education', defaultTitle: 'Education', included: true },
        { id: 'projects', defaultTitle: 'Academic & Research Projects', included: true },
        { id: 'skills', defaultTitle: 'Technical Skills', included: true },
        { id: 'hobbies', defaultTitle: 'Interests & Hobbies', included: true }
    ]);

    const [titles, setTitles] = useState({
        summary: 'Professional Summary',
        experience: 'Experience',
        education: 'Education',
        projects: 'Academic & Research Projects',
        skills: 'Technical Skills',
        hobbies: 'Interests & Hobbies'
    });

    const handleMove = (index, direction) => {
        const newSections = [...sections];
        if (direction === 'up' && index > 0) {
            [newSections[index], newSections[index - 1]] = [newSections[index - 1], newSections[index]];
        } else if (direction === 'down' && index < newSections.length - 1) {
            [newSections[index], newSections[index + 1]] = [newSections[index + 1], newSections[index]];
        }
        setSections(newSections);
    };

    const handleTitleChange = (id, newVal) => {
        setTitles(prev => ({ ...prev, [id]: newVal }));
    };

    const toggleSection = (index) => {
        const newSections = [...sections];
        newSections[index].included = !newSections[index].included;
        setSections(newSections);
    };

    const handleExport = async () => {
        setLoading(true);
        try {
            const activeOrder = sections
                .filter(s => s.included)
                .map(s => s.id);

            const payload = {
                section_order: activeOrder,
                section_titles: titles,
                file_format: format
            };

            const response = await exportCV(cvId, payload);
            
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            
            const contentDisposition = response.headers['content-disposition'];
            let filename = `cv_export.${format === 'tex' ? 'tex' : format === 'zip' ? 'zip' : format}`;
            if (contentDisposition) {
                const filenameMatch = contentDisposition.match(/filename="?([^"]+)"?/);
                if (filenameMatch && filenameMatch.length === 2) filename = filenameMatch[1];
            }
            
            link.setAttribute('download', filename);
            document.body.appendChild(link);
            link.click();
            link.remove();
            window.URL.revokeObjectURL(url);
            
            onClose();
        } catch (error) {
            console.error("Export failed:", error);
            alert("Failed to export CV. Please check the console.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            <div className="modal-backdrop fade show"></div>
            <div className="modal fade show d-block" tabIndex="-1">
                <div className="modal-dialog modal-dialog-centered">
                    <div className="modal-content shadow-lg border-0">
                        <div className="modal-header border-bottom-0 pb-0">
                            <h5 className="modal-title fw-bold">Export CV</h5>
                            <button type="button" className="btn-close" onClick={onClose}></button>
                        </div>
                        <div className="modal-body">
                            
                            <div className="mb-4">
                                <label className="form-label fw-bold small text-muted text-uppercase">File Format</label>
                                <div className="d-flex gap-2">
                                    {['pdf', 'docx', 'tex', 'zip'].map(fmt => (
                                        <button 
                                            key={fmt}
                                            type="button" 
                                            onClick={() => setFormat(fmt)}
                                            className={`btn btn-sm flex-fill d-flex align-items-center justify-content-center gap-2 ${format === fmt ? 'btn-primary' : 'btn-outline-secondary'}`}
                                        >
                                            {fmt === 'pdf' && <FileText size={14} />}
                                            {fmt === 'docx' && <FileType size={14} />}
                                            {fmt === 'tex' && <FileCode size={14} />}
                                            {fmt.toUpperCase()}
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <div className="mb-3">
                                <label className="form-label fw-bold small text-muted text-uppercase mb-2">Section Customization</label>
                                <div className="d-flex flex-column gap-2">
                                    {sections.map((section, index) => (
                                        <div key={section.id} className={`d-flex align-items-center gap-2 p-2 border rounded ${section.included ? 'bg-light' : 'bg-secondary bg-opacity-10'}`}>
                                            
                                            <button 
                                                onClick={() => toggleSection(index)}
                                                className={`btn btn-link p-0 ${section.included ? 'text-primary' : 'text-muted'}`}
                                            >
                                                {section.included ? <CheckSquare size={18} /> : <Square size={18} />}
                                            </button>

                                            <div className="d-flex flex-column gap-1">
                                                <button 
                                                    type="button"
                                                    onClick={() => handleMove(index, 'up')}
                                                    disabled={index === 0}
                                                    className="btn btn-xs btn-link p-0 text-muted"
                                                >
                                                    <ArrowUp size={14} />
                                                </button>
                                                <button 
                                                    type="button"
                                                    onClick={() => handleMove(index, 'down')}
                                                    disabled={index === sections.length - 1}
                                                    className="btn btn-xs btn-link p-0 text-muted"
                                                >
                                                    <ArrowDown size={14} />
                                                </button>
                                            </div>

                                            <div className="flex-grow-1">
                                                <input 
                                                    type="text" 
                                                    className={`form-control form-control-sm border-0 bg-transparent fw-medium ${!section.included && 'text-decoration-line-through text-muted'}`}
                                                    value={titles[section.id]}
                                                    onChange={(e) => handleTitleChange(section.id, e.target.value)}
                                                    placeholder={section.defaultTitle}
                                                    disabled={!section.included}
                                                />
                                            </div>
                                            
                                            <span className="badge bg-secondary opacity-25 text-uppercase" style={{fontSize: '0.65rem'}}>
                                                {section.id}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                        </div>
                        <div className="modal-footer border-top-0 pt-0">
                            <button type="button" className="btn btn-light" onClick={onClose}>Cancel</button>
                            <button 
                                type="button" 
                                className="btn btn-success d-flex align-items-center gap-2"
                                onClick={handleExport}
                                disabled={loading}
                            >
                                {loading ? <span className="spinner-border spinner-border-sm"></span> : <Download size={16} />}
                                Download {format.toUpperCase()}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default ExportCVModal;