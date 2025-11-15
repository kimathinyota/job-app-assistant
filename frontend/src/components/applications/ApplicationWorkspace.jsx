// frontend/src/components/applications/ApplicationWorkspace.jsx
import React, { useState, useEffect } from 'react';
// --- (NEW) ---
import { useParams, useNavigate } from 'react-router-dom';
import { CheckCircle, Info } from 'lucide-react'; 

// API Imports
import { 
    fetchApplicationDetails, 
    fetchJobDetails, 
    fetchMappingDetails,
    updateApplication 
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient';

// Component Imports
import Step1_TriageView from './Step1_TriageView';
import Step2_GenerateCV from './Step2_GenerateCV';
import Step3_BuildCoverLetter from './Step3_BuildCoverLetter';
import Step4_Submit from './Step4_Submit';
import JobPreviewModal from './JobPreviewModal';

import './ApplicationWorkspace.css';

// --- (MODIFIED) ---
// Props are removed
const ApplicationWorkspace = () => {
    // --- (NEW) ---
    const { applicationId } = useParams(); // Get ID from URL
    const navigate = useNavigate(); // Get navigation function

    // UI State
    const [currentStep, setCurrentStep] = useState(1);
    const [isJobModalOpen, setIsJobModalOpen] = useState(false);

    // Data State
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [data, setData] = useState({
        application: null,
        job: null,
        cv: null,
        mapping: null
    });

    // --- (MODIFIED) ---
    // Data fetching now uses 'applicationId' from useParams
    useEffect(() => {
        const loadAllData = async () => {
            try {
                setLoading(true);
                const appRes = await fetchApplicationDetails(applicationId);
                const app = appRes.data;

                const [jobRes, cvRes, mappingRes] = await Promise.all([
                    fetchJobDetails(app.job_id),
                    fetchCVDetails(app.base_cv_id),
                    fetchMappingDetails(app.mapping_id)
                ]);

                setData({
                    application: app,
                    job: jobRes.data,
                    cv: cvRes,
                    mapping: mappingRes.data
                });

            } catch (err) {
                setError("Failed to load application data.");
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        loadAllData();
    }, [applicationId]);

    // --- HANDLERS ---
    const handleNextStep = () => setCurrentStep(prev => prev + 1);
    const handlePrevStep = () => setCurrentStep(prev => prev - 1);

    const handleStepClick = (stepId) => {
        if (stepId < currentStep) {
            setCurrentStep(stepId);
        }
    };

    const reloadMapping = async () => {
        const mappingRes = await fetchMappingDetails(data.application.mapping_id);
        setData(prev => ({ ...prev, mapping: mappingRes.data }));
    };

    // --- (NEW) ---
    // This is called on the final step to return to the app list
    const handleOnComplete = () => {
        navigate('/applications');
    };

    const handleCoverLetterCreated = async (newCoverLetterId) => {
        if (!data.application || data.application.cover_letter_id) return; 
        try {
            await updateApplication(applicationId, { cover_letter_id: newCoverLetterId });
            setData(prev => ({
                ...prev,
                application: { ...prev.application, cover_letter_id: newCoverLetterId }
            }));
        } catch (err) {
            console.error("Failed to link cover letter:", err);
        }
    };

    // --- RENDER LOADING STATES ---
    if (loading) return (
        <div className="d-flex justify-content-center align-items-center py-5">
            <div className="spinner-border text-primary" role="status">
                <span className="visually-hidden">Loading...</span>
            </div>
        </div>
    );
    
    if (error) return <div className="alert alert-danger m-4">{error}</div>;
    
    const { application, job, cv, mapping } = data;
    if (!application || !job || !cv || !mapping) return <p className="p-4">Initializing Workspace...</p>;

    const steps = [
        { id: 1, label: 'Job Analysis' },
        { id: 2, label: 'Tailor CV' },
        { id: 3, label: 'Cover Letter' },
        { id: 4, label: 'Submit' }
    ];

    return (
        // This component renders full-width, so we add our own padding
        <div className="application-workspace animate-fade-in p-4">
            
            <style>{`
                .stepper-clickable:hover {
                    opacity: 0.8;
                    transform: scale(1.1);
                }
                .cursor-pointer {
                    cursor: pointer;
                }
            `}</style>
            
            {/* --- (MODIFIED) HEADER --- */}
            <div className="d-flex align-items-center justify-content-between gap-2 mb-4">
                {/* Left Group: Title (Back button is GONE) */}
                <div style={{ minWidth: 0 }}>
                    <h4 className="mb-0 fw-bold text-truncate">{job.title}</h4>
                    <small className="text-muted text-truncate d-block">{job.company}</small>
                </div>
                
                {/* Right Group: View Job Button (Icon only) */}
                <button 
                    className="btn btn-outline-info btn-sm d-flex align-items-center p-2" 
                    onClick={() => setIsJobModalOpen(true)}
                    title="View Job Description"
                    style={{ flexShrink: 0 }}
                >
                    <Info size={16} />
                </button>
            </div>
            {/* --- END OF HEADER --- */}

            {/* Progress Stepper (with clickable logic) */}
            <div className="card border-0 shadow-sm mb-4">
                <div className="card-body py-3">
                    <div className="d-flex justify-content-between align-items-center position-relative">
                        <div className="position-absolute top-50 start-0 w-100 bg-light" style={{ height: '2px', zIndex: 0 }}></div>
                        
                        {steps.map((step) => {
                            const isCompleted = step.id < currentStep;
                            const isActive = step.id === currentStep;
                            return (
                                <div 
                                    key={step.id} 
                                    className={`d-flex flex-column align-items-center position-relative ${isCompleted ? 'cursor-pointer' : ''}`} 
                                    style={{ zIndex: 1 }}
                                    onClick={() => handleStepClick(step.id)}
                                >
                                    <div 
                                        className={`rounded-circle d-flex align-items-center justify-content-center border border-2 ${
                                            isCompleted || isActive ? 'bg-primary border-primary text-white' : 'bg-white border-secondary-subtle text-muted'
                                        } ${isCompleted ? 'stepper-clickable' : ''}`}
                                        style={{ width: '32px', height: '32px', transition: 'all 0.3s' }}
                                    >
                                        {isCompleted ? <CheckCircle size={16} /> : <span style={{ fontSize: '12px', fontWeight: 'bold' }}>{step.id}</span>}
                                    </div>
                                    <span className={`mt-2 small fw-bold ${isActive || isCompleted ? 'text-primary' : 'text-muted'}`}>
                                        {step.label}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Step Content Area */}
            <div className="card shadow-sm border-light">
                <div className="card-body p-4">
                    {currentStep === 1 && (
                        <Step1_TriageView
                            job={job}
                            cv={cv}
                            mapping={mapping}
                            onMappingChanged={reloadMapping}
                            onNext={handleNextStep}
                        />
                    )}
                    {currentStep === 2 && (
                        <Step2_GenerateCV
                            job={job}
                            cv={cv}
                            mapping={mapping}
                            onPrev={handlePrevStep}
                            onNext={handleNextStep}
                            onMappingChanged={reloadMapping}
                        />
                    )}
                    {currentStep === 3 && (
                        <Step3_BuildCoverLetter
                            application={application}
                            job={job}
                            mapping={mapping}
                            fullCV={cv}
                            onPrev={handlePrevStep}
                            onNext={handleNextStep}
                            onCoverLetterCreated={handleCoverLetterCreated}
                        />
                    )}
                    {currentStep === 4 && (
                        <Step4_Submit
                            applicationId={application.id}
                            onPrev={handlePrevStep}
                            onComplete={handleOnComplete} // <-- Use new handler
                        />
                    )}
                </div>
            </div>

            <JobPreviewModal
                isOpen={isJobModalOpen}
                onClose={() => setIsJobModalOpen(false)}
                job={job}
            />
        </div>
    );
};

export default ApplicationWorkspace;