import React, { useState, useEffect } from 'react';
import { ArrowLeft, CheckCircle } from 'lucide-react';

// --- 1. RESTORED API IMPORTS ---
import { 
    fetchApplicationDetails, 
    fetchJobDetails, 
    fetchMappingDetails,
    updateApplication 
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient';

// --- 2. RESTORED COMPONENT IMPORTS ---
import Step1_TriageView from './Step1_TriageView'; // Was Step1_Mapping in your snippet, assuming TriageView is the correct UI component
import Step2_GenerateCV from './Step2_GenerateCV';
import Step3_BuildCoverLetter from './Step3_BuildCoverLetter';
import Step4_Submit from './Step4_Submit';
import JobPreviewModal from './JobPreviewModal';

import './ApplicationWorkspace.css';

const ApplicationWorkspace = ({ applicationId, onExitWorkspace }) => {
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

    // --- 3. RESTORED DATA FETCHING LOGIC ---
    useEffect(() => {
        const loadAllData = async () => {
            try {
                setLoading(true);
                // 1. Fetch Application
                const appRes = await fetchApplicationDetails(applicationId);
                const app = appRes.data;

                // 2. Fetch dependencies in parallel
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

    // --- 4. RESTORED HANDLERS ---
    const handleNextStep = () => setCurrentStep(prev => prev + 1);
    const handlePrevStep = () => setCurrentStep(prev => prev - 1);

    const reloadMapping = async () => {
        const mappingRes = await fetchMappingDetails(data.application.mapping_id);
        setData(prev => ({ ...prev, mapping: mappingRes.data }));
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

    // --- 5. RENDER LOADING STATES ---
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

    // --- 6. MAIN RENDER (THE NEW UI SHELL) ---
    const steps = [
        { id: 1, label: 'Job Analysis' },
        { id: 2, label: 'Tailor CV' },
        { id: 3, label: 'Cover Letter' },
        { id: 4, label: 'Submit' }
    ];

    return (
        <div className="application-workspace animate-fade-in">
            
            {/* Header */}
            <div className="d-flex align-items-center justify-content-between mb-4">
                <div className="d-flex align-items-center gap-3">
                    <button 
                        onClick={onExitWorkspace} 
                        className="btn btn-outline-secondary d-flex align-items-center gap-2"
                    >
                        <ArrowLeft size={16} />
                        Back
                    </button>
                    <div>
                        <h4 className="mb-0 fw-bold">{job.title}</h4>
                        <small className="text-muted">@ {job.company}</small>
                    </div>
                </div>
                <button 
                    className="btn btn-outline-info btn-sm" 
                    onClick={() => setIsJobModalOpen(true)}
                >
                    View Job Description
                </button>
            </div>

            {/* Progress Stepper */}
            <div className="card border-0 shadow-sm mb-4">
                <div className="card-body py-3">
                    <div className="d-flex justify-content-between align-items-center position-relative">
                        <div className="position-absolute top-50 start-0 w-100 bg-light" style={{ height: '2px', zIndex: 0 }}></div>
                        
                        {steps.map((step) => {
                            const isCompleted = step.id < currentStep;
                            const isActive = step.id === currentStep;
                            return (
                                <div key={step.id} className="d-flex flex-column align-items-center position-relative" style={{ zIndex: 1 }}>
                                    <div 
                                        className={`rounded-circle d-flex align-items-center justify-content-center border border-2 ${
                                            isCompleted || isActive ? 'bg-primary border-primary text-white' : 'bg-white border-secondary-subtle text-muted'
                                        }`}
                                        style={{ width: '32px', height: '32px', transition: 'all 0.3s' }}
                                    >
                                        {isCompleted ? <CheckCircle size={16} /> : <span style={{ fontSize: '12px', fontWeight: 'bold' }}>{step.id}</span>}
                                    </div>
                                    <span className={`mt-2 small fw-bold ${isActive ? 'text-primary' : 'text-muted'}`}>
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
                            onComplete={onExitWorkspace}
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