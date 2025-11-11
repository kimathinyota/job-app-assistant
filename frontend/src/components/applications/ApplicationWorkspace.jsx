// frontend/src/components/applications/ApplicationWorkspace.jsx
import React, { useState, useEffect } from 'react';
import { 
    fetchApplicationDetails, 
    fetchJobDetails, 
    fetchMappingDetails,
    updateApplication // <-- 1. IMPORT THE NEW API FUNCTION
} from '../../api/applicationClient.js';
import { fetchCVDetails } from '../../api/cvClient.js';

import Step1_TriageView from './Step1_TriageView.jsx';
import Step2_GenerateCV from './Step2_GenerateCV.jsx';
import Step3_BuildCoverLetter from './Step3_BuildCoverLetter.jsx';
import Step4_Submit from './Step4_Submit.jsx';
import JobPreviewModal from './JobPreviewModal.jsx';

import './ApplicationWorkspace.css';

const ApplicationWorkspace = ({ applicationId, onExitWorkspace }) => {
    const [currentStep, setCurrentStep] = useState(1);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isJobModalOpen, setIsJobModalOpen] = useState(false);
    
    const [data, setData] = useState({
        application: null,
        job: null,
        cv: null,
        mapping: null
    });

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

    const handleNextStep = () => setCurrentStep(prev => prev + 1);
    const handlePrevStep = () => setCurrentStep(prev => prev - 1);

    const reloadMapping = async () => {
        const mappingRes = await fetchMappingDetails(data.application.mapping_id);
        setData(prev => ({ ...prev, mapping: mappingRes.data }));
    };

    // --- 2. ADD THIS CALLBACK HANDLER ---
    /**
     * Called by Step3 when a cover letter is first created.
     * This links the new CL to the application in the DB and updates local state.
     */
    const handleCoverLetterCreated = async (newCoverLetterId) => {
        // Only run if the application doesn't already have a linked CL
        if (!data.application || data.application.cover_letter_id) return; 
        
        try {
            // Update the backend application object
            await updateApplication(applicationId, { cover_letter_id: newCoverLetterId });
            
            // Update local state so the `application` prop is correct
            setData(prev => ({
                ...prev,
                application: { ...prev.application, cover_letter_id: newCoverLetterId }
            }));
        } catch (err) {
            console.error("Failed to link cover letter to application:", err);
            // Don't block the UI, but log the error
        }
    };


    if (loading) return <p>Loading Workspace...</p>;
    if (error) return <p className="text-danger">{error}</p>;
    
    const { application, job, cv, mapping } = data;

    if (!application || !job || !cv || !mapping) {
        console.warn("Waiting for all data to be populated:", { application, job, cv, mapping });
        return <p>Loading Workspace data...</p>;
    }

    return (
        <div className="container-fluid text-start">
            <button onClick={onExitWorkspace} className="btn btn-outline-secondary btn-sm mb-3">
                &larr; Back to All Applications
            </button>
            
            <div className="d-flex justify-content-between align-items-center mb-3">
                <h2 className="h4 mb-0">{job.title} <span className="text-muted fw-normal">@ {job.company}</span></h2>
                <button 
                    className="btn btn-outline-info" 
                    onClick={() => setIsJobModalOpen(true)}
                >
                    <i className="bi bi-eye-fill me-2"></i>
                    View Job Details
                </button>
            </div>


            {/* The Stepper UI */}
            <div className="stepper-ui mb-4">
                <div className={`step ${currentStep === 1 ? 'active' : (currentStep > 1 ? 'complete' : '')}`}>
                    <div className="step-circle">1</div>
                    <div>Map CV</div>
                </div>
                <div className={`step ${currentStep === 2 ? 'active' : (currentStep > 2 ? 'complete' : '')}`}>
                    <div className="step-circle">2</div>
                    <div>Generate CV</div>
                </div>
                <div className={`step ${currentStep === 3 ? 'active' : (currentStep > 3 ? 'complete' : '')}`}>
                    <div className="step-circle">3</div>
                    <div>Build Cover Letter</div>
                </div>
                <div className={`step ${currentStep === 4 ? 'active' : ''}`}>
                    <div className="step-circle">4</div>
                    <div>Submit & Track</div>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="wizard-content card shadow-sm">
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
                            fullCV={cv} // <-- ADD THIS LINE
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