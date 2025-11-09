// frontend/src/components/applications/ApplicationWorkspace.jsx
import React, { useState, useEffect } from 'react';
import { 
    fetchApplicationDetails, 
    fetchJobDetails, 
    fetchMappingDetails
} from '../../api/applicationClient';
import { fetchCVDetails } from '../../api/cvClient'; // From your existing client

// --- THIS IS THE CHANGE ---
import Step1_TriageView from './Step1_TriageView'; // <-- Use the new Triage View
// import Step1_Mapping from './Step1_mapping'; // <-- No longer needed
// --- END OF CHANGE ---
import Step2_GenerateCV from './Step2_GenerateCV';
import Step3_BuildCoverLetter from './Step3_BuildCoverLetter';
import Step4_Submit from './Step4_Submit';

import './ApplicationWorkspace.css'; // We'll create this CSS file

const ApplicationWorkspace = ({ applicationId, onExitWorkspace }) => {
    const [currentStep, setCurrentStep] = useState(1);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    // This state will hold all the data for the wizard
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
                // 1. Fetch the application
                const appRes = await fetchApplicationDetails(applicationId);
                const app = appRes.data;

                // 2. Fetch all linked data in parallel
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

    // This forces the mapping data to be fresh in child components
    const reloadMapping = async () => {
        const mappingRes = await fetchMappingDetails(data.application.mapping_id);
        setData(prev => ({ ...prev, mapping: mappingRes.data }));
    };


    if (loading) return <p>Loading Workspace...</p>;
    if (error) return <p className="text-danger">{error}</p>;

    const { application, job, cv, mapping } = data;

    // --- ADD THIS BLOCK OF CODE ---
    // This ensures that even if loading is false,
    // we wait until all data objects are populated before rendering.
    if (!application || !job || !cv || !mapping) {
        console.warn("Waiting for all data to be populated:", { application, job, cv, mapping });
        return <p>Loading Workspace data...</p>;
    }
    // --- END OF FIX ---

    return (
        <div className="container-fluid text-start">
            <button onClick={onExitWorkspace} className="btn btn-outline-secondary btn-sm mb-3">
                &larr; Back to All Applications
            </button>
            
            <h2 className="h4">{job.title} <span className="text-muted fw-normal">@ {job.company}</span></h2>

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
                    {/* --- THIS IS THE CHANGE --- */}
                    {currentStep === 1 && (
                        <Step1_TriageView
                            job={job}
                            cv={cv}
                            mapping={mapping}
                            onMappingChanged={reloadMapping}
                            onNext={handleNextStep}
                        />
                    )}
                    {/* --- END OF CHANGE --- */}
                    {currentStep === 2 && (
                        <Step2_GenerateCV
                            job={job}
                            cv={cv}
                            mapping={mapping}
                            onPrev={handlePrevStep}
                            onNext={handleNextStep}
                            onMappingChanged={reloadMapping} // <-- ADD THIS PROP
                        />
                    )}
                    {currentStep === 3 && (
                        <Step3_BuildCoverLetter
                            application={application}
                            mapping={mapping}
                            onPrev={handlePrevStep}
                            onNext={handleNextStep}
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
        </div>
    );
};

export default ApplicationWorkspace;