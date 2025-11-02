// frontend/src/components/applications/Step4_Submit.jsx
import React, { useState } from 'react';
import { updateApplicationStatus } from '../../api/applicationClient';

const Step4_Submit = ({ applicationId, onPrev, onComplete }) => {
    const [isLoading, setIsLoading] = useState(false);

    const handleMarkAsApplied = async () => {
        setIsLoading(true);
        try {
            await updateApplicationStatus(applicationId, 'applied');
            alert("Application marked as 'Applied'!");
            onComplete(); // Exit the workspace
        } catch (err) {
            alert("Failed to update status.");
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div>
            <h4 className="h5">Step 4: Submit & Track</h4>
            <p className="text-muted">
                You have generated your prompts. Once you have submitted your application 
                (e.g., on the company's website), mark it as "Applied" here.
            </p>
            
            <div className="text-center p-4">
                <button 
                    className="btn btn-success btn-lg" 
                    onClick={handleMarkAsApplied}
                    disabled={isLoading}
                >
                    {isLoading ? "Saving..." : "Mark as Applied"}
                </button>
            </div>
            
            <div className="d-flex justify-content-between mt-4">
                <button className="btn btn-secondary" onClick={onPrev}>
                    &lt; Back: Build Cover Letter
                </button>
            </div>
        </div>
    );
};

export default Step4_Submit;