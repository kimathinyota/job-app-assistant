import React from 'react';
import { useSearchParams, useParams, useNavigate } from 'react-router-dom';
import { createApplication } from '../api/applicationClient';

import { RoleCaseView } from './rolecase/RoleCaseView';
export const RoleCasePage = () => {
  // 1. Match the parameter name defined in App.jsx (path="application/:applicationId/mapping")
  const { applicationId } = useParams(); 
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  // 2. Draft Mode (Query Params)
  const jobId = searchParams.get('jobId');
  const cvId = searchParams.get('cvId');

  const handleSaveDraft = async () => {
    if (!jobId || !cvId) return;
    try {
      const response = await createApplication(jobId, cvId, null);
      const newAppId = response.data.id; 
      navigate(`/application/${newAppId}`);
    } catch (err) {
      alert("Failed to save application.");
      console.error(err);
    }
  };

  return (
    <div className="min-h-screen w-full bg-white">
      <RoleCaseView 
        applicationId={applicationId} 
        jobId={jobId} 
        cvId={cvId}
        onSaveDraft={handleSaveDraft}
      />
    </div>
  );
};

export default RoleCasePage;