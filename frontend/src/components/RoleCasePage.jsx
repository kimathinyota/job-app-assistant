import React from 'react';
import { useSearchParams, useParams, useNavigate } from 'react-router-dom';
import { RoleCaseView } from './rolecase/RoleCaseView'; // Ensure this path matches where you saved the main component
import { createApplication } from '../api/applicationClient';

export const RoleCasePage = () => {
  const { appId } = useParams(); 
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();

  // 1. Draft Mode (Query Params)
  const jobId = searchParams.get('jobId');
  const cvId = searchParams.get('cvId');

  // 2. Application Mode (URL Params)
  const applicationId = appId;

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
    <div className="h-screen w-full bg-white">
      <RoleCaseView 
        applicationId={applicationId} 
        jobId={jobId} 
        cvId={cvId}
        onSaveDraft={handleSaveDraft}
      />
    </div>
  );
};