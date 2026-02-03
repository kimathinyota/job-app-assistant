// frontend/src/components/rolecase/RoleCaseView.jsx
import React, { useState, useEffect } from 'react';
import { fetchForensicAnalysis, generateRoleCase, rejectMatch, createManualMatch } from '../../api/applicationClient';
import { FitScoreHeader } from './FitScoreHeader';
import { RequirementList } from './RequirementList';
import { EvidenceLinkerModal } from './EvidenceLinkerModal'; // New, smarter modal

export const RoleCaseView = ({ applicationId, jobId, cvId, onSaveDraft }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Checking your fit...");
  
  // Modal State
  const [activeFeature, setActiveFeature] = useState(null); 

  useEffect(() => {
    loadData();
  }, [applicationId, jobId, cvId]);

  const loadData = async () => {
    setLoading(true);
    try {
      if (applicationId) {
        setLoadingStage("Loading analysis...");
        const response = await fetchForensicAnalysis(applicationId);
        setAnalysis(response.data);
      } else if (jobId && cvId) {
        setLoadingStage("Reading Job Description...");
        // This takes a few seconds, so the message matters
        const response = await generateRoleCase(jobId, cvId);
        setAnalysis(response.data);
      }
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  // Actions
  const handleReject = async (featureId) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await rejectMatch(appId, featureId);
      setAnalysis(response.data.new_forensics); 
    } catch (err) {
      alert("Could not update match.");
    }
  };

  const handleManualLink = async (payload) => {
    // payload = { cv_item_id, cv_item_type, evidence_text }
    try {
      const appId = analysis.application_id || applicationId;
      const response = await createManualMatch(appId, activeFeature.requirement_id, payload);
      setAnalysis(response.data);
      setActiveFeature(null);
    } catch (err) {
      alert("Could not save match.");
    }
  };

  if (loading) {
    return (
      <div className="d-flex flex-column align-items-center justify-content-center vh-100 bg-light">
        <div className="spinner-border text-primary mb-3" role="status"></div>
        <div className="text-muted small fw-bold text-uppercase letter-spacing-1">{loadingStage}</div>
      </div>
    );
  }

  if (!analysis) return <div className="p-5 text-center text-muted">Analysis unavailable.</div>;

  return (
    <div className="d-flex flex-column h-100 bg-white">
      {/* 1. The "Hook" - Big Score Header */}
      <FitScoreHeader stats={analysis.stats} />

      {/* 2. The Checklist - Scrollable Area */}
      <div className="container-fluid flex-grow-1 overflow-auto bg-light py-4">
        <div className="row justify-content-center">
          <div className="col-12 col-lg-8 col-xl-7">
            <RequirementList 
              groups={analysis.groups} 
              onReject={handleReject} 
              onLinkEvidence={(feature) => setActiveFeature(feature)}
            />
          </div>
        </div>
      </div>

      {/* 3. The "Next Step" Footer (Draft Mode) */}
      {!applicationId && onSaveDraft && (
        <div className="bg-white border-top p-3 shadow-lg fixed-bottom">
          <div className="container d-flex justify-content-between align-items-center">
            <div>
              <div className="fw-bold text-dark">Ready to apply?</div>
              <div className="text-muted small">This analysis will be saved to your application.</div>
            </div>
            <button 
              onClick={onSaveDraft}
              className="btn btn-primary px-4 fw-bold rounded-pill"
            >
              Start Application
            </button>
          </div>
        </div>
      )}

      {/* 4. The "I Have This" Modal */}
      {activeFeature && (
        <EvidenceLinkerModal 
          isOpen={!!activeFeature}
          feature={activeFeature}
          onClose={() => setActiveFeature(null)}
          onSubmit={handleManualLink}
        />
      )}
    </div>
  );
};