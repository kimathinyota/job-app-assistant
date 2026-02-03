// frontend/src/components/rolecase/RoleCaseView.jsx
import React, { useState, useEffect } from 'react';
import { fetchForensicAnalysis, generateRoleCase, rejectMatch, createManualMatch } from '../../api/applicationClient';
import { ForensicHeader } from './ForensicHeader';
import { EvidenceBoard } from './EvidenceBoard';
import { ManualMatchModal } from './ManualMatchModal';

export const RoleCaseView = ({ applicationId, jobId, cvId, onSaveDraft }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Initializing...");
  const [activeManualFeature, setActiveManualFeature] = useState(null);

  useEffect(() => {
    loadData();
  }, [applicationId, jobId, cvId]);

  const loadData = async () => {
    setLoading(true);
    try {
      if (applicationId) {
        setLoadingStage("Loading Case File...");
        const response = await fetchForensicAnalysis(applicationId);
        setAnalysis(response.data);
      } else if (jobId && cvId) {
        setLoadingStage("Running AI Forensic Analysis...");
        const response = await generateRoleCase(jobId, cvId);
        setAnalysis(response.data);
      }
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async (featureId) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await rejectMatch(appId, featureId);
      setAnalysis(response.data.new_forensics); 
    } catch (err) {
      alert("Failed to reject match.");
    }
  };

  const handleManualSubmit = async (payload) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await createManualMatch(appId, activeManualFeature, payload);
      setAnalysis(response.data);
      setActiveManualFeature(null);
    } catch (err) {
      alert("Failed to save match.");
    }
  };

  if (loading) {
    return (
      <div className="d-flex flex-column align-items-center justify-content-center vh-100 bg-light">
        <div className="spinner-border text-primary mb-3" role="status"></div>
        <div className="text-muted fw-bold">{loadingStage}</div>
      </div>
    );
  }

  if (!analysis) return <div className="p-5 text-center text-muted">Could not load analysis data.</div>;

  return (
    <div className="d-flex flex-column h-100 bg-light">
      <style>
        {`
          .rolecase-container { height: calc(100vh - 60px); display: flex; flex-direction: column; }
          .board-scroll-area { overflow-x: auto; flex: 1; padding-bottom: 1rem; }
          .custom-scrollbar::-webkit-scrollbar { height: 8px; width: 8px; }
          .custom-scrollbar::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
          .glass-panel { background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(10px); }
        `}
      </style>

      {/* 1. Header */}
      <ForensicHeader stats={analysis.stats} />

      {/* 2. Board Area */}
      <div className="board-scroll-area p-4">
        <EvidenceBoard 
          groups={analysis.groups} 
          onReject={handleReject} 
          onOpenManual={(featureId) => setActiveManualFeature(featureId)} 
        />
      </div>

      {/* 3. Floating Action Bar (Draft Mode Only) */}
      {!applicationId && onSaveDraft && (
        <div className="sticky-bottom bg-white border-top p-3 shadow-lg">
          <div className="container-fluid d-flex justify-content-between align-items-center">
            <div className="text-muted small">
              <span className="fw-bold text-dark">Draft Mode:</span> Changes will be saved to your new application.
            </div>
            <button 
              onClick={onSaveDraft}
              className="btn btn-primary fw-bold shadow-sm d-flex align-items-center gap-2"
            >
              <span>Create Application</span>
              <i className="bi bi-arrow-right"></i>
            </button>
          </div>
        </div>
      )}

      {/* Modals */}
      <ManualMatchModal 
        isOpen={!!activeManualFeature} 
        onClose={() => setActiveManualFeature(null)}
        onSubmit={handleManualSubmit}
      />
    </div>
  );
};