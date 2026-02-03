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
      // Error handling logic
    }
  };

  const handleManualSubmit = async (payload) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await createManualMatch(appId, activeManualFeature, payload);
      setAnalysis(response.data);
      setActiveManualFeature(null);
    } catch (err) {
      // Error handling logic
    }
  };

  if (loading) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center bg-slate-50 space-y-4 min-h-[400px]">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
        <div className="text-slate-500 font-medium text-sm animate-pulse">{loadingStage}</div>
      </div>
    );
  }

  if (!analysis) return <div className="p-10 text-center text-slate-500">Could not load analysis data.</div>;

  return (
    <div className="flex flex-col h-full bg-slate-50/50">
      {/* 1. Header */}
      <ForensicHeader stats={analysis.stats} />

      {/* 2. Board Area */}
      <div className="flex-1 overflow-x-auto p-6">
        <EvidenceBoard 
          groups={analysis.groups} 
          onReject={handleReject} 
          onOpenManual={(featureId) => setActiveManualFeature(featureId)} 
        />
      </div>

      {/* 3. Floating Action Bar (Only for Draft Mode) */}
      {!applicationId && onSaveDraft && (
        <div className="sticky bottom-0 bg-white border-t border-slate-200 p-4 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.05)] z-30">
          <div className="max-w-7xl mx-auto flex justify-between items-center">
            <div className="text-sm text-slate-600">
              <span className="font-semibold text-slate-800">Draft Mode:</span> Changes here will be saved when you create the application.
            </div>
            <button 
              onClick={onSaveDraft}
              className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-lg font-semibold shadow-sm shadow-indigo-200 transition-all flex items-center space-x-2"
            >
              <span>Create Application</span>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3"></path></svg>
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