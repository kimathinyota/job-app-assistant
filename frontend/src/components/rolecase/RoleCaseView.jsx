import React, { useState, useEffect } from 'react';
import { fetchForensicAnalysis, generateRoleCase, rejectMatch, createManualMatch } from '../../api/applicationClient';
import { ForensicHeader } from './ForensicHeader';
import { EvidenceBoard } from './EvidenceBoard';
import { ManualMatchModal } from './ManualMatchModal';

export const RoleCaseView = ({ applicationId, jobId, cvId, onSaveDraft }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Initializing...");
  
  // Modal State
  const [activeManualFeature, setActiveManualFeature] = useState(null); // ID of feature being edited

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
      console.error("RoleCase Load Error:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async (featureId) => {
    // Optimistic UI or wait for response? We wait for response to get accurate stats.
    try {
      const appId = analysis.application_id || applicationId;
      const response = await rejectMatch(appId, featureId);
      // The backend returns the full new forensics object in .new_forensics
      setAnalysis(response.data.new_forensics); 
    } catch (err) {
      alert("Failed to reject match.");
    }
  };

  const handleManualSubmit = async (payload) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await createManualMatch(appId, activeManualFeature, payload);
      setAnalysis(response.data); // Update with new stats
      setActiveManualFeature(null);
    } catch (err) {
      alert("Failed to save manual match.");
    }
  };

  if (loading) {
    return (
      <div className="h-full flex flex-col items-center justify-center space-y-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        <p className="text-gray-500 font-medium">{loadingStage}</p>
      </div>
    );
  }

  if (!analysis) return <div className="p-10 text-center">Failed to load analysis.</div>;

  return (
    <div className="flex flex-col h-full bg-gray-50">
      {/* 1. VITAL SIGNS */}
      <ForensicHeader stats={analysis.stats} />

      {/* 2. THE BOARD */}
      <div className="flex-1 overflow-x-auto overflow-y-hidden p-4">
        <EvidenceBoard 
          groups={analysis.groups} 
          onReject={handleReject} 
          onOpenManual={(featureId) => setActiveManualFeature(featureId)} 
        />
      </div>

      {/* 3. DRAFT ACTION */}
      {!applicationId && onSaveDraft && (
        <div className="p-4 bg-white border-t flex justify-between items-center shadow-lg z-10">
          <span className="text-gray-600 text-sm">Review complete?</span>
          <button 
            onClick={onSaveDraft}
            className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-md font-medium transition-colors"
          >
            Save as Application
          </button>
        </div>
      )}

      {/* MODALS */}
      {activeManualFeature && (
        <ManualMatchModal 
          isOpen={!!activeManualFeature} 
          onClose={() => setActiveManualFeature(null)}
          onSubmit={handleManualSubmit}
        />
      )}
    </div>
  );
};