// frontend/src/components/rolecase/RoleCaseView.jsx
import React, { useState, useEffect } from 'react';
import { 
  fetchForensicAnalysis, 
  generateRoleCase, 
  rejectMatch, 
  createManualMatch,
  promoteMatch,   
  approveMatch,
  triggerApplicationAnalysis // Ensure this is imported
} from '../../api/applicationClient';
import { getCurrentUser } from '../../api/authClient'; // Need user for socket
import { useJobSocket } from '../../hooks/useJobSocket'; // Socket Hook

import { FitScoreHeader } from './FitScoreHeader';
import { RequirementList } from './RequirementList';
import { EvidenceLinkerModal } from './EvidenceLinkerModal';
import CVItemPreviewModal from '../applications/CVItemPreviewModal'; 

export const RoleCaseView = ({ applicationId, jobId, cvId, onSaveDraft }) => {
  const [user, setUser] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Checking your fit...");
  
  // State for Modals
  const [activeFeature, setActiveFeature] = useState(null); 
  const [previewItem, setPreviewItem] = useState(null);     

  // --- 1. Load User ---
  useEffect(() => {
      getCurrentUser().then(res => setUser(res.data || res)).catch(console.error);
  }, []);

  // --- 2. Socket Listener ---
  useJobSocket(user?.id, (event) => {
      // CASE A: Application Mode (Result of triggerApplicationAnalysis)
      if (applicationId && event.type === 'APP_SCORED' && event.payload.app_id === applicationId) {
          console.log("App Analysis Ready!");
          // Reload full object to get charts/gaps
          fetchForensicAnalysis(applicationId)
              .then(res => setAnalysis(res.data))
              .catch(err => console.error(err))
              .finally(() => setLoading(false));
      }
      
      // CASE B: Standalone Mode (Result of generateRoleCase)
      // The payload IS the full object here because it's not saved to an App record
      if (!applicationId && event.type === 'ROLE_CASE_GENERATED' && event.payload.job_id === jobId) {
          console.log("Standalone Analysis Ready!");
          setAnalysis(event.payload);
          setLoading(false);
      }
  });

  // --- 3. Initial Data Load ---
  useEffect(() => {
    if (!user) return; // Wait for user to enable socket logic if needed

    const loadData = async () => {
      setLoading(true);
      try {
        if (applicationId) {
          // --- MODE 1: EXISTING APPLICATION ---
          setLoadingStage("Loading analysis...");
          const response = await fetchForensicAnalysis(applicationId);
          const data = response.data;

          // If analysis exists but is "empty" (score 0), it means inference hasn't run.
          // Trigger it now.
          if (!data || !data.stats || data.stats.overall_match_score === 0) {
             setLoadingStage("Running AI Analysis...");
             // Fire & Forget - We wait for socket event above
             await triggerApplicationAnalysis(applicationId); 
             // We stay in loading state until socket event fires
          } else {
             setAnalysis(data);
             setLoading(false);
          }

        } else if (jobId && cvId) {
          // --- MODE 2: STANDALONE DRAFT ---
          setLoadingStage("Generating Report...");
          // Fire & Forget - We wait for socket event 'ROLE_CASE_GENERATED'
          await generateRoleCase(jobId, cvId);
          // We stay in loading state until socket event fires
        }
      } catch (err) {
        console.error("Error loading RoleCase:", err);
        setLoading(false); 
      }
    };

    loadData();
  }, [applicationId, jobId, cvId, user]);

  // --- ACTIONS ---

  const handleReject = async (featureId) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await rejectMatch(appId, featureId);
      if (response.data.new_forensics) setAnalysis(response.data.new_forensics);
      else setAnalysis(response.data); 
    } catch (err) {
      alert("Could not reject match.");
    }
  };

  const handleManualLink = async (payload) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await createManualMatch(appId, activeFeature.requirement_id, payload);
      setAnalysis(response.data);
      setActiveFeature(null);
    } catch (err) {
      alert("Could not save match.");
    }
  };

  const handlePromote = async (featureId, altId) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await promoteMatch(appId, featureId, altId);
      setAnalysis(response.data);
    } catch (err) {
      console.error(err);
      alert("Failed to promote match.");
    }
  };

  const handleApprove = async (featureId) => {
    try {
      const appId = analysis.application_id || applicationId;
      const response = await approveMatch(appId, featureId);
      setAnalysis(response.data);
    } catch (err) {
      console.error(err);
      alert("Failed to approve match.");
    }
  };

  // --- VIEW HANDLER ---
  const handleViewEvidence = (item, specificMatchText = null) => {
    const rootItem = item.lineage && item.lineage.length > 0 ? item.lineage[0] : null;
    
    if (rootItem) {
      setPreviewItem({
        id: rootItem.id,
        type: rootItem.type,
        title: rootItem.name,
        highlight: specificMatchText || item.best_match_excerpt || item.match_summary 
      });
    } else {
      alert("No linked document found for this evidence.");
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
      <FitScoreHeader stats={analysis.stats} />

      <div className="container-fluid flex-grow-1 overflow-auto bg-light py-4">
        <div className="row justify-content-center">
          <div className="col-12 col-lg-8 col-xl-7">
            <RequirementList 
              groups={analysis.groups} 
              onReject={handleReject} 
              onLinkEvidence={(feature) => setActiveFeature(feature)}
              onPromote={handlePromote} 
              onApprove={handleApprove}
              onViewEvidence={handleViewEvidence} 
            />
          </div>
        </div>
      </div>

      {!applicationId && onSaveDraft && (
        <div className="bg-white border-top p-3 shadow-lg fixed-bottom">
          <div className="container d-flex justify-content-between align-items-center">
            <div>
              <div className="fw-bold text-dark">Ready to apply?</div>
              <div className="text-muted small">This analysis will be saved to your application.</div>
            </div>
            <button onClick={onSaveDraft} className="btn btn-primary px-4 fw-bold rounded-pill">
              Start Application
            </button>
          </div>
        </div>
      )}

      {/* MODALS */}
      {activeFeature && (
        <EvidenceLinkerModal 
          isOpen={!!activeFeature}
          feature={activeFeature}
          onClose={() => setActiveFeature(null)}
          onSubmit={handleManualLink}
        />
      )}

      {previewItem && (
        <CVItemPreviewModal
          isOpen={!!previewItem}
          onClose={() => setPreviewItem(null)}
          itemId={previewItem.id}
          itemType={previewItem.type}
          highlightText={previewItem.highlight} 
        />
      )}
    </div>
  );
};