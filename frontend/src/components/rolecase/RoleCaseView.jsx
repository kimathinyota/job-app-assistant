// frontend/src/components/rolecase/RoleCaseView.jsx
import React, { useState, useEffect } from 'react';
import { 
  fetchForensicAnalysis, 
  generateRoleCase, 
  rejectMatch, 
  createManualMatch,
  promoteMatch,   
  approveMatch    
} from '../../api/applicationClient';
import { FitScoreHeader } from './FitScoreHeader';
import { RequirementList } from './RequirementList';
import { EvidenceLinkerModal } from './EvidenceLinkerModal';
import CVItemPreviewModal from '../applications/CVItemPreviewModal'; // <--- Import existing modal

export const RoleCaseView = ({ applicationId, jobId, cvId, onSaveDraft }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingStage, setLoadingStage] = useState("Checking your fit...");
  
  // State for Modals
  const [activeFeature, setActiveFeature] = useState(null); // For "Add Evidence"
  const [previewItem, setPreviewItem] = useState(null);     // For "View Full Text"

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
        const response = await generateRoleCase(jobId, cvId);
        setAnalysis(response.data);
      }
    } catch (err) {
      console.error("Error:", err);
    } finally {
      setLoading(false);
    }
  };

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
    // Construct the object the Modal expects
    // We try to find the Root ID from the lineage to open the right item
    const rootItem = item.lineage && item.lineage.length > 0 ? item.lineage[0] : null;
    
    if (rootItem) {
      setPreviewItem({
        id: rootItem.id,
        type: rootItem.type,
        title: rootItem.name,
        // Pass the snippet to highlight
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
              onViewEvidence={handleViewEvidence} // <--- Pass this down
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

      {/* 4. MODALS */}
      {activeFeature && (
        <EvidenceLinkerModal 
          isOpen={!!activeFeature}
          feature={activeFeature}
          onClose={() => setActiveFeature(null)}
          onSubmit={handleManualLink}
        />
      )}

      {/* 5. PREVIEW MODAL */}
      {previewItem && (
        <CVItemPreviewModal
          isOpen={!!previewItem}
          onClose={() => setPreviewItem(null)}
          itemId={previewItem.id}
          itemType={previewItem.type}
          highlightText={previewItem.highlight} // <--- Requires update in Modal component
        />
      )}
    </div>
  );
};