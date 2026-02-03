// frontend/src/components/rolecase/FitScoreHeader.jsx
import React from 'react';
import { CheckCircle, AlertCircle } from 'lucide-react';

export const FitScoreHeader = ({ stats }) => {
  const { overall_match_score, critical_gaps_count, met_reqs, total_reqs } = stats;

  // Determine the "Vibe"
  let colorClass = "text-primary";
  let message = "Good Start";
  let bgClass = "bg-primary";

  if (overall_match_score >= 85) {
    colorClass = "text-success";
    bgClass = "bg-success";
    message = "Excellent Match";
  } else if (overall_match_score < 50) {
    colorClass = "text-danger";
    bgClass = "bg-danger";
    message = "Low Match";
  } else if (critical_gaps_count > 0) {
    colorClass = "text-warning";
    bgClass = "bg-warning";
    message = "Potential Match";
  }

  return (
    <div className="bg-white border-bottom px-4 py-4 sticky-top shadow-sm z-2">
      <div className="container d-flex align-items-center justify-content-between">
        
        {/* Left: The Score & Message */}
        <div className="d-flex align-items-center gap-4">
          <div className="position-relative d-flex justify-content-center align-items-center">
            <svg viewBox="0 0 36 36" className="d-block" style={{ width: '60px', height: '60px' }}>
              <path
                className="text-light"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="#e9ecef"
                strokeWidth="3"
              />
              <path
                className={colorClass}
                strokeDasharray={`${overall_match_score}, 100`}
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
              />
            </svg>
            <div className={`position-absolute fw-bold h5 mb-0 ${colorClass}`}>
              {Math.round(overall_match_score)}%
            </div>
          </div>
          
          <div>
            <h5 className="fw-bold mb-0 text-dark">{message}</h5>
            <p className="text-muted small mb-0">
              Matched {met_reqs} of {total_reqs} requirements
            </p>
          </div>
        </div>

        {/* Right: The Call to Action (The Gaps) */}
        <div>
          {critical_gaps_count > 0 ? (
            <div className="d-flex align-items-center gap-2 px-3 py-2 bg-danger bg-opacity-10 rounded-pill border border-danger border-opacity-25">
              <AlertCircle size={18} className="text-danger" />
              <span className="text-danger fw-bold small">{critical_gaps_count} Must-Haves Missing</span>
            </div>
          ) : (
            <div className="d-flex align-items-center gap-2 px-3 py-2 bg-success bg-opacity-10 rounded-pill border border-success border-opacity-25">
              <CheckCircle size={18} className="text-success" />
              <span className="text-success fw-bold small">Core Skills Verified</span>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};