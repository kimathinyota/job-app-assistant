import React from 'react';

export const ForensicHeader = ({ stats }) => {
  const { 
    overall_match_score, 
    coverage_pct, 
    critical_gaps_count, 
    evidence_sources 
  } = stats;

  const getScoreColor = (score) => {
    if (score >= 80) return "text-green-600";
    if (score >= 50) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <div className="bg-white border-b px-6 py-4 shadow-sm flex items-center justify-between shrink-0">
      
      {/* LEFT: The Big Score */}
      <div className="flex items-center space-x-4">
        <div className="text-center">
          <div className={`text-3xl font-bold ${getScoreColor(overall_match_score)}`}>
            {overall_match_score}%
          </div>
          <div className="text-xs text-gray-500 uppercase tracking-wide">Match Score</div>
        </div>
        <div className="h-10 w-px bg-gray-200 mx-4"></div>
        <div className="text-center">
          <div className="text-2xl font-semibold text-gray-700">{coverage_pct}%</div>
          <div className="text-xs text-gray-500">Coverage</div>
        </div>
      </div>

      {/* CENTER: Critical Alerts */}
      <div className={`flex items-center space-x-3 px-4 py-2 rounded-lg border ${critical_gaps_count > 0 ? 'bg-red-50 border-red-200' : 'bg-green-50 border-green-200'}`}>
        <span className={`text-2xl ${critical_gaps_count > 0 ? 'text-red-500' : 'text-green-500'}`}>
          {critical_gaps_count > 0 ? '⚠️' : '🛡️'}
        </span>
        <div>
          <div className={`font-bold ${critical_gaps_count > 0 ? 'text-red-800' : 'text-green-800'}`}>
            {critical_gaps_count} Critical Gaps
          </div>
          <div className="text-xs text-gray-600">
            {critical_gaps_count > 0 ? "Fix these to pass screening." : "You are safe from auto-rejection."}
          </div>
        </div>
      </div>

      {/* RIGHT: Authority DNA */}
      <div className="flex space-x-6 text-sm">
        <div className="text-center">
          <div className="font-bold text-blue-600">{evidence_sources.Professional || 0}</div>
          <div className="text-gray-400 text-xs">Professional</div>
        </div>
        <div className="text-center">
          <div className="font-bold text-purple-600">{evidence_sources.Academic || 0}</div>
          <div className="text-gray-400 text-xs">Academic</div>
        </div>
        <div className="text-center">
          <div className="font-bold text-orange-600">{evidence_sources.Personal || 0}</div>
          <div className="text-gray-400 text-xs">Personal</div>
        </div>
      </div>
    </div>
  );
};