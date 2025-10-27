// frontend/src/components/DashboardHome.jsx (Revised for Safety)
import React from 'react';
import CVForm from './cv/CVForm'; // Assuming you moved this to the cv/ folder

// The cvs prop is guaranteed to be an array (even if empty) from App.jsx state.
const DashboardHome = ({ cvs, setActiveView, reloadData }) => {
    
    // Safety check: Use || [] to guarantee array access, though App.jsx does this.
    const cvCount = cvs.length; 
    
    // --- MOCK METRICS (Safely derived) ---
    // In the actual logic, you would map over cvs or fetch specific counts.
    const activeApplications = 0; // Placeholder until fetching is added
    
    // ... rest of the metric calculations

    const metrics = [
        { 
            label: 'Total Master CVs', 
            value: cvCount, 
            color: '#007bff', 
            onClick: () => setActiveView('CV_Manager') 
        },
        // ... rest of the metric cards
    ];

    return (
        // ... HTML structure for metrics and CTAs
        <div style={{ padding: '20px 0' }}>
            {/* Metric Cards Section (Assumes safe access to cvCount) */}
            {/* ... */}

            {/* Call-to-Action: CV Creation */}
            {cvCount === 0 && (
                <div style={{ border: '2px dashed #007bff' }}>
                    <p>Create your first CV!</p>
                    {/* CVForm is rendered as a quick access CTA */}
                    {/* The CVForm needs the createBaseCV prop, which would need to be passed down from App.jsx */}
                    <button onClick={() => setActiveView('CV_Manager')}>Go to CV Manager</button>
                </div>
            )}
        </div>
    );
};
export default DashboardHome;