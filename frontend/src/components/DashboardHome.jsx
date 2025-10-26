import React from 'react';
const DashboardHome = ({ cvs, setActiveView }) => (
    <div>
        <h2 style={{ color: '#333' }}>Quick Dashboard Overview</h2>
        {/* Metric Cards and CTAs will go here */}
        <p>You have **{cvs.length}** Master CVs. Go to Manager to add more structure!</p>
        <button onClick={() => setActiveView('CV_Manager')}>Go to CV Manager</button>
    </div>
);
export default DashboardHome;