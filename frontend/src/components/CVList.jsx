// frontend/src/components/CVList.jsx
import React from 'react';

// Displays a list of CV objects.
const CVList = ({ cvs }) => {
    if (cvs.length === 0) {
        return <p style={{ color: '#888' }}>No CVs found. Create one above to get started!</p>;
    }

    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '20px', justifyContent: 'left', margin: '20px 0' }}>
            {cvs.map((cv) => (
                <div key={cv.id} style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '5px', width: '300px', backgroundColor: '#fff', color: '#333', textAlign: 'left', boxShadow: '0 2px 5px rgba(0,0,0,0.1)' }}>
                    <strong style={{ display: 'block', fontSize: '1.2em', marginBottom: '5px' }}>{cv.name}</strong> 
                    <p style={{ fontSize: '12px', color: '#aaa', marginTop: '0' }}>ID: {cv.id}</p>
                    <p style={{ fontSize: '0.9em', color: '#555' }}>{cv.summary || "No summary provided."}</p>
                    <p style={{ fontSize: '0.8em', color: '#777' }}>
                        Updated: {new Date(cv.updated_at).toLocaleDateString()}
                    </p>
                    <button style={{ padding: '5px 10px', marginTop: '10px', cursor: 'pointer', backgroundColor: '#007bff', color: 'white', border: 'none' }}>
                        View/Edit Details
                    </button>
                </div>
            ))}
        </div>
    );
};

export default CVList;