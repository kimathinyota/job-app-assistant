import React from 'react';

const CVSelector = ({ cvs, onSelect, selectedCVId }) => {
    // Safety check: if no CVs exist, return a message
    if (!cvs || cvs.length === 0) {
        return <p>No CVs available to select. Create one first!</p>;
    }

    const handleSelect = (cvId) => {
        if (onSelect) {
            onSelect(cvId);
        }
    };

    return (
        <div style={{ display: 'flex', gap: '15px', marginBottom: '20px', paddingBottom: '10px', borderBottom: '1px solid #ddd', overflowX: 'auto' }}>
            {cvs.map(cv => (
                <div 
                    key={cv.id} 
                    onClick={() => handleSelect(cv.id)} 
                    style={{ 
                        padding: '10px 15px', 
                        border: selectedCVId === cv.id ? '2px solid #007bff' : '1px solid #ccc', 
                        borderRadius: '5px', 
                        cursor: 'pointer', 
                        backgroundColor: selectedCVId === cv.id ? '#e6f7ff' : '#fff',
                        fontWeight: 'bold',
                        minWidth: '150px'
                    }}
                >
                    {cv.name}
                </div>
            ))}
        </div>
    );
};

export default CVSelector;