// frontend/src/components/cv/CVList.jsx (Used as the selector component)
import React from 'react';

const CVSelector = ({ cvs, onSelect, selectedCVId }) => {
    if (cvs.length === 0) return null;

    const getStyle = (cv) => ({
        padding: '8px 12px',
        border: selectedCVId === cv.id ? '2px solid #007bff' : '1px solid #ddd',
        borderRadius: '5px',
        cursor: 'pointer',
        backgroundColor: selectedCVId === cv.id ? '#e6f7ff' : '#fff',
        fontWeight: selectedCVId === cv.id ? 'bold' : 'normal',
        transition: 'all 0.2s',
        boxShadow: selectedCVId === cv.id ? '0 0 5px rgba(0, 123, 255, 0.5)' : 'none'
    });

    return (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '20px', padding: '10px 0', borderBottom: '1px solid #eee' }}>
            <span style={{ fontWeight: 'bold', marginRight: '10px', alignSelf: 'center' }}>Select Base CV:</span>
            {cvs.map(cv => (
                <div 
                    key={cv.id} 
                    onClick={() => onSelect(cv.id)} 
                    style={getStyle(cv)}
                >
                    {cv.name}
                </div>
            ))}
        </div>
    );
};

export default CVSelector;
